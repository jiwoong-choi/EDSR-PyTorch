import json
import os
from functools import reduce
from typing import Union, List, Dict, Tuple

import numpy as np
import onnx
import onnxsim
import torch

import utility
from model import Model
from model.common import Upsampler
from model.edsr import EDSR
from model.mdsr import MDSR


def gcd(a, b):
    r = a % b
    return b if r == 0 else gcd(b, r)


def pretty_str(x: Union[List, Dict, Tuple]):
    if type(x) is dict:
        x = [f'{key},{val}' for key, val in x.items()]
        sep = ';'
    else:
        x = [str(item) for item in x]
        sep = ','
    return sep.join(x)


def getConvIds(onnx_graph):
    conv_ids = []
    for i in range(len(onnx_graph.node)):
        if onnx_graph.node[i].op_type == 'Conv':
            conv_ids.append(onnx_graph.node[i].output[0])
    return conv_ids


class MDSRHead(torch.nn.Module):
    def __init__(self, mdsr: MDSR):
        super(MDSRHead, self).__init__()
        self.sub_mean = mdsr.sub_mean
        self.head = mdsr.head
        self.pre_process = mdsr.pre_process[0]
        self.body = mdsr.body

    def forward(self, x):
        x = self.sub_mean(x.permute((0, 3, 1, 2)))
        x = self.head(x)
        x = self.pre_process(x)

        return self.body(x) + x


class UpsamplerWrapper(torch.nn.Module):
    def __init__(self, upsampler: Upsampler, unpad):
        super(UpsamplerWrapper, self).__init__()
        self.upsampler = upsampler
        self.scale = upsampler[1].upscale_factor
        if unpad is not None:
            self.unpad = torch.tensor(unpad)

    def forward(self, x):
        if self.unpad[0] > 0:
            x = x[:, :, self.unpad[0]:-self.unpad[0], :]
        if self.unpad[1] > 0:
            x = x[:, :, :, self.unpad[1]:-self.unpad[1]]
        return self.upsampler(x)


class MDSRTail(torch.nn.Module):
    def __init__(self, mdsr: MDSR, sr_pad):
        super(MDSRTail, self).__init__()
        self.tail = mdsr.tail
        self.add_mean = mdsr.add_mean
        self.sr_pad = torch.tensor([sr_pad])

    def forward(self, x):
        x = self.tail(x)
        x = self.add_mean(x)

        x = x.permute((0, 2, 3, 1)).clamp(0, 255)
        if self.sr_pad > 0:
            x = x[:, self.sr_pad:-self.sr_pad, self.sr_pad:-self.sr_pad, :]

        return x


class ModelWrapper(torch.nn.Module):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.model = Model(args, utility.checkpoint(args)).model
        self.batch_size = args.micro_batch_size
        self.scales = list(sorted(args.scale))
        self.output_shape = (self.batch_size, args.height, args.width, 3)
        self.sr_pad = self.scales[0] * args.pad_factor if args.model == 'EDSR' else \
            args.pad_factor * reduce(lambda x, y: x * y, self.scales) // reduce(gcd, self.scales)
        self.pads = [self.sr_pad // scale for scale in self.scales]

        if not self.is_edsr:
            self.head = MDSRHead(self.model)
            self.tail = MDSRTail(self.model, self.sr_pad)
            unpads = []
            h, w = self.input_shape[1:3]
            H, W = self.output_shape[1:3]
            for scale_idx, scale in enumerate(self.scales):
                unpad_x = (w - (W // scale + 2 * self.pads[scale_idx])) // 2
                unpad_y = (h - (H // scale + 2 * self.pads[scale_idx])) // 2
                unpads.append((unpad_y, unpad_x))
            self.upsamplers = []
            for unpad, upsampler in zip(unpads, self.model.upsample):
                self.upsamplers.append(UpsamplerWrapper(upsampler, unpad))

        dir = os.path.split(args.pre_train)[0]
        prefix = f'{args.model.lower()}-x{args.scale[0]}r{args.n_resblocks}c{args.n_feats}-w{self.input_shape[2]}h{self.input_shape[1]}'
        if args.micro_batch_size > 1:
            prefix += f'mb{args.micro_batch_size}'
        if args.opset_version != 10:
            prefix += f'op{args.opset_version}'
        self.model_prefix = os.path.join(dir, prefix)
        self.opset_version = args.opset_version

    @property
    def is_edsr(self):
        return isinstance(self.model, EDSR)

    @property
    def input_shape(self):
        return (
            self.batch_size,
            (args.height + 2 * self.sr_pad) // self.scales[0],
            (args.width + 2 * self.sr_pad) // self.scales[0],
            3
        )

    @property
    def dummy_input(self):
        return (
            torch.randn(*self.input_shape),
            torch.tensor([np.random.choice(self.scales)])
        )

    @property
    def input_names(self):
        if self.is_edsr:
            return ['lr']
        else:
            return ['lr', 'scale']

    @property
    def input_shape_dict(self):
        if self.is_edsr:
            return {'lr': self.input_shape}
        else:
            return {
                'lr': self.input_shape,
                'scale': (1,)
            }

    @property
    def output_names(self):
        return ['sr']

    def forward(self, x: torch.Tensor, scale: torch.Tensor):
        if self.is_edsr:
            x = self.model(x.permute((0, 3, 1, 2))).permute((0, 2, 3, 1)).clamp(0, 255)
            if self.pads[0] > 0:
                x = x[:, self.scales[0] * self.pads[0]:-self.scales[0] * self.pads[0],
                    self.scales[0] * self.pads[0]:-self.scales[0] * self.pads[0], :]
            return x
        else:
            x = self.head(x)
            upsample_features = []
            for upsampler in self.upsamplers:
                upsample_features.append((scale == upsampler.scale) * upsampler(x))
            x = torch.zeros_like(upsample_features[0])
            for upsample_feature in upsample_features:
                x += upsample_feature
            return self.tail(x)


def export_onnx_model(args):
    wrapper = ModelWrapper(args)
    onnx_path = wrapper.model_prefix + '.onnx'
    config_path = wrapper.model_prefix + '.json'

    print(f'Dumping ONNX protobuf (opset {wrapper.opset_version}) at {onnx_path} ...')
    torch.onnx.export(
        wrapper,
        wrapper.dummy_input,
        onnx_path,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
        opset_version=wrapper.opset_version
    )

    onnx_model = onnx.load(onnx_path)
    conv_ids = getConvIds(onnx_model.graph)

    config = {
        'num_ipus': 1,
        'batches_per_step': args.batches_per_step,
        'conv_ids': conv_ids,
        'conv_mem_portion': args.conv_mem_portion,
        'pads': wrapper.pads,
        'input_shape': wrapper.input_shape,
        'output_shape': wrapper.output_shape,
        'scales': wrapper.scales,
        'border_type': 'REFLECT101'
    }
    print(f'Dumping model config at {config_path} ...')
    with open(config_path, 'w') as fp:
        json.dump(config, fp, sort_keys=True, indent=2)

    if not args.skip_simplify:
        print('Simplifying ONNX protobuf ...')
        simplified_onnx_model, check_ok = onnxsim.simplify(
            onnx_path, check_n=args.check_n, perform_optimization=not args.skip_optimization,
            skip_fuse_bn=not args.enable_fuse_bn, input_shapes=wrapper.input_shape_dict
        )

        if check_ok:
            print(f'Dumping simplified ONNX protobuf at {onnx_path} ...')
            onnx.save(simplified_onnx_model, onnx_path)
        else:
            print(f'Simplified model failed the checking with random input tensor')
            print(f'Keeping the original ONNX protobuf')


if __name__ == '__main__':
    from option import args
    export_onnx_model(args)
