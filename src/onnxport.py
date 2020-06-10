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

FLOAT16_EPS = 0.000000059605

def gcd(a, b):
    r = a % b
    return b if r == 0 else gcd(b, r)

def lcm(*numbers):
    assert len(numbers) > 0
    if len(numbers) == 1:
        return numbers[0]
    else:
        return reduce(lambda x, y: x * y, numbers) // reduce(gcd, numbers)

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
        self.scale = torch.tensor([upsampler[1].upscale_factor], dtype=torch.float)
        self.unpad_left, self.unpad_right, self.unpad_top, self.unpad_bottom = unpad

    def forward(self, x):
        if self.unpad_left > 0:
            x = x[:, :, self.unpad_left:-self.unpad_right, :]
        if self.unpad_top > 0:
            x = x[:, :, :, self.unpad_top:-self.unpad_bottom]
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

def float_equal(x: torch.Tensor, y: torch.Tensor):
    diff = torch.abs(x - y)
    return 1 - diff / (diff + torch.tensor(FLOAT16_EPS))

class ModelWrapper(torch.nn.Module):
    def __init__(self, args):
        lcm_scales = lcm(*args.scale)
        assert args.height % lcm_scales == 0 and args.width % lcm_scales == 0, f'Both output width ({args.width}) and output height ({args.height}) must be divisible by all of the scale factors {args.scale}'
        super(ModelWrapper, self).__init__()
        self.model = Model(args, utility.checkpoint(args)).model
        self.batch_size = args.micro_batch_size
        self.scales = list(sorted(args.scale))
        self.output_shape = (self.batch_size, args.height, args.width, 3)
        self.sr_pad = lcm_scales * args.pad_factor
        self.pads = []
        for scale in self.scales:
            input_width, input_height = self.input_shape[1:3]
            output_width, output_height = self.output_shape[1:3]
            horizontal_pad = (input_width - output_width // scale)
            vertical_pad = (input_height - output_height // scale)
            left_pad = horizontal_pad // 2
            right_pad = horizontal_pad - left_pad
            top_pad = vertical_pad // 2
            bottom_pad = vertical_pad - top_pad
            self.pads.append((top_pad, bottom_pad, left_pad, right_pad))

        if not self.is_edsr:
            self.head = MDSRHead(self.model)
            self.tail = MDSRTail(self.model, self.sr_pad)
            unpads = []
            for scale_idx, scale in enumerate(self.scales):
                unpads.append([pad - self.sr_pad // scale for pad in self.pads[scale_idx]])
            self.upsamplers = []
            for unpad, upsampler in zip(unpads, self.model.upsample):
                self.upsamplers.append(UpsamplerWrapper(upsampler, unpad))

        dir = os.path.split(args.pre_train)[0]
        prefix = f'{args.model.lower()}-x{pretty_str(args.scale)}r{args.n_resblocks}c{args.n_feats}-w{self.input_shape[2]}h{self.input_shape[1]}'
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
        min_scale = min(self.scales)
        return (
            self.batch_size,
            (self.output_shape[1] + 2 * self.sr_pad) // min_scale,
            (self.output_shape[2] + 2 * self.sr_pad) // min_scale,
            3
        )

    def tile_height(self, scale_idx):
        return self.output_shape[1] // self.scales[scale_idx]

    def tile_width(self, scale_idx):
        return self.output_shape[2] // self.scales[scale_idx]

    @property
    def dummy_input(self):
        if self.is_edsr:
            return (torch.randn(*self.input_shape), )
        else:
            return (
                torch.randn(*self.input_shape),
                torch.tensor([np.random.choice(self.scales)]).type(torch.float)
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

    def forward(self, x: torch.Tensor, scale: torch.Tensor = 0):
        if self.is_edsr:
            x = self.model(x.permute((0, 3, 1, 2))).permute((0, 2, 3, 1)).clamp(0, 255)
            if self.sr_pad > 0:
                x = x[:, self.sr_pad:-self.sr_pad, self.sr_pad:-self.sr_pad, :]
            return x
        else:
            x = self.head(x)
            upsample_features = []
            for upsampler in self.upsamplers:
                upsample_features.append(float_equal(scale, upsampler.scale) * upsampler(x))
            x = upsample_features[0]
            for i in range(1, len(upsample_features)):
                x += upsample_features[i]
            return self.tail(x)


def make_plans(wrapper: ModelWrapper):
    sizes = {'1080p': (1920, 1080), '720p': (1280, 720)}
    plans = dict()
    if wrapper.is_edsr:
        tile_width, tile_height = wrapper.tile_width(0), wrapper.tile_height(0)
        for name, (width, height) in sizes.items():
            if width % tile_width == 0 and height % tile_height == 0:
                plans[name] = wrapper.scales[0]
    else:
        scale_candidates = {name: [] for name in sizes}
        for scale_idx, scale in enumerate(wrapper.scales):
            tile_width, tile_height = wrapper.tile_width(scale_idx), wrapper.tile_height(scale_idx)
            for name, (width, height) in sizes.items():
                if width % tile_width == 0 and height % tile_height == 0:
                    scale_candidates[name].append(scale)
        for name, scales in scale_candidates.items():
            if scales:
                plans[name] = max(scales)
    return plans


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
    plans = make_plans(wrapper)

    config = {
        'num_ipus': 1,
        'batches_per_step': args.batches_per_step,
        'conv_ids': conv_ids,
        'conv_mem_portion': args.conv_mem_portion,
        'pads': wrapper.pads,
        'sr_pad': wrapper.sr_pad,
        'input_shape': wrapper.input_shape,
        'output_shape': wrapper.output_shape,
        'scales': wrapper.scales,
        'plans': plans,
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
