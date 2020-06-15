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
from tile_size_utils import (
    STANDARD_RESOLUTIONS,
    SEMI_STANDARD_RESOLUTIONS,
    block_dimension, lcm,
    get_divisors
)

FLOAT16_EPS = 0.000000059605


def pretty_str(x: Union[List, Dict, Tuple]):
    if type(x) is dict:
        x = [f'{key},{val}' for key, val in x.items()]
        sep = ';'
    else:
        x = [str(item) for item in x]
        sep = ','
    return sep.join(x)


def float_equal(x: torch.Tensor, y: torch.Tensor):
    diff = torch.abs(x - y)
    return 1 - diff / (diff + torch.tensor(FLOAT16_EPS))


def getConvIds(onnx_graph):
    conv_ids = []
    for i in range(len(onnx_graph.node)):
        if onnx_graph.node[i].op_type == 'Conv':
            conv_ids.append(onnx_graph.node[i].output[0])
    return conv_ids


def make_plan(width, height, tile_width, tile_height, batches_per_step, scale_idx):
    dims = block_dimension((width, height), (tile_width, tile_height))
    num_blocks = dims[0] * dims[1]
    if num_blocks % batches_per_step != 0:
        return None
    total_loads = num_blocks // batches_per_step
    num_workers = max(get_divisors(total_loads, 16))
    num_steps = total_loads // num_workers
    return {
        'num_workers': num_workers,
        'num_steps': num_steps,
        'scale_idx': scale_idx
    }


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


class ModelWrapper(torch.nn.Module):
    def __init__(self, args):
        lcm_scales = lcm(*args.scale)
        assert args.height % lcm_scales == 0 and args.width % lcm_scales == 0, f'Both output width ({args.width}) and output height ({args.height}) must be divisible by all of the scale factors {args.scale}'
        super(ModelWrapper, self).__init__()
        self.model = Model(args, utility.checkpoint(args)).model
        self.batch_size = args.micro_batch_size
        self.batches_per_step = args.batches_per_step
        self.scales = list(sorted(args.scale))
        self.output_shape = (self.batch_size, args.height, args.width, 3)
        self.sr_pad = lcm_scales * args.pad_factor
        self.pads = [self.sr_pad // scale for scale in self.scales]
        self.metapads = []
        for scale_idx, scale in enumerate(self.scales):
            input_width, input_height = self.input_shape[1:3]
            output_width, output_height = self.output_shape[1:3]
            horizontal_pad = (input_width - output_width // scale)
            vertical_pad = (input_height - output_height // scale)
            left_pad = horizontal_pad // 2
            right_pad = horizontal_pad - left_pad
            top_pad = vertical_pad // 2
            bottom_pad = vertical_pad - top_pad
            self.metapads.append(
                (
                    top_pad - self.pads[scale_idx],
                    bottom_pad - self.pads[scale_idx],
                    left_pad - self.pads[scale_idx],
                    right_pad - self.pads[scale_idx]
                )
            )

        if not self.is_edsr:
            self.head = MDSRHead(self.model)
            self.tail = MDSRTail(self.model, self.sr_pad)
            self.upsamplers = []
            for metapad, upsampler in zip(self.metapads, self.model.upsample):
                self.upsamplers.append(UpsamplerWrapper(upsampler, metapad))

        dir = os.path.split(args.pre_train)[0]
        prefix = f'{args.model.lower()}-x{pretty_str(args.scale)}-r{args.n_resblocks}c{args.n_feats}-w{self.input_shape[2]}h{self.input_shape[1]}'
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
            return (torch.randn(*self.input_shape),)
        else:
            return (
                torch.randn(*self.input_shape),
                torch.tensor([np.random.choice(self.scales)]).type(torch.float)
            )

    @property
    def plans(self):
        sizes = {**STANDARD_RESOLUTIONS, **SEMI_STANDARD_RESOLUTIONS}
        plans = dict()
        scale_candidates = {name: [] for name in sizes}
        for scale_idx, scale in enumerate(self.scales):
            tile_width, tile_height = self.tile_width(scale_idx), self.tile_height(scale_idx)
            for name, (width, height) in sizes.items():
                if width % tile_width == 0 and height % tile_height == 0:
                    scale_candidates[name].append(scale)
        for name, scales in scale_candidates.items():
            if scales:
                width, height = sizes.get(name)
                scale = min(scales)
                scale_idx = self.scales.index(scale)
                tile_width = self.tile_width(scale_idx)
                tile_height = self.tile_height(scale_idx)
                plan = make_plan(width, height, tile_width, tile_height, self.batches_per_step, scale_idx)
                if plan is not None:
                    plans[name] = plan
        # if self.is_edsr:
        #     tile_width, tile_height = self.tile_width(0), self.tile_height(0)
        #     for name, (width, height) in sizes.items():
        #         if width % tile_width == 0 and height % tile_height == 0:
        #             scale = self.scales[0]
        #             plan = make_plan(width, height, tile_width, tile_height, self.batches_per_step, scale)
        #             if plan is not None:
        #                 plans[name] = plan
        # else:
        #     scale_candidates = {name: [] for name in sizes}
        #     for scale_idx, scale in enumerate(self.scales):
        #         tile_width, tile_height = self.tile_width(scale_idx), self.tile_height(scale_idx)
        #         for name, (width, height) in sizes.items():
        #             if width % tile_width == 0 and height % tile_height == 0:
        #                 scale_candidates[name].append(scale)
        #     for name, scales in scale_candidates.items():
        #         if scales:
        #             width, height = sizes.get(name)
        #             scale = min(scales)
        #             scale_idx = self.scales.index(scale)
        #             tile_width = self.tile_width(scale_idx)
        #             tile_height = self.tile_height(scale_idx)
        #             plan = make_plan(width, height, tile_width, tile_height, self.batches_per_step, scale)
        #             if plan is not None:
        #                 plans[name] = plan
        return plans

    @property
    def config(self):
        cfg = {
            'pads': self.pads,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'scales': self.scales,
            'plans': self.plans
        }
        if not self.is_edsr:
            cfg.update({'metapads': self.metapads})
        return cfg

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
        'border_type': 'REFLECT101'
    }
    config.update(wrapper.config)

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
    from option import args as arguments

    export_onnx_model(arguments)
