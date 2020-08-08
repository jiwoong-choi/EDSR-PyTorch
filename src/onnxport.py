import json
import os
from typing import Union, List, Dict, Tuple

import numpy as np
import onnx
import onnxsim
import torch

import utility
from model import Model
from model.edsr import EDSR
from model.mdsr import MDSR
from tile_size_utils import (
    STANDARD_RESOLUTIONS,
    SEMI_STANDARD_RESOLUTIONS,
    OTHER_RESOLUTIONS,
    block_dimension, lcm
)

FLOAT16_EPS = 0.000000059605
RESOLUTIONS = {**STANDARD_RESOLUTIONS, **SEMI_STANDARD_RESOLUTIONS, **OTHER_RESOLUTIONS}
RESOLUTIONS = {key: value for key, value in RESOLUTIONS.items() if value[0] < 2000}


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


def getConvIds(onnx_path):
    onnx_graph = onnx.load(onnx_path).graph
    conv_ids = []
    for node in onnx_graph.node:
        if node.op_type == 'Conv':
            conv_ids.extend(node.output)
    return conv_ids


def make_plan(width, height, tile_width, tile_height, batches_per_step, scale_idx):
    dims = block_dimension((width, height), (tile_width, tile_height))
    num_blocks = dims[0] * dims[1]
    if num_blocks % batches_per_step != 0:
        return None
    return scale_idx


class ModelWrapper(torch.nn.Module):
    def __init__(self, args):
        lcm_scales = lcm(*args.scale)
        assert args.height % lcm_scales == 0 and args.width % lcm_scales == 0, f'Both output width ({args.width}) and output height ({args.height}) must be divisible by all of the scale factors {args.scale}'
        super(ModelWrapper, self).__init__()
        self.model = Model(args, utility.checkpoint(args)).model
        assert self.model.__class__ in (EDSR, MDSR)
        self.batch_size = args.micro_batch_size
        self.batches_per_step = args.batches_per_step
        self.scales = list(sorted(args.scale))
        self.output_shape = (self.batch_size, args.height, args.width, 3)
        self.sr_pad = lcm_scales * args.pad_factor
        self.pads = [self.sr_pad // scale for scale in self.scales]

        dir = os.path.split(args.pre_train)[0]
        prefix = f'{args.model.lower()}-x{pretty_str(args.scale)}-r{args.n_resblocks}c{args.n_feats}-w{self.input_shape[2]}h{self.input_shape[1]}'
        if args.micro_batch_size > 1:
            prefix += f'mb{args.micro_batch_size}'
        if args.opset_version != 10:
            prefix += f'op{args.opset_version}'
        if args.pre_train == 'download':
            download_dir = '../../models/download'
            os.makedirs(download_dir, exist_ok=True)
            self.model_prefix = os.path.join(download_dir, prefix)
        else:
            self.model_prefix = os.path.join(dir, prefix)
        self.opset_version = args.opset_version

    @property
    def onnx_path(self):
        return self.model_prefix + '.onnx'

    @property
    def config_path(self):
        return self.model_prefix + '.json'

    @property
    def is_edsr(self):
        return isinstance(self.model, EDSR)

    @property
    def is_mdsr(self):
        return isinstance(self.model, MDSR)

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

        if self.is_mdsr:
            return (
                torch.randn(*self.input_shape),
                torch.tensor([np.random.choice(self.scales)]).type(torch.float)
            )

    @property
    def scale_candidates(self):
        scale_candidates = dict()
        for scale_idx, scale in enumerate(self.scales):
            tile_width, tile_height = self.tile_width(scale_idx), self.tile_height(scale_idx)
            for name, (width, height) in RESOLUTIONS.items():
                if width % tile_width == 0 and height % tile_height == 0:
                    num_tiles = (width // tile_width) * (height // tile_height)
                    if num_tiles % (self.batches_per_step * self.batch_size) == 0:
                        if name in scale_candidates:
                            scale_candidates[name].append(scale)
                        else:
                            scale_candidates[name] = [scale]
        return scale_candidates

    @property
    def default_scale_map(self):
        plans = dict()
        for name, scales in self.scale_candidates.items():
            if scales:
                plans[name] = min(scales)
        return plans

    @property
    def immutable_config(self):
        cfg = {
            'pads': self.pads,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'scales': self.scales,
            'conv_ids': getConvIds(self.onnx_path)
        }
        return cfg

    @property
    def mutable_config(self):
        return {
            'num_ipus': 1,
            'num_replica': 1,
            'macro_batch_size': 1,
            'conv_mem_portion': 0.15,
            'border_type': 'REFLECT101',
            'batches_per_step': self.batches_per_step,
            'scale_map': self.default_scale_map
        }

    @property
    def input_names(self):
        if self.is_edsr:
            return ['lr']

        if self.is_mdsr:
            return ['lr', 'scale']

    @property
    def input_shape_dict(self):
        if self.is_edsr:
            return {'lr': self.input_shape}

        if self.is_mdsr:
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

        if self.is_mdsr:
            x = self.model.sub_mean(x.permute((0, 3, 1, 2)))
            x = self.model.head(x)
            x = self.model.pre_process[0](x)

            x = self.model.body(x) + x

            upsample_features = []
            for upsampler in self.model.upsample:
                weight = float_equal(scale, upsampler[1].upscale_factor)
                feat = upsampler(x)
                feat = self.model.tail(feat)
                feat = self.model.add_mean(feat)

                feat = feat.permute((0, 2, 3, 1)).clamp(0, 255)
                if self.sr_pad > 0 or feat.shape[1:3] != self.output_shape[1:3]:
                    feat = feat[:, self.sr_pad:self.sr_pad + self.output_shape[1],
                           self.sr_pad:self.sr_pad + self.output_shape[2], :]
                upsample_features.append(weight * feat)
            x = upsample_features[0]
            for i in range(1, len(upsample_features)):
                x += upsample_features[i]
            return x


def export_onnx_model(args):
    wrapper = ModelWrapper(args)

    print(f'Dumping ONNX protobuf (opset {wrapper.opset_version}) at {wrapper.onnx_path} ...')
    torch.onnx.export(
        wrapper,
        wrapper.dummy_input,
        wrapper.onnx_path,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
        opset_version=wrapper.opset_version
    )

    print(f'Dumping model config at {wrapper.config_path} ...')
    with open(wrapper.config_path, 'w') as fp:
        json.dump(
            {
                'mutable': wrapper.mutable_config,
                'immutable': wrapper.immutable_config
            },
            fp,
            sort_keys=True,
            indent=2
        )

    if not args.skip_simplify:
        print('Simplifying ONNX protobuf ...')
        simplified_onnx_model, check_ok = onnxsim.simplify(
            wrapper.onnx_path, check_n=args.check_n, perform_optimization=not args.skip_optimization,
            skip_fuse_bn=not args.enable_fuse_bn, input_shapes=wrapper.input_shape_dict
        )

        if check_ok:
            print(f'Dumping simplified ONNX protobuf at {wrapper.onnx_path} ...')
            onnx.save(simplified_onnx_model, wrapper.onnx_path)
        else:
            print(f'Simplified model failed the checking with random input tensor')
            print(f'Keeping the original ONNX protobuf')


if __name__ == '__main__':
    from option import args as arguments

    export_onnx_model(arguments)
