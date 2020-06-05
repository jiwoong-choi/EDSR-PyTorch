import json
import os
from functools import reduce
from typing import Union, List, Dict, Tuple

import onnx
import onnxsim
import torch

import model
import utility
from model.common import Upsampler
from model.mdsr import MDSR
from option import args


def gcd(a, b):
    r = a % b
    if r == 0:
        return b
    else:
        return gcd(b, r)


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


def getConvIds(onnx_graph):
    conv_ids = []
    for i in range(len(onnx_graph.node)):
        if onnx_graph.node[i].op_type == 'Conv':
            conv_ids.append(onnx_graph.node[i].output[0])
    return conv_ids


def pretty_str(x: Union[List, Dict, Tuple]):
    if type(x) is dict:
        x = [f'{key},{val}' for key, val in x.items()]
        sep = ';'
    else:
        x = [str(item) for item in x]
        sep = ','
    return sep.join(x)


if __name__ == '__main__':
    args.model = 'MDSR'

    # Unpack params from argument
    output_width = args.width
    output_height = args.height
    base_pad = args.pad
    scales = list(sorted(args.scale))
    conv_mem_portion = args.conv_mem_portion
    micro_batch_size = args.micro_batch_size
    batches_per_step = args.batches_per_step

    # Derived params
    output_shape = (micro_batch_size, output_height, output_width, 3)
    sr_pad = base_pad * reduce(lambda x, y: x * y, scales) // reduce(gcd, scales)
    pad = dict()
    tile_width = dict()
    tile_height = dict()
    for scale in scales:
        pad[scale] = sr_pad // scale
        tile_width[scale] = output_width // scale
        tile_height[scale] = output_height // scale

    input_shapes = {
        scale: (
            micro_batch_size,
            tile_height[scale],
            tile_width[scale],
            3
        ) for scale in scales
    }
    onnx_input_shape = (
        micro_batch_size,
        tile_height[scales[0]] + 2 * pad[scales[0]],
        tile_width[scales[0]] + 2 * pad[scales[0]],
        3
    )

    unpad = dict()
    for scale in scales:
        unpad_x = (onnx_input_shape[2] - (tile_width[scale] + 2 * pad[scale])) // 2
        unpad_y = (onnx_input_shape[1] - (tile_height[scale] + 2 * pad[scale])) // 2
        unpad[scale] = (unpad_y, unpad_x)

    # Set up output prefix
    dir = os.path.split(args.pre_train)[0]
    filename = f'{args.model.lower()}-x{pretty_str(scales)}-r{args.n_resblocks}c{args.n_feats}w{onnx_input_shape[2]}h{onnx_input_shape[1]}'
    if micro_batch_size > 1:
        filename += f'mb{micro_batch_size}'
    if args.opset_version != 10:
        filename += f'op{args.opset_version}'
    model_prefix = os.path.join(dir, filename)

    mdsr_model = model.Model(args, utility.checkpoint(args)).model
    modules = {
        'head': MDSRHead(mdsr_model),
        'tail': MDSRTail(mdsr_model, sr_pad)
    }
    for upsampler in mdsr_model.upsample:
        scale = upsampler[1].upscale_factor
        modules[f'upsample_x{scale}'] = UpsamplerWrapper(upsampler, unpad[scale])

    input_scale = 3
    dummy_input = torch.randn(*onnx_input_shape)
    print(f'input shape: {dummy_input.shape}')
    head_output = modules.get('head')(dummy_input)
    print(f'head output shape: {head_output.shape}')
    upsample_outputs = []
    for scale in args.scale:
        upsample_output = modules.get(f'upsample_x{scale}')(head_output)
        print(f'upsample_x{scale} output shape: {upsample_output.shape}')
        upsample_outputs.append(upsample_output)
    assert len(set(output.shape for output in upsample_outputs)) == 1
    final_output = modules.get('tail')(upsample_outputs[0])
    assert final_output.shape == output_shape

    input_tensors = {
        'head': dummy_input,
        'tail': upsample_outputs[0]
    }
    for scale in args.scale:
        input_tensors[f'upsample_x{scale}'] = head_output

    config_path = model_prefix + '.json'

    print(f'Dumping MDSR config file at {config_path} ...')
    with open(config_path, 'w') as fp:
        json.dump(
            {
                'num_ipus': 1,
                'batches_per_step': args.batches_per_step,
                'conv_mem_portion': args.conv_mem_portion,
                'pad': list(pad.values()),
                'input_shapes': list(input_shapes.values()),
                'onnx_input_shape': onnx_input_shape,
                'scales': scales,
                'border_type': 'REFLECT101'
            },
            fp,
            indent=2,
            sort_keys=True
        )

    for module_name in modules.keys():
        onnx_path = model_prefix + f'_{module_name}.onnx'
        print(f'Dumping ONNX protobuf (opset {args.opset_version}) at {onnx_path} ...')
        torch.onnx.export(
            modules.get(module_name),
            input_tensors.get(module_name),
            onnx_path,
            opset_version=args.opset_version
        )

        onnx_model = onnx.load(onnx_path)

        if not args.skip_simplify:
            print('Simplifying ONNX protobuf ...')
            simplified_onnx_model, check_ok = onnxsim.simplify(
                onnx_path, check_n=args.check_n, perform_optimization=not args.skip_optimization,
                skip_fuse_bn=not args.enable_fuse_bn, input_shapes={}
            )

            if check_ok:
                print(f'Dumping simplified ONNX protobuf at {onnx_path} ...')
                onnx.save(simplified_onnx_model, onnx_path)
            else:
                print(f'Simplified model failed the checking with random input tensor')
                print(f'Keeping the original ONNX protobuf')
