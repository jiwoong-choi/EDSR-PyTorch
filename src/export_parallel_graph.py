import json
import os
from typing import List

import onnx
import onnxsim
import torch

from onnxport import ModelWrapper, getConvIds

NUM_WORKERS = 16


def getVirtualGraphMapping(onnx_path):
    onnx_graph = onnx.load(onnx_path).graph
    virtual_graph_mapping = dict()
    branch_id = 0
    for node in onnx_graph.node:
        if f'lr{branch_id + 1}' in node.input:
            branch_id += 1
        if len(node.input) > 0:
            for output in node.output:
                virtual_graph_mapping[output] = branch_id

    return virtual_graph_mapping


class ParallelModel(torch.nn.Module):
    def __init__(self, args):
        super(ParallelModel, self).__init__()

        self.branches = []
        for i in range(NUM_WORKERS):
            self.branches.append(ModelWrapper(args))
        for branch in self.branches:
            for p in branch.parameters():
                p.requires_grad = False

        self.is_edsr = self.branches[0].is_edsr
        if self.is_edsr:
            self.dummy_input = (
                [self.branches[0].dummy_input[0]] * NUM_WORKERS,
                [torch.tensor([[0]]).type(torch.float)] * NUM_WORKERS
            )
        else:
            dummy_lr_tile, dummy_scale = self.branches[0].dummy_input
            self.dummy_input = ([dummy_lr_tile] * NUM_WORKERS, [dummy_scale] * NUM_WORKERS)
        model_prefix = self.branches[0].model_prefix
        dir, prefix = os.path.split(model_prefix)
        self.model_prefix = os.path.join(dir, 'parallel-' + prefix)
        self.input_names = [f'lr{i}' for i in range(NUM_WORKERS)] + [f'scale{i}' for i in range(NUM_WORKERS)]
        self.output_names = [f'sr{i}' for i in range(NUM_WORKERS)]
        self.opset_version = self.branches[0].opset_version
        self.mutable_config = self.branches[0].mutable_config
        self.mutable_config['num_ipus'] = NUM_WORKERS
        base_shape_dict = self.branches[0].input_shape_dict
        lr_shape_dict = {f'lr{i}': base_shape_dict.get('lr') for i in range(NUM_WORKERS)}
        scale_shape_dict = {f'scale{i}': base_shape_dict.get('scale') for i in range(NUM_WORKERS)}
        self.input_shape_dict = lr_shape_dict if self.is_edsr else {**lr_shape_dict, **scale_shape_dict}

    @property
    def immutable_config(self):
        cfg = {
            'pads': self.branches[0].pads,
            'input_shape': self.branches[0].input_shape,
            'output_shape': self.branches[0].output_shape,
            'scales': self.branches[0].scales
        }
        if not self.is_edsr:
            cfg.update({'metapads': self.branches[0].metapads})
        return cfg

    def forward(self, lr_tiles: List[torch.Tensor], scales: List[torch.Tensor]):
        sr_tiles = []
        for lr_tile, scale, branch in zip(lr_tiles, scales, self.branches):
            sr_tiles.append(branch(lr_tile, scale))
        return sr_tiles


def export_onnx_model(args):
    wrapper = ParallelModel(args)
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

    immutable_config = {
        'conv_ids': getConvIds(onnx_path),
        'virtual_graph_mapping': getVirtualGraphMapping(onnx_path)
    }
    mutable_config = {
        'num_ipus': 1,
        'batches_per_step': args.batches_per_step,
        'conv_mem_portion': args.conv_mem_portion,
        'border_type': 'REFLECT101'
    }
    immutable_config.update(wrapper.immutable_config)
    mutable_config.update(wrapper.mutable_config)
    config = {
        'mutable': mutable_config,
        'immutable': immutable_config
    }

    print(f'Dumping model config at {config_path} ...')
    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=2)

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
