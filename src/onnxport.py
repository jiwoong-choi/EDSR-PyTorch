import os
import json

import onnx
import onnxsim
import torch

import model
import utility
from option import args


class ModelWrapper(torch.nn.Module):
    def __init__(self, submodule, tile_width, tile_height, pad, batch_size):
        super(ModelWrapper, self).__init__()
        self.submodule = submodule
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.pad = pad
        self.batch_size = batch_size

    @property
    def input_shape(self):
        return (self.batch_size, self.tile_height + 2 * self.pad, self.tile_width + 2 * self.pad, 3)

    def forward(self, x: torch.Tensor, idx_scale):
        x = self.submodule(x.permute((0, 3, 1, 2)), idx_scale).permute((0, 2, 3, 1)).clamp(0, 255)
        if self.pad > 0:
            x = x[:, 2*self.pad:-2*self.pad, 2*self.pad:-2*self.pad, :]
        return x

def getConvIds(onnx_graph):
    conv_ids = []
    for i in range(len(onnx_graph.node)):
        if onnx_graph.node[i].op_type == 'Conv':
            conv_ids.append(onnx_graph.node[i].output[0])
    return conv_ids

if __name__ == '__main__':
    checkpoint = utility.checkpoint(args)
    module = ModelWrapper(
        model.Model(args, checkpoint),
        args.width,
        args.height,
        args.pad,
        args.onnx_batch_size
    )

    dir = os.path.split(args.pre_train)[0]
    filename = f'{args.model.lower()}-x{args.scale[0]}-r{args.n_resblocks}c{args.n_feats}_h{module.tile_height}w{module.tile_width}'
    if module.pad > 0:
        filename += f'p{module.pad}'
    if module.batch_size > 1:
        filename += f'_bs{module.batch_size}'
    if args.opset_version != 10:
        filename += f'_opset{args.opset_version}'
    filename += '.onnx'
    onnx_path = os.path.join(dir, filename)

    print(f'input shape: {module.input_shape}')
    dummy_input = torch.randn(*module.input_shape)

    print(f'Dumping ONNX protobuf (opset {args.opset_version}) at {onnx_path} ...')
    torch.onnx.export(
        module,
        (
            dummy_input,
            torch.tensor(args.scale, dtype=torch.int)
        ),
        onnx_path,
        opset_version=args.opset_version
    )

    onnx_model = onnx.load(onnx_path)
    config_path = os.path.splitext(onnx_path)[0] + '.json'

    print(f'Dumping ONNX config file at {config_path} ...')
    with open(config_path, 'w') as fp:
        json.dump(
            {
                'num_ipus': 1,
                'batches_per_step': 9,
                'conv_ids': getConvIds(onnx_model.graph),
                'conv_mem_portion': args.conv_mem_portion,
                'pad': args.pad,
                'input_shape': module.input_shape,
                'scale': args.scale[0]
            },
            fp, indent=2
        )

    if not args.skip_simplify:
        print('Simplifying ONNX protobuf ...')
        simplified_onnx_model, check_ok = onnxsim.simplify(
            onnx_path, check_n=args.check_n, perform_optimization=not args.skip_optimization,
            skip_fuse_bn=not args.enable_fuse_bn, input_shapes={'0': module.input_shape}
        )

        if check_ok:
            print(f'Dumping simplified ONNX protobuf at {onnx_path} ...')
            onnx.save(simplified_onnx_model, onnx_path)
        else:
            print(f'Simplified model failed the checking with random input tensor')
            print(f'Keeping the original ONNX protobuf')
