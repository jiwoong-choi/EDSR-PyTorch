import os
import json

import onnx
import onnxsim
import torch

import model
import utility
from model.mdsr import MDSR
from option import args

class SimplePixelShuffle(torch.nn.Module):
    def __init__(self, scale):
        super(SimplePixelShuffle, self).__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1, self.scale, self.scale, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(x.shape[0], -1, x.shape[2] * self.scale, x.shape[4] * self.scale)
        return x


class MDSRWrapper(torch.nn.Module):
    def __init__(self, mdsr: MDSR, pad):
        super(MDSRWrapper, self).__init__()
        self.sub_mean = mdsr.sub_mean
        self.head = mdsr.head
        self.pre_process = mdsr.pre_process[0]
        self.body = mdsr.body
        self.upsample_x2 = mdsr.upsample[0]
        self.upsample_x2[1] = SimplePixelShuffle(2)
        self.upsample_x3 = mdsr.upsample[1]
        self.upsample_x3[1] = SimplePixelShuffle(3)
        self.tail = mdsr.tail
        self.add_mean = mdsr.add_mean
        self.pad = torch.tensor([pad])

    def forward(self, x, scale):
        x = self.sub_mean(x.permute((0, 3, 1, 2)))
        x = self.head(x)
        x = self.pre_process(x)

        x = self.body(x) + x

        if scale > 2:
            x = self.upsample_x3(x)
        else:
            x = self.upsample_x2(x)
        x = self.tail(x)
        x = self.add_mean(x)

        x = x.permute((0, 2, 3, 1)).clamp(0, 255)
        sr_pad = scale * self.pad
        x = x[:, sr_pad:-sr_pad, sr_pad:-sr_pad, :]

        return x

def getConvIds(onnx_graph):
    conv_ids = []
    for i in range(len(onnx_graph.node)):
        if onnx_graph.node[i].op_type == 'Conv':
            conv_ids.append(onnx_graph.node[i].output[0])
    return conv_ids

if __name__ == '__main__':
    args.model = 'MDSR'

    # Unpack params from argument
    tile_width = args.width
    tile_height = args.height
    pad = args.pad
    scale0 = torch.tensor(args.scale[0])
    conv_mem_portion = args.conv_mem_portion
    micro_batch_size = args.micro_batch_size
    batches_per_step = args.batches_per_step

    # Derived params
    batch_size = micro_batch_size * batches_per_step
    input_shape = (micro_batch_size, tile_height + 2 * pad, tile_width + 2 * pad, 3)

    mdsr_model = model.Model(args, utility.checkpoint(args)).model
    print('Creating MDSR Wrapper ...')
    mdsr_wrapper = MDSRWrapper(mdsr_model, pad)

    dir = os.path.split(args.pre_train)[0]
    filename = f'{args.model.lower()}-x{",".join([str(x) for x in args.scale])}-r{args.n_resblocks}c{args.n_feats}_h{tile_height}w{tile_width}cmp{args.conv_mem_portion:.2f}'
    if pad > 0:
        filename += f'p{pad}'
    if batch_size > 1:
        filename += f'_bs{batch_size}'
    if args.opset_version != 10:
        filename += f'_opset{args.opset_version}'
    filename += '.onnx'
    onnx_path = os.path.join(dir, filename)


    input_scale = 3
    dummy_inputs = (torch.randn(*input_shape), torch.tensor([input_scale]))
    print(f'input shape: {input_shape}, input scale: {input_scale}')
    dummy_output = mdsr_wrapper(dummy_inputs[0], torch.tensor(input_scale))
    print(f'output shape: {tuple(dummy_output.shape)}')

    print(f'Dumping ONNX protobuf (opset {args.opset_version}) at {onnx_path} ...')
    torch.onnx.export(
        torch.jit.script(mdsr_wrapper),
        dummy_inputs,
        onnx_path,
        example_outputs=(
            dummy_output,
        ),
        input_names=['lr', 'scale'],
        output_names=['sr'],
        opset_version=args.opset_version
    )

    onnx_model = onnx.load(onnx_path)
    config_path = os.path.splitext(onnx_path)[0] + '.json'

    print(f'Dumping ONNX config file at {config_path} ...')
    with open(config_path, 'w') as fp:
        json.dump(
            {
                'num_ipus': 1,
                'batches_per_step': args.batches_per_step,
                'conv_ids': getConvIds(onnx_model.graph),
                'conv_mem_portion': args.conv_mem_portion,
                'pad': args.pad,
                'input_shape': input_shape,
                'scale': args.scale[0],
                'max_num_ipus': 16,
                'border_type': 'REFLECT101'
            },
            fp,
            indent=2,
            sort_keys=True
        )

    if not args.skip_simplify:
        print('Simplifying ONNX protobuf ...')
        simplified_onnx_model, check_ok = onnxsim.simplify(
            onnx_path, check_n=args.check_n, perform_optimization=not args.skip_optimization,
            skip_fuse_bn=not args.enable_fuse_bn, input_shapes={'x': input_shape, 'scale': (1,)}
        )

        if check_ok:
            print(f'Dumping simplified ONNX protobuf at {onnx_path} ...')
            onnx.save(simplified_onnx_model, onnx_path)
        else:
            print(f'Simplified model failed the checking with random input tensor')
            print(f'Keeping the original ONNX protobuf')
