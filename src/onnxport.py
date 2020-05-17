import os

import torch

import EDSR.src.model as model
import EDSR.src.utility as utility
from EDSR.src.option import args


class ModelWrapper(torch.nn.Module):
    def __init__(self, submodule):
        super(ModelWrapper, self).__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor, idx_scale):
        return self.submodule(x.permute((0, 3, 1, 2)), idx_scale) \
            .permute((0, 2, 3, 1)).clamp(0, 255)


if __name__ == '__main__':
    w = 480
    h = 270

    checkpoint = utility.checkpoint(args)
    module = ModelWrapper(model.Model(args, checkpoint))

    input_shape = (1, h, w, 3)
    print(f'input shape: {input_shape}')
    dummy_input = torch.randn(*input_shape)

    dir = os.path.split(args.pre_train)[0]
    filename = f'w{args.model.lower()}-x{args.scale[0]}-r{args.n_resblocks}c{args.n_feats}_h{input_shape[1]}w{input_shape[2]}.onnx'
    path = os.path.join(dir, filename)
    print(f'Dumping ONNX protobuf at {path} ...')
    torch.onnx.export(
        module,
        (
            dummy_input,
            torch.tensor(args.scale, dtype=torch.int)
        ),
        path,
        opset_version=10
    )
