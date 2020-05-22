import os

import torch

import model
import utility
from option import args

class MultiBranchModel(torch.nn.Module):
    WIDTH = 1920
    HEIGHT = 1080
    def __init__(self,
         args, checkpoint,
         num_tiles_x, num_tiles_y
     ):
        super(MultiBranchModel, self).__init__()
        self.num_tiles_x = num_tiles_x
        self.num_tiles_y = num_tiles_y
        self.width = MultiBranchModel.WIDTH // self.num_tiles_x
        self.height = MultiBranchModel.HEIGHT // self.num_tiles_y
        self.branches = []
        for i in range(self.num_branch):
            self.branches.append(model.Model(args, checkpoint))
        for branch in self.branches:
            for p in branch.parameters():
                p.requires_grad = False
        self.grid_x = list(range(0, MultiBranchModel.WIDTH, self.width))
        self.grid_y = list(range(0, MultiBranchModel.HEIGHT, self.height))

    @property
    def num_branch(self):
        return self.num_tiles_x * self.num_tiles_y

    def forward(self, x: torch.Tensor, idx_scale):
        x = x.permute((0, 3, 1, 2))
        lr_tiles = [x[:, :, gy:gy+self.height, gx:gx+self.width] for gx in self.grid_x for gy in self.grid_y]
        sr_tiles = []
        for lr_tile, model in zip(lr_tiles, self.branches):
            sr_tiles.append(model(lr_tile, idx_scale))
        sr_image = torch.cat(
            [
                torch.cat(sr_tiles[self.num_tiles_y * i:self.num_tiles_y * (i + 1)], dim=2)
                for i in range(self.num_tiles_x)
            ],
            dim=3
        ).permute((0, 2, 3, 1)).clamp(0, 255)
        return sr_image


if __name__ == '__main__':
    checkpoint = utility.checkpoint(args)
    module = MultiBranchModel(args, checkpoint, 4, 2)

    input_shape = (1, module.HEIGHT, module.WIDTH, 3)
    print(f'input shape: {input_shape}')
    dummy_input = torch.randn(*input_shape)

    dir = os.path.split(args.pre_train)[0]
    filename = f'multibranch-{args.model.lower()}-x{args.scale[0]}-r{args.n_resblocks}c{args.n_feats}_h{input_shape[1]}w{input_shape[2]}.onnx'
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
