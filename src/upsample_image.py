import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import utility
from model import Model

IMAGE_DIR = '../../images/lr/540p'

if __name__ == '__main__':
    from option import args as arguments

    model = Model(arguments, utility.checkpoint(arguments))

    filenames = [
        filename for filename in os.listdir(IMAGE_DIR)
        if os.path.splitext(filename)[1] in ('.jpg', '.png', '.bmp')
    ]

    output_dir = os.path.join('SRImages', os.path.basename(IMAGE_DIR))
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for filename in tqdm(filenames):
            lr_img = cv2.imread(os.path.join(IMAGE_DIR, filename))
            input_tensor = torch.from_numpy(lr_img).unsqueeze(0).permute(0, 3, 1, 2).type(torch.float32)
            sr_img = model(input_tensor, 0) \
                .permute(0, 2, 3, 1).squeeze(0).clamp(0, 255).numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, filename), sr_img)
