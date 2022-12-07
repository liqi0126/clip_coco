# -*- coding: utf-8 -*-

import os
import clip
import torch

from fire import Fire
from PIL import Image

def main(img_path='test.jpg',
         device='cuda'):

    model, preprocess = clip.load('ViT-L/14@336px', device)
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        img_features = model.encode_image(image).cpu().numpy()

    print(img_features)

if __name__ == '__main__':
    Fire(main)
