# -*- coding: utf-8 -*-

import os
import clip
import torch

from fire import Fire
from PIL import Image

def main(text='a cat',
         device='cuda'):

    model, preprocess = clip.load('ViT-L/14@336px', device)
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text).cpu().numpy()

    print(text_features)

if __name__ == '__main__':
    Fire(main)

