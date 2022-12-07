# -*- coding: utf-8 -*-

import os
import clip
import json
from fire import Fire

import torch
from torch.utils.data import DataLoader

from data import CocoImgDataset, CocoCaptionDataset, CocoCatDataset


def main(data_root='/data/coco',
         data_type='val2017',
         task='imgs',  # img, captions, instances, stuff
         device='cuda',
         batch_size=512):

    model, preprocess = clip.load('ViT-L/14@336px', device)

    if task == 'imgs':
        dataset = CocoImgDataset(data_root, data_type, preprocess)
    elif task == 'captions':
        dataset = CocoCaptionDataset(data_root, data_type)
    elif task == 'instances':
        dataset = CocoCatDataset(data_root, data_type, 'instances')
    elif task == 'stuff':
        dataset = CocoCatDataset(data_root, data_type, 'stuff')
    else:
        raise NotImplementedError

    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=16)

    model.eval()

    feature_json = {}
    total = len(dataset)
    acc = 0
    with torch.no_grad():
        for i, (idx, content) in enumerate(data_loader):
            content = content.to(device)
            if task == 'imgs':
                feature = model.encode_image(content).cpu().numpy().astype('float')
            else:
                feature = model.encode_text(content.squeeze()).cpu().numpy().astype('float')

            for j in range(len(idx)):
                feature_json[idx[j].item()] = list(feature[j])

            acc += len(idx)
            print(f'{acc}/{total} processed')

    os.makedirs('features', exist_ok=True)
    if task == 'instances' or task == 'stuff':
        file = f'features/vit_L_14_336px_{task}_category.json'
    else:
        file = f'features/vit_L_14_336px_{task}_{data_type}.json'

    with open(file, 'w') as f:
        json.dump(feature_json, f, indent=4)

if __name__ == '__main__':
    Fire(main)
