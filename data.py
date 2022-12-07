# -*- coding: utf-8 -*-

import os
import clip
import torch

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from PIL import Image


class CocoCatDataset(Dataset):
    def __init__(self, data_root, data_type, task):
        self.root = data_root
        self.data_type = data_type
        self.coco = COCO(f'{data_root}/annotations/{task}_{data_type}.json')
        self.ids = list(self.coco.cats.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        cat = clip.tokenize(self.coco.cats[idx]['name'])
        return idx, cat


class CocoCaptionDataset(Dataset):
    def __init__(self, data_root, data_type):
        self.root = data_root
        self.data_type = data_type
        self.coco = COCO(f'{data_root}/annotations/captions_{data_type}.json')
        self.ids = list(self.coco.anns.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        caption = clip.tokenize(self.coco.anns[idx]['caption'])

        return idx, caption


class CocoImgDataset(Dataset):
    def __init__(self, data_root, data_type, transform=None):
        self.root = data_root
        self.data_type = data_type
        self.coco = COCO(f'{data_root}/annotations/instances_{data_type}.json')
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        image = Image.open(os.path.join(self.root, self.data_type, self.coco.imgs[img_id]['file_name'])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return img_id, image


