# -*- coding: utf-8 -*-

import json

from fire import Fire
from pycocotools.coco import COCO


def main(data_root='/data/coco',
         task='instances',
         data_type='val2017'):

    coco = COCO(f'{data_root}/annotations/captions_{data_type}.json')
    coco_instance = COCO(f'{data_root}/annotations/instances_{data_type}.json')
    coco_stuff = COCO(f'{data_root}/annotations/stuff_{data_type}.json')

    # image features
    with open(f'features/vit_L_14_336px_imgs_{data_type}.json', 'r') as f:
        imgs_feature = json.load(f)

    img_id = coco.getImgIds()[0]
    print(coco.loadImgs(img_id))
    print(imgs_feature[str(img_id)][:5])

    # caption features
    with open(f'features/vit_L_14_336px_captions_{data_type}.json', 'r') as f:
        caps_feature = json.load(f)

    cap_id = coco.getAnnIds()[0]
    print(coco.loadAnns(cap_id))
    print(caps_feature[str(cap_id)][:5])

    # instances
    with open('features/vit_L_14_336px_instances_category.json', 'r') as f:
        cat_features = json.load(f)

    cat_id = coco_instance.getCatIds()[0]
    print(coco_instance.loadCats(cat_id))
    print(cat_features[str(cat_id)][:5])

    # instances
    with open('features/vit_L_14_336px_stuff_category.json', 'r') as f:
        cat_features = json.load(f)

    cat_id = coco_stuff.getCatIds()[0]
    print(coco_stuff.loadCats(cat_id))
    print(cat_features[str(cat_id)][:5])



if __name__ == '__main__':
    Fire(main)
