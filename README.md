# clip_coco

## Environment setup

1. clone [cocoapi](https://github.com/cocodataset/cocoapi), goto `PythonAPI` and run `make`
2. follow [clip](https://github.com/openai/CLIP) to install clip

## Usage

- run `python clip_coco.py` to export coco dataset to json {str(object id): clip_feature}
- run `python clip_img.py` to get clip feature of an image
- run `python clip_text.py` to get clip feature of a text

- run `python read_feature.py` to get clip feature from exported json based on ID

