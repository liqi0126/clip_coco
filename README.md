# clip_coco

## Environment setup

1. clone [cocoapi](https://github.com/cocodataset/cocoapi), goto `PythonAPI` and run `make`
2. follow [clip](https://github.com/openai/CLIP) to install clip
3. download and unzip [features.tar.gz](https://drive.google.com/file/d/1v7sNMuzEUhiFfiOX4cGRNfNrWLP-fC9E/view?usp=share_link) (or generate by code)


## Usage

- run `python clip_coco.py` to export coco dataset to json {str(object id): clip_feature}
- run `python clip_img.py` to get clip feature of an image
- run `python clip_text.py` to get clip feature of a text

- run `python read_feature.py` to get clip feature from exported json based on ID

