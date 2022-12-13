# CS260 Course Project: Vectorized Similarity Search in Multi-modal Databases


* Zhenghao Peng 
* Qi Li 
* Siyan Zhao
* Yijia Xiao 
* Linqiao Jiang

2022 Fall



## Usage

Step 1. clone [cocoapi](https://github.com/cocodataset/cocoapi), goto `PythonAPI` and run `make`

Step 2. follow [clip](https://github.com/openai/CLIP) to install clip

Step 3A. download and unzip [features.tar.gz](https://drive.google.com/file/d/1v7sNMuzEUhiFfiOX4cGRNfNrWLP-fC9E/view?usp=share_link) (or generate by code, see Step 3B)

Step 3B. run `python clip_coco.py` to export coco dataset to json {str(object id): clip_feature}

Step 4. Open `play.ipynb` and change the path toward generated features / coco dataset to correct path and run! Enjoy!


