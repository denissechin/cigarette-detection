# Cigarette Detection

This repository contains generation code for object detection dataset and manuals for following model training and demo.

# YOLOv5 performance on video
![demo](https://user-images.githubusercontent.com/66907141/191371680-b99ca68b-2d7a-4f97-a4b4-aeaa7f4c73ba.gif)

# Installation

```
git clone https://github.com/denissechin/cigarette-detection.git
cd cigarette-detection
pip install -r requirements.txt
```

# Dataset generation concept
Tools from dataset_generation folder can be used to extract hand and face landmarks via `mediapipe` library and compute points where a cigarette potentially could be, current version can attach cigarette images to space between lips (if mouth is not opened wide) or space between index and middle fingers (if they are not far from each other).

# Dataset generation
generate_dataset.py takes in image directory with any image files and generates dataset in COCO format at save_path
```
cd dataset_generation
python generate_dataset.py --cigarette-dataset-path ./cigarette_dataset/ --image-path @path-to-your-image-dataset@ --save-path @save-dir@
```

# Some generation results
<img src="https://user-images.githubusercontent.com/66907141/191374517-ca80e850-e1ca-4efc-8a9b-bb9dd6e65893.png" width="300">
<img src="https://user-images.githubusercontent.com/66907141/191374959-19f285d2-352b-4ec8-a1be-d07086ffd2aa.png" width="300">
<img src="https://user-images.githubusercontent.com/66907141/191375270-cbdb229d-1cfd-49bb-adf9-b4cb65413971.png" width="300">

# Training

To train YOLOv5 you can use [this](https://colab.research.google.com/drive/1JHkZvUl_RkPDP28pPuw-NEHf-obBpR8h?usp=sharing) notebook, you just need to convert your generated dataset to YOLO format with `tools/coco_to_yolo.py`, compress and upload it to Google Drive.

## TODOs
* Manually label test dataset and compute metrics on it (WIP).
* Add perspective transforms to cigarette images before attaching.
* Adjust cigarette direction according to hand\face direction.
* Add augmentations feature.
