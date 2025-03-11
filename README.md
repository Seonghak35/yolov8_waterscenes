## YOLOV8 Object Detection Training on WaterScenes Datasets
---

## Contents
1. [Performance](#Performance)
2. [Environment](#Environment)
3. [Reference](#Reference)


## Performance
| Training Dataset | Weight files | Test Dataset | Input Image Size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| COCO-Train2017 | [yolov8_n.pth](https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n.pth) | COCO-Val2017 | 640x640 | 36.7 | 52.1
| COCO-Train2017 | [yolov8_s.pth](https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s.pth) | COCO-Val2017 | 640x640 | 44.1 | 61.0
| COCO-Train2017 | [yolov8_m.pth](https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m.pth) | COCO-Val2017 | 640x640 | 49.3 | 66.3
| COCO-Train2017 | [yolov8_l.pth](https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l.pth) | COCO-Val2017 | 640x640 | 52.0 | 68.9
| COCO-Train2017 | [yolov8_x.pth](https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x.pth) | COCO-Val2017 | 640x640 | 52.9 | 69.9

## Environment
torch==1.2.0
If you want to use AMP(Automatic Mixed Precision), torch>=1.7.1.

## Reference
https://github.com/ultralytics/ultralytics
