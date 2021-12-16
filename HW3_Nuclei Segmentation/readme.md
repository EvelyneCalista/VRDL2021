## VRDL 2021 HW3 Nuclei Segmentation
## Introduction
**The code is use [Detectron2](https://github.com/facebookresearch/detectron2) for homework of VRDL-Fall 2021 class in NYCU.**
## <div align="center">Data Preprocessing</div>
The provided dataset annotations are a mask images for each instances like below. Therefore we need to convert those mask by using createcustom.py
 ```bash
 $ python createcustom.py --path 'path/to/traindataset'

```
customdataset.py will give you annotation.json file. 

Arrange the training image folder into /train/images.


## <div align="center">Detectron2</div>
1. Training
 ```bash
 $ python createcustom.py --path 'path/to/traindataset'

```
2. Inference
The output would be 1 json file with coco format.
The provided dataset annotations are a mask images for each instances like below. Therefore we need to convert those mask by using createcustom.py
 ```bash
 $ python createcustom.py --path 'path/to/traindataset'

```

</details>
## <div align="center">Reference</div>
1. YOLOv5 [YOLOV5 Official Github](https://github.com/ultralytics/yolov5) 
