## VRDL 2021 HW3 Nuclei Segmentation
## Introduction
**The code is use [Detectron2](https://github.com/facebookresearch/detectron2) for homework of VRDL-Fall 2021 class in NYCU.**

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Usage</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Train</summary>

 
 ```bash
 $ python train.py --img 640 --batch 6 --epochs 20 --data dataset.yaml --weights yolov5m.pt

```

</details>

<details open>
<summary>Inference</summary>

 
 ```bash
 $ python detect.py --source /testfolder --weights /train/exp/weights/best.pt --conf 0.1
```

</details>
## <div align="center">Reference</div>
1. YOLOv5 [YOLOV5 Official Github](https://github.com/ultralytics/yolov5) 
