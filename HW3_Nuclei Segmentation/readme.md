## VRDL 2021 HW3 Nuclei Segmentation

## <div align="center">Introduction</div>
**The code is use [Detectron2](https://github.com/facebookresearch/detectron2) for homework of VRDL-Fall 2021 class in NYCU.**
The assignment was to do instance segmentation for nuclei segmentation. 14606 instances for nucleus category, in 24 images. For this assignment I use detectron2 with maskrcnn backbone ResNeXt-101-32x8d from coco instance segmentation config folder, where give the best result experiment. Detectron 2 is a library platform for object detection and segmentation algorithms from Facebook AI Research.

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
 $ python detectron2.py --data_annot 'path/to/traindataset' --train_folder 'path/to/trainfolder' --output_dir 'path/to/output_dir' --modelzoo 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml' --output_modelweight 'outputfilename'

```
The output would be 1 model weight file.

2. Inference

 ```bash
 $ python inference.py --data_annot 'path/to/traindataset' --train_folder 'path/to/trainfolder' --path_test 'path/to/test_folder' --output_dir 'path/to/output_dir' --modelzoo 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml' --modelweight 'modelweightname'

```

The output would be 1 json file with coco format.

## <div align="center">Reference</div>
1. detectron2 [detectron2 Official Github](https://github.com/facebookresearch/detectron2) 
