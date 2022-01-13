## VRDL 2021 HW3 Image Super Resolution

## <div align="center">Introduction</div>
**The code is use [IMDN](https://github.com/Zheng222/IMDN) for homework of VRDL-Fall 2021 class in NYCU.**
In this homework we need to do single image super-resolution task (SISR). Image super resolution (SR) is an important task in computer vision field, where recovering low-resolution images into high-resolution images. With the fast improvement in deep learning field of computer vision in the last decade, there are many technique ranges from convolutional neural network, generative adversarial network, and transformer approaches. The best result that I get is come from the IMDN model that can reach PSNR 27.8838.

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
