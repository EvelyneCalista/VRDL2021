from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import json
import glob
import os
import argparse
from detectron2.utils.visualizer import ColorMode
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
import pandas as pd

import json
import numpy as np
import os
import pycocotools.mask as mask_util
from tabulate import tabulate
import argparse
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import  BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

#from https://detectron2.readthedocs.io/en/latest/_modules/detectron2/evaluation/coco_evaluation.html?fbclid=IwAR0ERQpVHUXLpfCP8RfAqn0j9TqxfWW4nOPGoL9kAdpy3Nvf9wNW7ZPLH3g
def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k]+1,
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results
    
if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_annot", type=str)
    parser.add_argument("--train_folder", type=str)
    parser.add_argument("--path_test", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--modelzoo", type=str)
    parser.add_argument("--modelweight", type=str)
    args = parser.parse_args()
    register_coco_instances("my_dataset_train", {}, "/media/bsplab/Disk/Evelyne_Class/Sem1/CV_DL/HW3/detr-main/output_train1.json", "/media/bsplab/Disk/Evelyne_Class/Sem1/CV_DL/HW3/dataset1/train2017")
 
    metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")

    cfg = get_cfg()
    #"COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(args.modelzoo))
    cfg.DATASETS.TEST = ()   
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    # 4999 iterations seems good enough, but you can certainly train longer
    cfg.SOLVER.MAX_ITER = 1000
    # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes (data, fig, hazelnut)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  # build output folder
    cfg.MODEL.WEIGHTS ="args.modelweight"
    predictor = DefaultPredictor(cfg)

    ##get image id from json file##
    with open('./test_img_ids.json') as json_file:
        data = json.load(json_file)
    # print(type(data))
    df = pd.json_normalize(data)
    a = dict(zip(df.file_name, df.id))
    # print(a["TCGA-A7-A13E-01Z-00-DX1.png"])
    path_test = args.path_test
    results =[]
    for i in glob.glob (os.path.join(path_test,'*.png')):
        image_names = os.path.basename(i)
        im = cv2.imread(i)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                metadata=metadata,
                scale=1,
                instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))  
        plt.imshow(v.get_image())
        cv2.imwrite('demo'+image_names+'.png',v.get_image())
        hasil = instances_to_coco_json(outputs["instances"].to("cpu"), a[image_names])

        results.extend(hasil)

    mmcv.dump(results, 'answer.json', indent=4)
