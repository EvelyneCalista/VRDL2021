from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_annot", type=str)
    parser.add_argument("--train_folder", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--modelzoo", type=str)
    parser.add_argument("--output_modelweight", type=str)
    args = parser.parse_args()
    setup_logger()
    register_coco_instances("my_dataset_train", {}, args.data_annot, args.train_folder)
    metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.modelzoo))
    cfg.train.init_checkpoint = model_zoo.get_checkpoint_url(args.modelzoo)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.WEIGHTS = os.path.join('model', "model_final_Cascade.pkl")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()   
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)  
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS =args.output_modelweight

if __name__ == "__main__":
    main()
    
