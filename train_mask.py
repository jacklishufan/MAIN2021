from detectron2.data.datasets import register_coco_instances
from orka_datasets import *
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from trainer import *

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("audiogram_segmentation_train",
                        #"audiogram_original"
)
cfg.DATASETS.TEST = ("audiogram_segmentation_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-4
single_iteration = 1* cfg.SOLVER.IMS_PER_BATCH
iterations_for_one_epoch = int(369 / single_iteration)
cfg.SOLVER.MAX_ITER = iterations_for_one_epoch * 20
cfg.TEST.EVAL_PERIOD = iterations_for_one_epoch * 5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.TEST.EVAL_PERIOD = 100
cfg.OUTPUT_DIR = "models/mask_detector"

    
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
