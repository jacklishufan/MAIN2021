import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

class AxisDetector():
    def __init__(self, model_path, threshold=0.5, cpu=False):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
        self.cfg.INPUT.MIN_SIZE_TEST = 1000
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 24
        if cpu:
            self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)

    def inference(self, img):
        outputs = self.predictor(img)  # notice that here img should be in BGR format
        boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        classes = outputs['instances'].pred_classes.cpu().numpy()
        return [boxes, classes]