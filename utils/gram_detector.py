import argparse
import cv2
import glob
import ntpath
import numpy as np
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class GramDetector:
    def __init__(self, model_path, threshold=0.7, padding=200, cpu=False):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
        self.cfg.INPUT.MIN_SIZE_TEST = 1000
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.padding = padding
        if cpu:
            self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)

    def inference(self, img):
        img = cv2.copyMakeBorder(img, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_REPLICATE)
        outputs = self.predictor(img)  # notice that here img should be in BGR format
        boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        classes = outputs['instances'].pred_classes.cpu().numpy()
        boxes, classes = self._refine_boxes(img, boxes, classes)

        return [boxes, classes]

    def _refine_boxes(self, img, boxes, classes):
        res_boxes = []
        res_classes = []

        for i, box in enumerate(boxes):
            if box[0] >= self.padding + img.shape[1] or box[2] < self.padding or \
                    box[1] >= self.padding + img.shape[0] or box[3] < self.padding:
                continue
            else:
                x1 = min(max(box[0], self.padding), img.shape[1] - 1 + self.padding)
                y1 = min(max(box[1], self.padding), img.shape[0] - 1 + self.padding)
                x2 = min(max(box[2], self.padding), img.shape[1] - 1 + self.padding)
                y2 = min(max(box[3], self.padding), img.shape[0] - 1 + self.padding)
                x1, y1, x2, y2 = x1 - self.padding, y1 - self.padding, x2 - self.padding, y2 - self.padding
                res_boxes.append([x1, y1, x2, y2])
                res_classes.append(classes[i])

        res_boxes = np.array(res_boxes)
        res_classes = np.array(res_classes)

        return [res_boxes, res_classes]

