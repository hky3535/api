"""hekaiyue 何恺悦 2024-01-14"""
import os
import sys
engines_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, engines_path + "/source/pytorch-YOLOv4")

from tool.darknet2pytorch import Darknet
from tool.torch_utils import do_detect

import cv2
import numpy


class Engine:
    def __init__(self, weight, config, device="gpu"):
        self.cuda = True if device == "gpu" else False
        self.model = Darknet(config)
        self.model.load_weights(weight)
        if self.cuda: self.model.cuda()

    def infer(self, frame, conf=0.25, iou=0.7, classes=False):
        # preprocess
        h, w, _ = frame.shape
        img = frame
        img = cv2.resize(img, (self.model.width, self.model.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # inference
        pred = do_detect(self.model, img, conf, iou, self.cuda)
        # generate outputs
        result = pred[0]
        results = list()
        for *xyxy, _, conf, cls in reversed(result):
            if classes is not False and int(cls) not in classes: continue # 筛选classes
            x0, y0, x1, y1 = xyxy
            results.append([int(x0*w), int(y0*h), int(x1*w), int(y1*h), int(conf*100), int(cls)])
        # results = numpy.asarray(results)
        return results

