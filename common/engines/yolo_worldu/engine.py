"""hekaiyue 何恺悦 2024-03-15"""
import os
import sys
engines_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, engines_path + "/source/ultralytics")

from ultralytics import YOLO

import numpy


class Engine:
    def __init__(self, weight, device="gpu", half=False):
        self.device = "0" if device == "gpu" else "cpu"
        self.half = half
        self.model = YOLO(weight)
        self.last_classes = False
    
    def reset_classes(self, classes): # 重置提示词
        if classes != self.last_classes:
            if classes is False: return
            self.model.set_classes(classes)

    def infer(self, frame, conf=0.1, iou=0.7, classes=False):
        self.reset_classes(classes=classes)
        if classes is False: classes = None
        results = self.model(
            source=frame, 
            conf=conf, 
            iou=iou, 
            half=self.half, 
            device=self.device
        )
        results = results[0].cpu().boxes.numpy()
        results = numpy.column_stack((
            results.xyxy, 
            results.conf * 100, 
            results.cls
        )).astype(int)
        results = results.tolist()
        # results = numpy.asarray(results)
        return results

