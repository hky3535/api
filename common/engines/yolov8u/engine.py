"""hekaiyue 何恺悦 2024-01-14"""
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

    def infer(self, frame, conf=0.25, iou=0.7, classes=False):
        if classes is False: classes = None
        results = self.model(
            source=frame, 
            conf=conf, 
            iou=iou, 
            half=self.half, 
            device=self.device, 
            classes=classes
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

