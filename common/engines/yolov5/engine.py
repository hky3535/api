"""hekaiyue 何恺悦 2024-01-14"""
import os
import sys
engines_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, engines_path + "/source/yolov5")

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

import cv2
import numpy
import torch


class Engine:
    def __init__(self, weight, device="gpu", half=False):
        self.half = half
        device = "cuda" if device == "gpu" else "cpu"
        self.device = torch.device(device)
        self.model = attempt_load(weight, device=self.device) # 加载模型到device
        self.stride = max(int(self.model.stride.max()), 32) # 获取模型的步长
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names # 获取类别名称
        self.model.half() if self.half else self.model.float()
        self.model.to(self.device).eval()

    def infer(self, frame, conf=0.25, iou=0.7, classes=False):
        if classes is False: classes = None
        # preprocess
        img = frame
        img = letterbox(img, auto=False, stride=self.stride)[0]     # padded resize
        img = img.transpose((2, 0, 1))[::-1]                        # 调整图像维度顺序
        img = numpy.ascontiguousarray(img)                          # 创建一个连续的内存数组
        img = torch.from_numpy(img).to(self.device)                 # 加载进算力卡
        img = img.half() if self.half else img.float()              # to fp16/32
        img /= 255                                                  # 0 - 255 to 0.0 - 1.0
        img = img[None]                                             # expand for batch dim
        # inference
        with torch.no_grad():
            pred = self.model(img)
        # postprocess
        pred = non_max_suppression(pred, conf, iou, classes)
        # generate outputs
        result = pred[0].cpu().numpy()
        result[:, :4] = scale_boxes(img.shape[2:], result[:, :4], frame.shape).round()

        results = list()
        for *xyxy, conf, cls in reversed(result):
            x0, y0, x1, y1 = xyxy
            results.append([int(x0), int(y0), int(x1), int(y1), int(conf * 100), int(cls)])
        # results = numpy.asarray(results)
        return results

