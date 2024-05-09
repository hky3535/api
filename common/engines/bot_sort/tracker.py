"""hekaiyue 何恺悦 2023-11-16"""
import os
import sys
engines_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, engines_path + "/source/yolo_tracking")

from boxmot import BoTSORT

import numpy
import cv2
import base64
from pathlib import Path


class Engine:
    def __init__(self, weight, device="gpu", half=False, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6, match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25):
        device = "cuda" if device == "gpu" else "cpu"
        self.bot_sort = BoTSORT(
            model_weights=Path(weight),
            device=device,
            fp16=half,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=30,
            match_thresh=match_thresh,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            cmc_method="sparseOptFlow",
            frame_rate=30,
        )

    def infer(self, frame, results):
        if len(results) == 0: return results
        results = numpy.asarray(results)
        results = results.astype(float)
        results[:, 4] /= 100 # 置信度进行反向
        results = self.bot_sort.update(dets=results, img=frame)
        if len(results) == 0: return results.tolist()
        results[:, 5] *= 100 # 置信度进行正向
        results = results[:, :-1] # 去除最后的内部id
        results = results.astype(int)
        results = results.tolist()
        return results

