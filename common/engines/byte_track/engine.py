"""hekaiyue 何恺悦 2023-11-16"""
import os
import sys
engines_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, engines_path + "/source/yolo_tracking")

from boxmot import BYTETracker

import numpy


class Engine:
    def __init__(self, track_thresh=0.45, match_thresh=0.8):
        self.byte_track = BYTETracker(
            track_thresh=track_thresh, 
            match_thresh=match_thresh, 
            track_buffer=25, 
            frame_rate=30
        )

    def infer(self, results):
        if len(results) == 0: return results
        results = numpy.asarray(results)
        results = results.astype(float)
        results[:, 4] /= 100 # 置信度进行反向
        results = self.byte_track.update(dets=results, _=False)
        if len(results) == 0: return results.tolist()
        results[:, 5] *= 100 # 置信度进行正向
        results = results[:, :-1] # 去除最后的内部id
        results = results.astype(int)
        results = results.tolist()
        return results

