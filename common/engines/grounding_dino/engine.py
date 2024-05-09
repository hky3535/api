"""hekaiyue 何恺悦 2024-01-14"""
import os
import sys
engines_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, engines_path + "/source/GroundingDINO")

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import cv2
import numpy
import torch
from PIL import Image


class Engine:
    def __init__(self, weight, config, device="gpu"):
        self.device = "cuda" if device == "gpu" else "cpu"

        args = SLConfig.fromfile(config)
        args.device = self.device
        args.text_encoder_type = engines_path + "/source/bert-base-uncased"
        self.model = build_model(args)
        checkpoint = torch.load(weight, map_location="cpu")
        load_res = self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = self.model.eval()
        self.model = self.model.to(self.device)

    def infer(self, frame, conf=0.1, iou=0.7, classes=False):
        # 特殊映射（对齐ultralytics）
        text_prompt = classes
        text_threshold = conf
        if text_prompt is False: return numpy.asarray([])
        # preprocess
        img = frame
        h, w, _ = frame.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img, _ = transform(img, None) # 3, h, w
        img = img.to(self.device)
        _text_prompt = ".".join(text_prompt) + "."
        # inference
        with torch.no_grad():
            outputs = self.model(img[None], captions=[_text_prompt])
        # postprocess
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
            # filter output
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > conf
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
            # get phrase
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(_text_prompt)
            # build pred
        results = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            # 逆推文字对应类别序号
            if pred_phrase in text_prompt: cls = text_prompt.index(pred_phrase)
            else: cls = -1
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([w, h, w, h])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            results.append([
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), 
                int(logit.max().item() * 100), int(cls)
            ])
        results = nms(detections=results, iou_threshold=iou)
        # results = numpy.asarray(results)
        return results


def nms(detections, iou_threshold):
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3]) # 边际框
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1) # 交集
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        union_area = box1_area + box2_area - intersection_area # 并集
        iou = intersection_area / union_area # 交并比
        return iou
    
    detections.sort(key=lambda x: x[4], reverse=True) # Sort detections by confidence in descending order
    selected_detections = [] # Initialize list to store the selected detections
    
    while len(detections) > 0:
        best_detection = detections[0]
        selected_detections.append(best_detection) # Select the detection with the highest confidence
        
        detections = detections[1:] # Remove the selected detection from the list
        
        remaining_detections = [] # Calculate the overlap (IoU) between the selected detection and the remaining detections
        for detection in detections:
            iou = calculate_iou(best_detection, detection)
            if iou < iou_threshold:
                remaining_detections.append(detection)
        
        detections = remaining_detections # Update the detections list with the remaining detections
    
    return selected_detections

