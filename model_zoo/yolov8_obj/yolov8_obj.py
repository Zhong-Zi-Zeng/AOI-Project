from __future__ import annotations, absolute_import
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'yolov7_obj'))
from typing import Union, Any
from model_zoo.base.BaseDetectModel import BaseDetectModel
from engine.timer import TIMER
from ultralytics import YOLO
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, check_path)
import numpy as np
import cv2
import torch
import random
import subprocess


class Yolov8Obj(BaseDetectModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def _config_transform(self):
        pass

    def _load_model(self):
        # Load model
        self.model = YOLO(self.cfg['weight'])

    def train(self):
        """
            Run每個model自己的training command
        """
        pass

    def _predict(self,
                 source: Union[str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 nms_thres: float = 0.5,
                 *args: Any,
                 **kwargs: Any
                 ) -> dict:

        if not hasattr(self, 'model'):
            self._check_weight_path(self.cfg['weight'])
            self._load_model()

        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        with TIMER[0]:
            # ------------------------------Pre-process (Start)----------------------------
            with TIMER[1]:
                # Load image
                if isinstance(source, str):
                    original_image = cv2.imread(source)
                elif isinstance(source, np.ndarray):
                    original_image = source
                else:
                    raise ValueError

            # ----------------------------Pre-process (End)----------------------------

            # ----------------------------Inference (Start))----------------------------
            # Inference
            with TIMER[2], torch.no_grad():
                pred = self.model.predict(original_image,
                                          imgsz=tuple(self.cfg['imgsz']),
                                          conf=conf_thres,
                                          iou=nms_thres)[0]

            # ----------------------------Inference (End)----------------------------

            # ----------------------------NMS-process (Start)----------------------------
            with TIMER[3]:
                pass

            # ----------------------------NMS-process (End)----------------------------

            # ----------------------------Post-process (Start)----------------------------
            # For evaluation
            class_list = []
            score_list = []
            bbox_list = []

            for i, det in enumerate(pred.boxes):  # detections per image
                cls = int(det.cls[0].cpu())
                conf = float(det.conf[0].cpu())
                bbox = det.xywh[0].cpu()

                class_list.append(cls)
                score_list.append(conf)
                bbox_list.append(list(map(float, bbox)))

            return {
                'result_image': pred.plot(),
                'class_list': class_list,
                'score_list': score_list,
                'bbox_list': bbox_list
            }
