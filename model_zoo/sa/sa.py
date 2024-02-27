from __future__ import annotations, absolute_import
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'yolov7_obj'))
from typing import Union, Any
from model_zoo.base.BaseSemanticModel import BaseSemanticModel
from engine.timer import TIMER
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, check_path)
import numpy as np
import cv2
import torch
import random
import subprocess


class SA(BaseSemanticModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def _config_transform(self):
        # Update data file
        cfg_file = load_yaml(self.cfg['cfg_file'])

        # ===========Dataset===========
        cfg_file['coco_root'] = os.path.join(os.getcwd(), self.cfg['coco_root'])

        # ===========Augmentation===========
        cfg_file['hsv_h'] = self.cfg['hsv_h']
        cfg_file['hsv_s'] = self.cfg['hsv_s']
        cfg_file['hsv_v'] = self.cfg['hsv_v']
        cfg_file['degrees'] = self.cfg['degrees']
        cfg_file['translate'] = self.cfg['translate']
        cfg_file['scale'] = self.cfg['scale']
        cfg_file['shear'] = self.cfg['shear']
        cfg_file['perspective'] = self.cfg['perspective']
        cfg_file['flipud'] = self.cfg['flipud']
        cfg_file['fliplr'] = self.cfg['fliplr']

        # ===========Training===========
        cfg_file['batch_size'] = self.cfg['batch_size']
        cfg_file['eval_interval'] = self.cfg['save_period']
        cfg_file['save_interval'] = self.cfg['eval_period']
        cfg_file['use_points'] = self.cfg['use_points']
        cfg_file['use_boxes'] = self.cfg['use_boxes']
        cfg_file['device'] = self.cfg['device']
        cfg_file['end_epoch'] = self.cfg['end_epoch']
        cfg_file['weight'] = self.cfg['weight']

        # ===========Optimizer===========
        cfg_file['optimizer'] = self.cfg['optimizer']
        cfg_file['lr'] = self.cfg['lr']
        cfg_file['minimum_lr'] = self.cfg['minimum_lr']

        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg_file.yaml')
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'cfg_file.yaml'), cfg_file)

    def _load_model(self):
        # Load model
        pass

    def train(self):
        """
            Run每個model自己的training command
        """
        subprocess.run(['python',
                        os.path.join(get_model_path(self.cfg), 'main.py'),
                        '--cfg', self.cfg['cfg_file'],
                        '--work_dir', get_work_dir_path(self.cfg)])

    # TODO: 完成sa的predict部分
    def _predict(self,
                 source: Union[str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 *args: Any,
                 **kwargs: Any
                 ) -> dict:

        if not hasattr(self, 'model'):
            self._check_weight_path(self.cfg['weight'])
            self._load_model()

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
                pass
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

            # for i, det in enumerate(pred.boxes):  # detections per image
            #     cls = int(det.cls[0].cpu())
            #     conf = float(det.conf[0].cpu())
            #     bbox = det.xywh[0].cpu()
            #
            #     class_list.append(cls)
            #     score_list.append(conf)
            #     bbox_list.append(list(map(float, bbox)))
            #
            # return {
            #     'result_image': pred.plot(),
            #     'class_list': class_list,
            #     'score_list': score_list,
            #     'bbox_list': bbox_list
            # }
