from __future__ import annotations
from typing import Optional, Union, Any
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'mmdetection'))
from model_zoo.base.BaseDetectModel import BaseDetectModel
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, load_python, update_python_file)
from engine.timer import TIMER
from mmdet.apis import DetInferencer
import numpy as np
import subprocess
import cv2
import torch


class CODETR(BaseDetectModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)
        self.cfg = cfg

    def _config_transform(self):
        config_dict = load_python(self.cfg['cfg_file'])

        # Optimizer
        if self.cfg['optimizer'] == 'SGD':
            optimizer = dict(_delete_=True, type='OptimWrapper',
                             optimizer=dict(type='SGD', lr=self.cfg['lr'], momentum=0.937, weight_decay=0.0005))
        elif self.cfg['optimizer'] == 'Adam':
            optimizer = dict(_delete_=True, type='OptimWrapper',
                             optimizer=dict(type='Adam', lr=self.cfg['lr'], betas=(0.937, 0.999), weight_decay=0.0005))
        else:
            optimizer = dict(_delete_=True, type='OptimWrapper',
                             optimizer=dict(type='AdamW', lr=self.cfg['lr'], betas=(0.937, 0.999), weight_decay=0.0005))

        # Update base file path
        new_base = []
        for base in config_dict['_base_']:
            new_base.append(os.path.join(get_model_path(self.cfg), 'configs', base))

        # Update config file
        variables = {
            '_base_': new_base,
            'data_root': self.cfg['coco_root'],
            'classes': self.cfg['class_names'],
            'batch_size': self.cfg['batch_size'],
            'epoch': self.cfg['end_epoch'],
            'height': self.cfg['imgsz'][0],
            'width': self.cfg['imgsz'][1],
            'num_classes': self.cfg['number_of_class'],
            'lr': self.cfg['lr'],
            'start_factor': self.cfg['initial_lr'] / self.cfg['lr'],
            'minimum_lr': self.cfg['minimum_lr'],
            'warmup_begin': self.cfg['start_epoch'],
            'warmup_end': self.cfg['warmup_epoch'],
            'optim_wrapper': optimizer,
            'check_interval': self.cfg['save_period'],
            'nms_threshold': self.cfg['nms_thres'],
        }

        update_python_file(self.cfg['cfg_file'], os.path.join(get_work_dir_path(self.cfg), 'cfg.py'), variables)
        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg.py')

    def _load_model(self):
        self.model = DetInferencer(model=self.cfg['cfg_file'],
                                   weights=self.cfg['weight'],
                                   show_progress=False)

    def train(self):
        subprocess.run([
            'python',
            os.path.join(get_model_path(self.cfg), 'tools', 'train.py'),
            self.cfg['cfg_file'],
            '--work-dir', get_work_dir_path(self.cfg)
        ])

    def _predict(self,
                 source: Union[str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 nms_thres: float = 0.5,
                 *args: Any,
                 **kwargs: Any
                 ) -> dict:
        if not hasattr(self, 'model'):
            self._load_model()

        with TIMER[0]:
            with TIMER[1]:
                pass
            with TIMER[2]:
                pass
            with TIMER[3]:
                pass

        return {
            'result_image': None,
            'class_list': None,
            'score_list': None,
            'bbox_list': None
        }
