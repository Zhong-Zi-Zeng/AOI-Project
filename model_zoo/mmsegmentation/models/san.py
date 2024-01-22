from __future__ import annotations
from typing import Union, Any
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'mmdetection'))
from model_zoo.base.BaseInstanceModel import BaseInstanceModel
from .BaseMMSegmentation import BaseMMdetection
from engine.general import (get_work_dir_path, get_model_path, load_python, update_python_file)
from engine.timer import TIMER
import numpy as np
import cv2


class SAN(BaseMMdetection, BaseInstanceModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)
        self.cfg = cfg

    def _config_transform(self):
        config_dict = load_python(self.cfg['cfg_file'])

        # Optimizer
        # if self.cfg['optimizer'] == 'SGD':
        #     optimizer = dict(_delete_=True, type='OptimWrapper',
        #                      optimizer=dict(type='SGD', lr=self.cfg['lr'], momentum=0.937, weight_decay=0.0005))
        # elif self.cfg['optimizer'] == 'Adam':
        #     optimizer = dict(_delete_=True, type='OptimWrapper',
        #                      optimizer=dict(type='Adam', lr=self.cfg['lr'], betas=(0.937, 0.999), weight_decay=0.0005))
        # else:
        #     optimizer = dict(_delete_=True, type='OptimWrapper',
        #                      optimizer=dict(type='AdamW', lr=self.cfg['lr'], betas=(0.937, 0.999), weight_decay=0.0005))

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
            'epochs': self.cfg['end_epoch'],
            'height': self.cfg['imgsz'][0],
            'width': self.cfg['imgsz'][1],
            'num_classes': self.cfg['number_of_class'],
            'lr': self.cfg['lr'],
            'start_factor': self.cfg['initial_lr'] / self.cfg['lr'],
            'minimum_lr': self.cfg['minimum_lr'],
            'warmup_begin': self.cfg['start_epoch'],
            'warmup_end': self.cfg['warmup_epoch'],
            # 'optim_wrapper': optimizer,
            'check_interval': self.cfg['save_period'],
            'nms_threshold': self.cfg['nms_thres'],
        }

        update_python_file(self.cfg['cfg_file'], os.path.join(get_work_dir_path(self.cfg), 'cfg.py'), variables)
        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg.py')

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
                # Load image
                if isinstance(source, str):
                    original_image = cv2.imread(source)
                elif isinstance(source, np.ndarray):
                    original_image = source
                else:
                    raise ValueError

            result = self.model(original_image, show=False, print_result=False, return_vis=True)

        return {"result_image": None,
                "class_list": None,
                "bbox_list": None,
                "score_list": None,
                "rle_list": None}
