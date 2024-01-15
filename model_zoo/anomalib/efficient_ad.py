from __future__ import annotations, absolute_import

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'anomalib'))

from typing import Union, Any
from model_zoo.base.BaseDetectModel import BaseDetectModel
from engine.timer import TIMER
from engine.general import (get_model_path, get_work_dir_path, load_yaml, save_yaml)
import numpy as np
import cv2
import subprocess


class EfficientAD(BaseDetectModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def _config_transform(self):
        # 讀取程式中的config
        config = load_yaml(self.cfg['cfg_file'])

        # 更新(custom調整的config)
        config['dataset']['path'] = self.cfg['dataset_dir']

        config['dataset']['abnormal_dir'] = []  # Clear
        config['dataset']['mask_dir'] = []
        for idx, cls_name in enumerate(self.cfg['class_names']):
            config['dataset']['abnormal_dir'].append(f'test/defect/{cls_name}')
            config['dataset']['mask_dir'].append(f'mask/defect/{cls_name}')

        config['model']['lr'] = self.cfg['lr']
        config['project']['path'] = os.path.join(get_work_dir_path(self.cfg))   # result
        config['trainer']['min_epochs'] = self.cfg['start_epoch']
        config['trainer']['max_epochs'] = self.cfg['end_epoch']
        config['dataset']['train_batch_size'] = self.cfg['batch_size']
        config['dataset']['eval_batch_size'] = self.cfg['batch_size']
        config['dataset']['image_size'] = self.cfg['imgsz'][0]
        config['trainer']['num_nodes'] = self.cfg['device']

        # 儲存
        save_yaml(os.path.join(get_work_dir_path(self.cfg), 'cfg.yaml'), config)
        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg.yaml')

        pass

    def train(self):
        """
            Run每個model自己的training command
        """
        subprocess.run([
            'python', os.path.join(get_model_path(self.cfg), 'tools', 'train.py'),
            '--config', self.cfg['cfg_file']
        ])

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
            with TIMER[2]:
                pass

            # ----------------------------Inference (End)----------------------------

            # ----------------------------NMS-process (Start)----------------------------
            with TIMER[3]:
                pass

            # ----------------------------NMS-process (End)----------------------------

            # ----------------------------Post-process (Start)----------------------------
            # For evaluation








            return {
                'result_image': None,
                'class_list': None,
                'score_list': None,
                'bbox_list': None
            }


