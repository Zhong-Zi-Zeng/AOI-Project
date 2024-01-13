from __future__ import annotations
from typing import Optional, Union, Any
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'mmdetection'))
from model_zoo.base.BaseInstanceModel import BaseInstanceModel
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, load_python, update_python_file)
from engine.timer import TIMER
from mmdet.apis import DetInferencer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycocotools.mask as ms
import numpy as np
import subprocess
import cv2


class Mask2Former(BaseInstanceModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)
        self.cfg = cfg

    def _config_transform(self):
        config_dict = load_python(self.cfg['cfg_file'])

        # Optimizer
        if self.cfg['optimizer'] == 'SGD':
            optimizer = dict(_delete_=True, type='OptimWrapper',
                             optimizer=dict(type='SGD', lr=self.cfg['lr'], momentum=0.937, weight_decay=0.0001),
                             clip_grad=dict(max_norm=0.1, norm_type=2),
                             paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
        elif self.cfg['optimizer'] == 'Adam':
            optimizer = dict(_delete_=True, type='OptimWrapper',
                             optimizer=dict(type='Adam', lr=self.cfg['lr'], betas=(0.937, 0.999), weight_decay=0.0001),
                             clip_grad=dict(max_norm=0.1, norm_type=2),
                             paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))
        else:
            optimizer = dict(_delete_=True, type='OptimWrapper',
                             optimizer=dict(type='AdamW', lr=self.cfg['lr'], betas=(0.937, 0.999), weight_decay=0.0001),
                             clip_grad=dict(max_norm=0.1, norm_type=2),
                             paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

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
            'num_things_classes': self.cfg['number_of_class'],
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
            'python', os.path.join(get_model_path(self.cfg), 'tools', 'train.py'),
            self.cfg['cfg_file'],
            '--work-dir', get_work_dir_path(self.cfg)
        ])

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> list:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if contour.size >= 6:
                polygons.append(contour.flatten().tolist())

        return polygons

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

            fig, ax = plt.subplots(1)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.imshow(original_image[..., ::-1])

            result = self.model(original_image, show=False, print_result=False, return_vis=True)

            class_list = []
            score_list = []
            bbox_list = []
            rle_list = []

            predictions = result['predictions'][0]
            classes = predictions['labels']
            scores = predictions['scores']
            rles = predictions['masks']
            bboxes = predictions['bboxes']

            for cls, conf, bbox, rle in zip(classes, scores, bboxes, rles):
                if conf < conf_thres:
                    continue

                polygons = self._mask_to_polygon(ms.decode(rle))

                for polygon in polygons:
                    poly = np.reshape(np.array(polygon), (-1, 2))
                    color = list(np.random.uniform(0, 255, size=(3,)))
                    x, y, w, h = cv2.boundingRect(poly)

                    # For mask
                    cv2.fillPoly(original_image, [poly], color=color)

                    # For bbox
                    cv2.putText(original_image, self.cfg['class_names'][cls], (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                color, 1, cv2.LINE_AA)
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), color=color, thickness=2)

                    class_list.append(cls)
                    score_list.append(conf)
                    bbox_list.append(list(map(float, [x, y, w, h])))
                    rle_list.append(rle)

        return {"result_image": original_image,
                "class_list": class_list,
                "bbox_list": bbox_list,
                "score_list": scores,
                "rle_list": rle_list}
