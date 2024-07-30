from __future__ import annotations
from typing import Union, Any
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'mmdetection'))
from model_zoo.base.BaseDetectModel import BaseDetectModel
from .BaseMMdetection import BaseMMdetection
from engine.timer import TIMER
import numpy as np
import cv2


class CODETR(BaseMMdetection, BaseDetectModel):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        optimizer = self._build_optimizer(
            weight_decay=0.0001,
            clip_grad=dict(max_norm=0.1, norm_type=2),
            paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
        )
        transforms = self._build_augmentation()
        BaseMMdetection.__init__(self, cfg, optimizer, transforms)
        BaseDetectModel.__init__(self, cfg)

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

            result = self.model(original_image, show=False, print_result=False, return_vis=False)

            class_list = []
            score_list = []
            bbox_list = []

            predictions = result['predictions'][0]
            classes = predictions['labels']
            scores = predictions['scores']
            bboxes = predictions['bboxes']

            for cls, conf, bbox in zip(classes, scores, bboxes):
                if conf < conf_thres:
                    continue

                x = bbox[0]
                y = bbox[1]
                w = bbox[2] - x
                h = bbox[3] - y

                class_list.append(cls)
                score_list.append(conf)
                bbox_list.append(list(map(float, [x, y, w, h])))

        return {
            'class_list': class_list,
            'score_list': score_list,
            'bbox_list': bbox_list
        }
