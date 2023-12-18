from __future__ import annotations
from typing import Optional, Union, Any
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'mmdetection'))
from model_zoo.base.BaseInstanceModel import BaseInstanceModel
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, load_python)
from engine.timer import TIMER
from mmdet.apis import DetInferencer
import pycocotools.mask as ms
import numpy as np


class CascadeMaskRCNN(BaseInstanceModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg=cfg)
        self.cfg = cfg

    def _config_transform(self):
        config_dict = load_python(self.cfg['cfg_file'])

        # TODO: Update dataset setting
        # Update dataset setting
        # config_dict['data_root'] = self.cfg['coco_root']
        # config_dict['classes'] = self.cfg['class_names']
        # config_dict['batch_size'] = self.cfg['batch_size']

    def _load_model(self):
        self.model = DetInferencer(model=self.cfg['cfg_file'],
                                   weights=self.cfg['weight'],
                                   show_progress=False)

    def train(self):
        pass

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
            result = self.model(source, show=False, print_result=False, return_vis=True)

            predictions = result['predictions'][0]
            vis = result['visualization'][0]
            labels = predictions['labels']
            scores = predictions['scores']
            rle_list = predictions['masks']

            # TODO: 過濾score小於conf_thres的預測
            _bboxes = []
            for bbox in predictions['bboxes']:
                x = bbox[0]
                y = bbox[1]
                w = bbox[2] - x
                h = bbox[3] - y
                _bboxes.append(list(map(float, [x, y, w, h])))

        return {"result_image": vis,
                "class_list": labels,
                "bbox_list": _bboxes,
                "score_list": scores,
                "rle_list": rle_list}
