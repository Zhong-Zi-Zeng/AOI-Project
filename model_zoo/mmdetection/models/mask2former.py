from __future__ import annotations
from typing import Union, Any
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'model_zoo', 'mmdetection'))
from model_zoo.base.BaseInstanceModel import BaseInstanceModel
from .BaseMMdetection import BaseMMdetection
from engine.general import (get_work_dir_path, get_model_path, load_python, update_python_file,
                            mask_to_polygon, rle_to_polygon)
from engine.timer import TIMER
from torchvision.ops import nms
import pycocotools.mask as ms
import numpy as np
import torch
import cv2


class Mask2Former(BaseMMdetection, BaseInstanceModel):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        optimizer = self._build_optimizer(
            weight_decay=0.0001,
            clip_grad=dict(max_norm=0.1, norm_type=2),
            paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
        )
        transforms = self._build_augmentation()
        BaseMMdetection.__init__(self, cfg, optimizer, transforms)
        BaseInstanceModel.__init__(self, cfg)

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

            class_list = []
            score_list = []
            bbox_xywh_list = []
            bbox_xxyy_list = []
            rle_list = []
            polygon_list = []

            predictions = result['predictions'][0]
            classes = predictions['labels']
            scores = predictions['scores']
            rles = predictions['masks']

            for cls, conf, rle in zip(classes, scores, rles):
                if conf < conf_thres:
                    continue
                polygons = mask_to_polygon(ms.decode(rle))

                for polygon in polygons:
                    poly = np.reshape(np.array(polygon, dtype=np.int32), (-1, 2))
                    try:
                        x, y, w, h = cv2.boundingRect(poly)
                    except:
                        print('Unable to generate a box from the given polygon.')
                        continue
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    class_list.append(cls)
                    score_list.append(conf)
                    bbox_xywh_list.append(list(map(float, [x, y, w, h])))
                    bbox_xxyy_list.append(list(map(float, [x1, y1, x2, y2])))
                    rle_list.append(rle)
                    polygon_list.append(poly)

            # NMS
            if bbox_xxyy_list and class_list and score_list and class_list:
                with TIMER[3]:
                    indices = nms(boxes=torch.FloatTensor(bbox_xxyy_list),
                                  scores=torch.FloatTensor(score_list),
                                  iou_threshold=nms_thres).cpu().numpy()

                class_list = np.array(class_list)[indices].tolist()
                bbox_xywh_list = np.array(bbox_xywh_list)[indices].tolist()
                score_list = np.array(score_list)[indices].tolist()
                rle_list = np.array(rle_list)[indices].tolist()
                polygon_list = np.array(polygon_list, dtype=object)[indices]

                for conf, cls, bbox, poly in zip(score_list, class_list, bbox_xywh_list, polygon_list):
                    # Draw bounding box and mask
                    text = f'{self.class_names[int(cls)]} {conf:.2f}'
                    self.plot_one_box_mask(image=original_image,
                                           xywh_bbox=bbox,
                                           text=text,
                                           polygon=poly,
                                           color=self.class_color[int(cls)])

        return {"result_image": original_image,
                "class_list": class_list,
                "bbox_list": bbox_xywh_list,
                "score_list": score_list,
                "rle_list": rle_list}
