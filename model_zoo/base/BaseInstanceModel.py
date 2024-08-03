from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from typing import Union, Any, Dict
from abc import abstractmethod
from model_zoo.base.BaseModel import BaseModel
from engine.general import (xywh_to_xyxy, rle_to_polygon, polygon_to_rle)
from engine.timer import TIMER
from torchvision.ops import nms
from patchify import patchify
import numpy as np
import torch
import cv2


class BaseInstanceModel(BaseModel):
    DEFAULT_KEY = {'class_list', 'score_list', 'bbox_list', 'rle_list'}

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    @abstractmethod
    def _predict(self,
                 source: Union[str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 nms_thres: float = 0.5,
                 *args: Any,
                 **kwargs: Any
                 ) -> dict:
        """
        1. Inference和Evaluation時會用到

        Args:
            source: 照片路徑或是已讀取的照片
            conf_thres: 信心度的threshold
            nms_thres: nms的threshold

        Returns:
            返回一個字典，格式如下:
            {
                class_list (list[int]): (M, ) 檢測到的類別編號，M為檢測到的物體數量
                score_list (list[float]): (M, ) 每個物體的信心值
                bbox_list (list[int]): (M, 4) 物體的bbox, 需為 x, y, w, h
                rle_list (list[dict["size":list, "counts": str]]): (M, ) RLE編碼格式
            }
        """
        pass

    def _plot_bbox_and_mask(self,
                            source: Union[str | np.ndarray[np.uint8]],
                            results: dict) -> np.ndarray[np.uint8]:
        """
        繪製模型預測結果
        Args:
            source: 照片路徑或是已讀取的照片
            results: 預測結果
        Returns:
            繪製後的圖片
        """
        # Load image
        if isinstance(source, str):
            original_image = cv2.imread(source)
        elif isinstance(source, np.ndarray):
            original_image = source
        else:
            raise ValueError

        for cls, bbox, conf, rle in zip(results['class_list'], results['bbox_list'], results['score_list'],
                                        results['rle_list']):
            # Draw bounding box and mask
            text = f'{self.class_names[int(cls)]} {conf:.2f}'
            self.plot_one_box_mask(image=original_image,
                                   xywh_bbox=bbox,
                                   text=text,
                                   rle=rle,
                                   color=self.class_color[int(cls)])

        return original_image

    def predict(self,
                source: Union[str | np.ndarray[np.uint8]],
                conf_thres: float = 0.25,
                nms_thres: float = 0.5,
                return_vis: bool = True,
                verbose: bool = False,
                *args: Any,
                **kwargs: Any) -> dict:

        if self.cfg['use_patch']:
            class_list = []
            score_list = []
            bbox_list = []
            rle_list = []

            # Load image
            if isinstance(source, str):
                original_image = cv2.imread(source)
            else:
                original_image = source
            result_image = np.copy(original_image)

            # Check whether both the height of image and the width of image are divisible by patch_size
            image_height, image_width, _ = original_image.shape
            if image_height % self.cfg['patch_size'] != 0 or image_width % self.cfg['patch_size'] != 0:
                pad_height = self.cfg['patch_size'] - (image_height % self.cfg['patch_size'])
                pad_width = self.cfg['patch_size'] - (image_width % self.cfg['patch_size'])
                original_image = np.pad(original_image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

            # Divide to patch (nf_height_patch, nf_width_patch, patch_size, patch_size, 3)
            original_patches = patchify(original_image,
                                        patch_size=(self.cfg['patch_size'], self.cfg['patch_size'], 3),
                                        step=self.cfg['step'])[:, :, 0, ...]

            # Adjust the coordinate of bbox and polygon
            nf_width_patch = original_patches.shape[1]
            nf_height_patch = original_patches.shape[0]

            for row_idx in range(nf_height_patch):
                for col_idx in range(nf_width_patch):
                    x_offset = col_idx * self.cfg['step']
                    y_offset = row_idx * self.cfg['step']
                    result = self._predict(original_patches[row_idx][col_idx], conf_thres, nms_thres, *args, **kwargs)

                    for bbox, conf, cls, rle in zip(result['bbox_list'], result['score_list'], result['class_list'],
                                                    result['rle_list']):
                        bbox[0] += x_offset
                        bbox[1] += y_offset

                        polygon = rle_to_polygon(rle)
                        polygon = np.hstack(polygon)
                        polygon = np.array(polygon, dtype=np.float32).reshape((-1, 2))
                        polygon[:, 0] += x_offset
                        polygon[:, 1] += y_offset
                        rle = polygon_to_rle(polygon, image_height, image_width)

                        bbox_list.append(bbox)
                        score_list.append(conf)
                        class_list.append(cls)
                        rle_list.append(rle)

            # NMS
            if class_list and bbox_list and score_list:
                indices = nms(boxes=torch.FloatTensor(np.array(xywh_to_xyxy(bbox_list))),
                              scores=torch.FloatTensor(np.array(score_list)),
                              iou_threshold=0.0).cpu().numpy()

                class_list = np.array(class_list)[indices].tolist()
                bbox_list = np.array(bbox_list)[indices].tolist()
                score_list = np.array(score_list)[indices].tolist()
                rle_list = np.array(rle_list)[indices].tolist()

            # Draw bbox
            for bbox, conf, cls, rle in zip(bbox_list, score_list, class_list, rle_list):
                text = f'{self.class_names[int(cls)]} {conf:.2f}'
                self.plot_one_box_mask(image=result_image,
                                       xywh_bbox=bbox,
                                       text=text,
                                       rle=rle,
                                       color=self.class_color[int(cls)])

            result = {
                'result_image': result_image,
                'class_list': class_list,
                'score_list': score_list,
                'bbox_list': bbox_list,
                'rle_list': rle_list
            }
        # If not use patch to predict
        else:
            result = self._predict(source, conf_thres, nms_thres, *args, **kwargs)

        if return_vis:
            result['result_image'] = self._plot_bbox_and_mask(source, result)

        # Print timer
        if verbose:
            print('\n\n')
            for timer in TIMER:
                print(f"{timer.name:15s} {timer.dt:.3f}", end=' | ')
            print('\n\n')

        if self.DEFAULT_KEY.issubset(result.keys()):
            return result
        else:
            raise ValueError("You must return the same key with default keys.")
