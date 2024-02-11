from __future__ import annotations
from typing import Union, Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np

from engine.general import check_path
import random
import cv2


class BaseModel(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # class names
        self.class_names = self.cfg['class_names']

        # class color
        class_color_cfg = cfg.get('class_color')

        if class_color_cfg is None:
            self.class_color = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
        else:
            self.class_color = class_color_cfg

        # Update config
        self._config_transform()

    @staticmethod
    def _check_weight_path(weight_path: str):
        """
            檢查weight路徑是否存在
        """
        assert check_path(weight_path), f"Can not find the weight. " \
                                        f"Please check the path."

    @staticmethod
    def plot_one_box_mask(image: np.ndarray,
                          color: Optional[list[int]] = None,
                          xywh_bbox: Optional[list[int]] = None,
                          polygon: Optional[np.ndarray[int]] = None,
                          text: Optional[str] = None):
        """
            繪製bbox、class name、mask到圖像中

            Args:
                image (np.ndarray): 原始圖像
                color (list[int]): 一個list裡面包含3個int元素，對應RGB，範圍介於 0 ~ 255
                xywh_bbox (list[int | float]): 包含x、y、w、h的座標
                polygon (np.ndarray[int]): N x 2 的 x、y座標
                text (str): 附加在bbox、mask左上方的文字
        """
        tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness

        if color is None:
            color = [random.randint(0, 255) for _ in range(3)]

        if xywh_bbox is not None:
            c1, c2 = (int(xywh_bbox[0]), int(xywh_bbox[1])), (
                int(xywh_bbox[2]) + int(xywh_bbox[0]), int(xywh_bbox[3]) + int(xywh_bbox[1]))
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

        if polygon is not None:
            cv2.fillPoly(image, [polygon], color=color)

        if text is not None:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, text, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    @abstractmethod
    def _config_transform(self):
        """
            將config轉換到各自需要的格式
        """
        pass

    @abstractmethod
    def _load_model(self):
        """
            載入model，為後面的inference或是evaluate使用
        """
        pass

    @abstractmethod
    def train(self):
        """
            Run每個model自己的training command
        """
        pass
