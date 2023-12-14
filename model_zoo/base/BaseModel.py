from __future__ import annotations
from typing import Union, Any, Dict
from abc import ABC, abstractmethod
import os
import numpy as np
import time
import torch
import contextlib


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, name: str, t=0.0):
        self.name = name
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt * 1000  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class BaseModel(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # 紀錄處理時間
        self.dt = (Profile(name='Total process time(ms):'),
                   Profile(name='Preprocess Time(ms):'),
                   Profile(name='Inference Time(ms):'),
                   Profile(name='NMS Time(ms):'))

    def get_timer(self) -> tuple:
        """
            返回每個階段的執行時間
        """
        return self.dt

    def _config_transform(self):
        """
            將輸入的config轉換到各自需要的格式
        """
        pass

    @abstractmethod
    def train(self):
        """
            Run每個model自己的training command
        """
        pass

    def predict(self,
                source: Union[str | np.ndarray[np.uint8]],
                conf_thres: float = 0.25,
                nms_thres: float = 0.5,
                *args: Any,
                **kwargs: Any):
        result = self._predict(source, conf_thres, nms_thres, *args, **kwargs)
        default_key = {'result_image', 'class_list', 'score_list', 'bbox_list', 'polygon_list'}

        if set(result.keys()) == set(default_key):
            return result
        else:
            raise ValueError("You must return the same key with default keys.")

    @abstractmethod
    def _predict(self,
                 source: Union[str | np.ndarray[np.uint8]],
                 conf_thres: float = 0.25,
                 nms_thres: float = 0.5,
                 *args: Any,
                 **kwargs: Any
                 ) -> dict:
        """
        1. Inference時會用到
        2. 需再4個階段中插入Profile去計時
        Example:
        >>>with self.dt[0]:
        >>>        # Pre-process (Start)
        >>>        with self.dt[1]:
        >>>            ...
        >>>        # Pre-process (End)
        >>>
        >>>        # Inference (Start)
        >>>        with self.dt[2]:
        >>>            ...
        >>>        # Inference (End)
        >>>
        >>>        # NMS-process (Start)
        >>>        with self.dt[3]:
        >>>            ...
        >>>        # NMS-process (End)

        Args:
            source: 照片路徑或是已讀取的照片
            conf_thres: 信心度的threshold
            nms_thres: nms的threshold

        Returns:
            返回一個字典，格式如下:
            {
                result_image (np.array[np.uint8]): 標註後的圖片
                class_list (list[int]): (M, ) 檢測到的類別編號，M為檢測到的物體數量
                score_list (list[float]): (M, ) 每個物體的信心值
                bbox_list (list[int]): (M, 4) 物體的bbox, 需為 x, y, w, h
                polygon_list (list[np.array]): (M, N, 2) 物體的輪廓座標xy，N為polygon數量
            }
        """
        pass
