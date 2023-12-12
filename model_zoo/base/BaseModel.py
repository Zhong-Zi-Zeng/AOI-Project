from __future__ import annotations
from typing import Union, Any, Dict
from abc import ABC, abstractmethod
import numpy as np
import time
import torch
import contextlib


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


class BaseInference(ABC):
    def __init__(self):
        # 紀錄處理時間
        self.dt = (Profile(), Profile(), Profile(), Profile())

    def timer(self) -> dict:
        """
            返回每個階段的執行時間
        """
        timer_name = ['Total process time(ms):',
                      'Preprocess time(ms):',
                      'Inference time(ms):',
                      'NMS time(ms):']
        return {name: time.dt * 1000 for time, name in zip(self.dt, timer_name)}

    def run(self,
            source: Union[str | np.ndarray[np.uint8]],
            conf_thres: float = 0.25,
            nms_thres: float = 0.5,
            *args: Any,
            **kwargs: Any):
        result = self._run(source, conf_thres, nms_thres, *args, **kwargs)
        default_key = {'result_image', 'class_list', 'score_list', 'bbox_list', 'polygon_list'}

        if set(result.keys()) == set(default_key):
            return result
        else:
            raise ValueError("You must return the same key with default keys.")

    @abstractmethod
    def _run(self,
             source: Union[str | np.ndarray[np.uint8]],
             conf_thres: float = 0.25,
             nms_thres: float = 0.5,
             *args: Any,
             **kwargs: Any
             ) -> dict:
        """
        需再4個階段中插入Profile去計時，
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
