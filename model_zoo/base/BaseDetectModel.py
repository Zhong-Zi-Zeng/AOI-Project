from __future__ import annotations
from typing import Union, Any, Dict
from abc import ABC, abstractmethod
import numpy as np

class BaseDetectModel(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Update config
        self._config_transform()
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
                result_image (np.array[np.uint8]): 標註後的圖片
                class_list (list[int]): (M, ) 檢測到的類別編號，M為檢測到的物體數量
                score_list (list[float]): (M, ) 每個物體的信心值
                bbox_list (list[int]): (M, 4) 物體的bbox, 需為 x, y, w, h
            }
        """
        pass

    def predict(self,
                source: Union[str | np.ndarray[np.uint8]],
                conf_thres: float = 0.25,
                nms_thres: float = 0.5,
                *args: Any,
                **kwargs: Any) -> dict:
        result = self._predict(source, conf_thres, nms_thres, *args, **kwargs)
        default_key = {'result_image', 'class_list', 'score_list', 'bbox_list'}

        if set(result.keys()) == set(default_key):
            return result
        else:
            raise ValueError("You must return the same key with default keys.")
