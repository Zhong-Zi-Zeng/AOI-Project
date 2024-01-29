from __future__ import annotations
from typing import Union, Any, Dict
from abc import ABC, abstractmethod
from engine.general import check_path
import time
import threading

class BaseModel(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Update config
        self._config_transform()

        # Start scheduler



    @staticmethod
    def _check_weight_path(weight_path: str):
        assert check_path(weight_path), f"Can not find the weight. " \
                                        f"Please check the path."

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
