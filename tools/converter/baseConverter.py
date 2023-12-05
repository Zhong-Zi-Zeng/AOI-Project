from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy
from .jsonParser import jsonParser
import os
import json


class BaseConverter(ABC):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 classes_txt: str):
        # 生成初始的資料夾
        os.makedirs(output_dir, exist_ok=True)

        with open(classes_txt, 'r') as file:
            self.classes_name = [cls.rstrip() for cls in file.readlines()]  # 儲存所有類別名稱

        self.image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                                 if self.is_image(os.path.join(source_dir, image_name))]  # 儲存所有image路徑

        self.json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                                if self.is_json(os.path.join(source_dir, json_name))]  # 儲存所有image路徑

    @abstractmethod
    def generate_original(self):
        pass

    @abstractmethod
    def generate_patch(self):
        pass

    @staticmethod
    def is_image(image_path: str) -> bool:
        """
           判斷該路徑是不是圖像
           Arg:
                image_path: 影像路徑
           Return:
                True or False
       """
        _allow_format = ['.jpg', '.png', '.bmp']
        return Path(image_path).suffix in _allow_format

    @staticmethod
    def is_json(json_path: str) -> bool:
        """
            判斷該路徑是不是json檔
            Arg:
                json_path: json檔路徑
            Return:
                True or False
        """
        return Path(json_path).suffix == '.json'
