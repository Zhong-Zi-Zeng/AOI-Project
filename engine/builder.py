from model_zoo import BaseInstanceModel, BaseDetectModel
from .general import (get_work_dir_path, get_works_dir_path, load_yaml)
import importlib
import os
import copy
import sys


class Builder:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def _merge_dicts(self, base: dict, custom: dict):
        """
            尋找共同的key並進行替換
            Args:
                base (dict): base的config
                custom (dict): custom的config
        """
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(custom[key], dict):
                self._merge_dicts(base[key], custom[key])
            else:
                base[key] = custom[key]

    @staticmethod
    def _create_work_dir(cfg: dict):
        work_dir_path = get_work_dir_path(cfg)

        if not os.path.isdir(work_dir_path):
            os.makedirs(work_dir_path)
        else:
            index = 2
            while True:
                new_work_dir_name = f"{cfg['work_dir_name']}_{index}"
                new_work_dir_path = os.path.join(get_works_dir_path(), new_work_dir_name)
                if not os.path.exists(new_work_dir_path):
                    os.makedirs(new_work_dir_path)
                    cfg['work_dir_name'] = new_work_dir_name
                    break
                else:
                    index += 1

    def build_config(self) -> dict:
        """
            從給定的config路徑去生成一個完整的config，如果與base config中有重複的key
            則以給定的為主

            Returns:
                config (dict): 最後合併好的config
        """

        custom_config = load_yaml(self.config_path)

        base_config = {}

        # Load base config
        if '_base_' in custom_config:
            config_dir = os.path.dirname(self.config_path)
            for base in custom_config['_base_']:
                self._merge_dicts(base_config, load_yaml(os.path.join(config_dir, base)))

        # Merge custom config into base config
        self._merge_dicts(base_config, custom_config)

        # Create work dir
        self._create_work_dir(base_config)

        return base_config

    @staticmethod
    def build_model(config: dict) -> BaseInstanceModel:
        """
            從給定的config中的"name", 去model_zoo中尋找對應的model
        """

        if config['model_name'] == 'Yolov7Seg':
            from model_zoo.yolov7_seg import Yolov7Seg as model
        elif config['model_name'] == 'Yolov7Obj':
            from model_zoo.yolov7_obj import Yolov7Obj as model
        elif config['model_name'] == 'CascadeMaskRCNN':
            from model_zoo.mmdetection import CascadeMaskRCNN as model
        else:
            raise ValueError("Can not find the model of {}".format(config['model_name']))

        return model(config)
