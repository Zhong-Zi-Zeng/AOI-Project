from model_zoo import BaseInstanceModel, BaseDetectModel
from .general import (get_work_dir_path, get_works_dir_path, load_yaml, save_yaml)
import importlib
import os
import copy
import sys


class Builder:
    def __init__(self, config_path: str, task: str):
        self.config_path = config_path
        self.task = task

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

    def _create_work_dir(self, cfg: dict):
        cfg['work_dir_name'] = os.path.join(self.task, cfg['model_name'])
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

    def _process_base_key(self, config_dir, config):
        """
            將所有config中base的部分進行讀取並合併後，再跟custom config去合併
        """
        merged_config = {}
        if '_base_' in config:
            for base_file in config['_base_']:
                base_path = os.path.join(config_dir, base_file)
                base_config = load_yaml(base_path)
                self._merge_dicts(merged_config, self._process_base_key(config_dir, base_config))
        self._merge_dicts(merged_config, config)
        return merged_config

    def build_config(self) -> dict:
        """
            從給定的config路徑去生成一個完整的config，如果與base config中有重複的key
            則以給定的為主

            Returns:
                config (dict): 最後合併好的config
        """
        custom_config = load_yaml(self.config_path)
        config_dir = os.path.dirname(self.config_path)

        # Load config
        final_config = self._process_base_key(config_dir, custom_config)

        # Create work dir
        self._create_work_dir(final_config)

        # Save final config
        save_yaml(os.path.join(get_work_dir_path(final_config), "final_config"), final_config)

        return final_config

    @staticmethod
    def build_model(config: dict) -> BaseInstanceModel:
        """
            從給定的config中的"name", 去model_zoo中尋找對應的model
        """
        module = importlib.import_module(f'model_zoo.{config["model_dir_name"]}')
        model = getattr(module, config["model_name"], None)

        if model is None:
            raise ValueError("Can not find the model of {}".format(config['model_name']))
        else:
            return model(config)
