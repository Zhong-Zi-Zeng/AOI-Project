from model_zoo.base.BaseModel import BaseModel
import yaml
import importlib
import os


class Builder:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def merge_dicts(self, base: dict, custom: dict):
        """
            尋找共同的key並進行替換
            Args:
                base (dict): base的config
                custom (dict): custom的config
        """
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(custom[key], dict):
                self.merge_dicts(base[key], custom[key])
            else:
                base[key] = custom[key]

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    def build_config(self) -> dict:
        """
            從給定的config路徑去生成一個完整的config，如果與base config中有重複的key
            則以給定的為主

            Returns:
                config (dict): 最後合併好的config
        """

        custom_config = self._load_config(self.config_path)

        base_config = {}

        # Load base config
        if '_base_' in custom_config:
            config_dir = os.path.dirname(self.config_path)
            for base in custom_config['_base_']:
                with open(os.path.join(config_dir, base), 'r') as file:
                    base_config_from_file = yaml.load(file, Loader=yaml.FullLoader)
                    self.merge_dicts(base_config, base_config_from_file)

        # Merge custom config into base config
        self.merge_dicts(base_config, custom_config)

        return base_config

    @staticmethod
    def build_model(config: dict) -> BaseModel:
        model_zoo = importlib.import_module('model_zoo')
        model = getattr(model_zoo, config['name'], None)

        if model is None:
            raise ValueError("Can not find the model of {}".format(config['name']))
        else:
            return model(config)
