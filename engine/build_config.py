import yaml
import os

def merge_dicts(base: dict, custom:dict):
    """
        尋找共同的key並進行替換
        Args:
            base (dict): base的config
            custom (dict): custom的config
    """
    for key, value in custom.items():
        if key in base and isinstance(base[key], dict) and isinstance(custom[key], dict):
            merge_dicts(base[key], custom[key])
        else:
            base[key] = custom[key]

def build_config_from_file(config_path: str) -> dict:
    """
        從給定的config路徑去生成一個完整的config，如果與base config中有重複的key
        則以給定的為主

        Args:
            config_path (str): config路徑
        Returns:
            config (dict): 最後合併好的config
    """

    with open(config_path, 'r') as file:
        custom_config = yaml.load(file, Loader=yaml.FullLoader)

    base_config = {}

    # Load base config
    if '_base_' in custom_config:
        config_dir = os.path.dirname(config_path)
        for base in custom_config['_base_']:
            with open(os.path.join(config_dir, base), 'r') as file:
                base_config_from_file = yaml.load(file, Loader=yaml.FullLoader)
                merge_dicts(base_config, base_config_from_file)

    # Merge custom config into base config
    merge_dicts(base_config, custom_config)

    return base_config

print(build_config_from_file(r"D:\Heng_shared\AOI-Project\configs\yolov7_seg.yaml"))