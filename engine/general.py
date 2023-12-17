from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
import os
import yaml
import json

ROOT = os.getcwd()


def load_python(path: str) -> dict:
    """讀取python檔案，並將其解析成dict"""
    data = {}
    with open(path, 'r') as cfg_file:
        exec(cfg_file.read(), data)

    del data['__builtins__']

    return data

def load_yaml(path: str) -> dict:
    assert os.path.isfile(path), 'Can not find this yaml file {}'.format(path)

    with open(path, 'r', encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def save_yaml(path: str, data: dict):
    with open(path, 'w') as file:
        yaml.dump(data, file)


def save_json(path: str,
              data: Union[list | dict],
              indent: Optional[int] = None):
    with open(path, 'w') as file:
        json.dump(data, file, indent=indent)


def get_model_path(__file__: str) -> str:
    """
        將原本指到py檔的路徑往前提一層
        ex:
            __file__ = "D://A/hello.py"
            轉換為
            __file__ = "D://A"
    """
    return str(Path(__file__).parent)

def get_work_dir_path(cfg: dict) -> str:
    """
        返回目前work_dir的路徑
    """
    work_dir_path = os.path.join(os.getcwd(), 'work_dirs', cfg['work_dir_name'])
    return work_dir_path


def get_works_dir_path() -> str:
    """
        返回目前work_dirs的路徑
    """
    return os.path.join(ROOT, 'work_dirs')
