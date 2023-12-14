from __future__ import annotations
from typing import Union, Optional
import os
import yaml
import json

ROOT = os.getcwd()


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
