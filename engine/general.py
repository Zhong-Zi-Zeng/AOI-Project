from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
from PIL import Image, ImageDraw
import pycocotools.mask as ms
import numpy as np
import os
import yaml
import json
import re
import ast
import astor

ROOT = os.getcwd()


def check_path(path: str) -> bool:
    return os.path.exists(path)


def polygon_to_rle(polygon: np.ndarray, height: int, width: int) -> dict:
    """Convert polygon to RLE format"""
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    rle = ms.encode(np.asfortranarray(np.array(mask)))
    rle['counts'] = str(rle['counts'], encoding='utf-8')

    return rle


def load_python(path: str) -> dict:
    """讀取python檔案，並將其解析成dict"""
    data = {}
    with open(path, 'r') as cfg_file:
        exec(cfg_file.read(), data)

    del data['__builtins__']

    return data

def update_python_file(old_python_file_path, new_python_file_path, variables):
    """使用AST来更新Python配置文件中的变量。"""

    assert old_python_file_path.endswith('.py'), "old_python_file_path must be a python file."
    assert new_python_file_path.endswith('.py'), "new_python_file_path must be a python file."

    with open(old_python_file_path, 'r') as file:
        content = file.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):

            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in variables:
                    value = variables[target.id]
                    if isinstance(value, str):
                        node.value = ast.Str(s=value)
                    elif isinstance(value, int) or isinstance(value, float):
                        node.value = ast.Num(n=value)
                    elif isinstance(value, list):
                        node.value = ast.List(elts=[ast.Str(s=elt) for elt in value], ctx=ast.Load())
                    elif isinstance(value, dict):
                        keys = [ast.Str(s=k) for k in value.keys()]
                        values = [ast.Str(s=v) for v in value.values()]
                        node.value = ast.Dict(keys=keys, values=values)
                    elif isinstance(value, bool):
                        node.value = ast.NameConstant(value=value)


    new_content = astor.to_source(tree)

    with open(new_python_file_path, 'w') as file:
        file.write(new_content)


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


def get_model_path(cfg: dict) -> str:
    """
        提取當前model的資料夾路徑
    """
    return os.path.join(os.getcwd(), 'model_zoo', cfg['model_dir_name'])


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
