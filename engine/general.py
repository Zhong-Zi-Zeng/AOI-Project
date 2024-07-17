from __future__ import annotations
from typing import Union, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import pycocotools.mask as ms
from io import BytesIO
import numpy as np
import os
import shutil
import random
import yaml
import json
import re
import ast
import astor
import cv2
import torch

ROOT = os.getcwd()


def check_path(path: str) -> bool:
    return os.path.exists(path)


def allowed_file(filename: str):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'json'}

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_image_to_numpy(image) -> np.ndarray:
    in_memory_file = BytesIO()
    image.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return image


def check_gpu_available(cfg: dict):
    gpus = torch.cuda.device_count()

    # check device
    if gpus == 0 and cfg['device'] != 'cpu':
        raise Exception('No GPU found, please set --device cpu')
    elif cfg["device"].isdigit() and int(cfg["device"]) >= gpus:
        raise Exception(f'--device {cfg["device"]} not found, available devices is {gpus - 1}')


def get_device(device: Union[str: int]) -> torch.device:
    device = torch.device(int(device) if device.isdigit() and device != 'cpu' else device)
    return device


def get_class_names_and_colors(cfg: dict) -> Tuple[list[str], list[list]]:
    assert cfg.get('coco_root') is not None, "Please set the 'coco_root' in the config file."

    # Class name
    coco = COCO(os.path.join(cfg['coco_root'], 'annotations', 'instances_train.json'))
    class_names = [info['name'] for info in coco.loadCats(coco.getCatIds())]

    # Class color
    class_color_cfg = cfg.get('class_color')
    if class_color_cfg is None:
        class_color = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
    else:
        class_color = class_color_cfg

    return class_names, class_color


def mask_to_polygon(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygons.append(contour.flatten().tolist())

    return polygons


def polygon_to_rle(polygon: np.ndarray, height: int, width: int) -> dict:
    """Convert polygon to RLE format"""
    mask = Image.new('L', (width, height), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    rle = ms.encode(np.asfortranarray(np.array(mask)))
    rle['counts'] = str(rle['counts'], encoding='utf-8')

    return rle


def rle_to_polygon(rle):
    masked_arr = ms.decode(rle)
    return mask_to_polygon(masked_arr)


def xywh_to_xyxy(bboxes: Union[list | np.ndarray]):
    _bbox = np.array(bboxes, dtype=np.float32)

    if _bbox.ndim == 1:
        _bbox = _bbox[None, ...]

    _bbox[:, 2] = _bbox[:, 0] + _bbox[:, 2]
    _bbox[:, 3] = _bbox[:, 1] + _bbox[:, 3]

    return _bbox


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


def load_json(path: str):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


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


def copy_logfile_to_work_dir(cfg: dict):
    """
        將log檔案複製到當前work_dir底下
    """
    shutil.copyfile('./output.log', get_work_dir_path(cfg) + '/output.log')


def clear_logfile():
    """
        將外部的logfile檔案清空
    """
    with open('./output.log', 'w') as f:
        f.write('')


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
