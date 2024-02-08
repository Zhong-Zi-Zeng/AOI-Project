from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))

from engine.builder import Builder
from typing import Optional, Tuple
from pycocotools.coco import COCO
from colorama import Fore, Back, Style, init
from model_zoo import BaseInstanceModel
from faster_coco_eval.extra import PreviewResults
from engine.general import (get_work_dir_path, save_json)
from engine.timer import TIMER
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import openpyxl
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('Model evaluation script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    parser.add_argument('--excel', '-e', type=str,
                        help='Existing Excel file.'
                             'If given the file, this script will append new value in the given file.'
                             'Otherwise, this script will create a new Excel file depending on the task type.')

    parser.add_argument('--detected_result', '-d', type=str,
                        help='Existing detected file.'
                             'If given the file, this script will be evaluate directly.')

    parser.add_argument('--dir_name', type=str,
                        help='The name of work dir.')

    parser.add_argument('--multi_conf', action="store_true",
                        help='Using multi confidence threshold lie in range 0.3 to 0.9.')

    return parser


class Evaluator:
    def __init__(self,
                 model: BaseInstanceModel,
                 cfg: dict,
                 detected_result: Optional[str] = None,
                 excel_path: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.detected_result = detected_result

        self.coco_gt = COCO(os.path.join(cfg["coco_root"], 'annotations', 'instances_val2017.json'))
        self.coco_dt = self.coco_gt.loadRes()

    def eval(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # Create builder
    builder = Builder(config_path=args.config, task='eval', work_dir_name=args.dir_name)

    # Build config
    cfg = builder.build_config()

    # Build model
    model = builder.build_model(cfg)

    # Use multi confidence threshold
    if args.multi_conf:
        confidences = np.arange(0.3, 1.0, 0.1)
    else:
        confidences = [cfg['conf_thres']]

    for conf in confidences:
        cfg['conf_thres'] = conf

        # Build evaluator
        evaluator = Evaluator(model=model, cfg=cfg, excel_path=args.excel, detected_result=args.detected_result)

        evaluator.eval()
