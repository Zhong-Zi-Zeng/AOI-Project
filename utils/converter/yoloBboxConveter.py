from __future__ import annotations
from .baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from tqdm import tqdm
from .jsonParser import jsonParser
from typing import Optional
import cv2
import json
import os
import numpy as np
import shutil
from collections import defaultdict
import argparse


class yoloBboxConverter(BaseConverter):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 classes_txt: str,
                 dataset_type: str,
                 patch_size: Optional[int] = None):
        super().__init__(source_dir, output_dir, classes_txt)
        self.source_dir = os.path.join(source_dir, dataset_type)
        self.output_dir = output_dir
        self.classes_txt = classes_txt
        self.patch_size = patch_size
        self.dataset_type = 'val' if dataset_type == 'test' else dataset_type   # train or val
        self.generate_dir()

    def generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'labels', 'val'), exist_ok=True)

    def generate_original(self):
        image_paths = []

        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            # 解析json
            image_height, image_width, _, classes, bboxes, _ = jsonParser(json_file).parse()

            # <images & labels train> or <images & labels val>
            # image
            shutil.copy(image_file,
                        os.path.join(self.output_dir, 'images', self.dataset_type, str(idx) + '.jpg'))

            # 存所有train or val圖片的路徑
            image_paths.append(os.path.join(self.output_dir, 'images', self.dataset_type, str(idx) + '.jpg'))
            # train_list.txt or val_list.txt
            with open(os.path.join(self.output_dir, self.dataset_type + '_list.txt'), 'w') as file:
                file.write('\n'.join(image_paths))


            # label
            with open(os.path.join(self.output_dir, 'labels', self.dataset_type, str(idx) + '.txt'), 'w') as file:
                for idx, bbox in enumerate(bboxes):
                    # Extract the class label without the '#'
                    class_name = classes[idx][1:]

                    # Find the index of a class label
                    class_idx = self.classes_name.index(class_name)

                    # Add the coordinates of each vertex to a list in YOLO format
                    # class, x, y, w, h(Normalize 0–1)
                    yolo_bbox = [str(class_idx)] + list(map(str, bbox))
                    file.write(" ".join(yolo_bbox) + "\n")


    def generate_patch(self):
        pass


