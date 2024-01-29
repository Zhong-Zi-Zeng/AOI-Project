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
                 classes_yaml: str,
                 dataset_type: str,
                 format: str,
                 patch_size: Optional[int] = None,
                 stride: Optional[int] = None,
                 store_none: bool = False):
        super().__init__(source_dir, output_dir, classes_yaml, dataset_type, format)
        self.source_dir = os.path.join(source_dir, dataset_type)
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.dataset_type = 'val' if dataset_type == 'test' else dataset_type  # train or val
        self.stride = stride
        self.store_none = store_none
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

        for idx, (image_file) in enumerate(tqdm(self.image_files_path, total=len(self.image_files_path))):
            # image
            image_name = Path(image_file).stem
            shutil.copy(image_file,
                        os.path.join(self.output_dir, 'images', self.dataset_type, image_name + '.jpg'))

            # 存所有train or val圖片的路徑
            image_paths.append(os.path.join('./', 'images', self.dataset_type, image_name + '.jpg'))

            # train_list.txt or val_list.txt
            with open(os.path.join(self.output_dir, self.dataset_type + '_list.txt'), 'w') as file:
                file.write('\n'.join(image_paths).replace('\\', '/'))

            # 依照image file去找對應的json檔，如果沒有找到就跳過
            json_file = image_file.replace(Path(image_file).suffix, '.json')

            if not os.path.isfile(json_file):
                continue

            # 解析json
            image_height, image_width, _, classes, bboxes, _ = jsonParser(json_file).parse()

            # label
            with open(os.path.join(self.output_dir, 'labels', self.dataset_type, image_name + '.txt'), 'w') as file:
                for idx, bbox in enumerate(bboxes):
                    # Normalize
                    x, y, w, h = bbox

                    x = 0 if x < 0 else x
                    y = 0 if y < 0 else y
                    x = image_width if x > image_width else x
                    y = image_height if y > image_height else y

                    # x, y, w, h -> cx, cy, w, h
                    bbox[0] = (x + w / 2) / image_width
                    bbox[1] = (y + h / 2) / image_height
                    bbox[2] /= image_width
                    bbox[3] /= image_height

                    # Extract the class label without the '#'
                    class_name = classes[idx].replace("#", '')

                    # Find the index of a superclass label
                    superclass_idx = self.classes_name[class_name]['id']

                    # Add the coordinates of each vertex to a list in YOLO format
                    # class, x, y, w, h(Normalize 0–1)
                    yolo_bbox = [str(superclass_idx)] + list(map(str, bbox))
                    file.write(" ".join(yolo_bbox) + "\n")

    def generate_patch(self):
        image_paths = []

        for idx, (image_file) in enumerate(tqdm(self.image_files_path, total=len(self.image_files_path))):
            # 依照image file去找對應的json檔，如果沒有找到就跳過
            json_file = image_file.replace(Path(image_file).suffix, '.json')
            if not os.path.isfile(json_file):
                continue

            # 解析json
            image_height, image_width, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # 切patch
            results = self.process_patch(image_file,
                                         image_height,
                                         image_width,
                                         mask,
                                         classes,
                                         bboxes,
                                         polygons,
                                         self.patch_size, self.stride, self.store_none)
            # 取有瑕疵的patch
            for i in range(len(results)):
                image_patch = results[i]['image']

                image_height = results[i]['label']['image_height'][0]
                image_width = results[i]['label']['image_width'][0]
                classes = results[i]['label']['classes']
                bboxes = results[i]['label']['bboxes']

                processed_image_count = results[i]['processed_image_count']

                # <images & labels train> or <images & labels val>
                # image
                image_name = f"patch_{processed_image_count}_{i}"
                image_patch.save(os.path.join(self.output_dir, 'images', self.dataset_type, image_name + '.jpg'))

                # 存所有train or val圖片的路徑
                image_paths.append(os.path.join('./', 'images', self.dataset_type, image_name + '.jpg'))

                # train_list.txt or val_list.txt
                with open(os.path.join(self.output_dir, self.dataset_type + '_list.txt'), 'w') as file:
                    file.write('\n'.join(image_paths).replace('\\', '/'))

                # label
                with open(os.path.join(self.output_dir, 'labels', self.dataset_type, image_name + '.txt'),
                          'w') as file:
                    for idx, bbox in enumerate(bboxes):
                        x, y, w, h = bbox

                        # Normalize xywh -> cxcywh
                        bbox[0] = (x + w / 2) / image_width
                        bbox[1] = (y + h / 2) / image_height
                        bbox[2] /= image_width
                        bbox[3] /= image_height

                        # Extract the class label without the '#'
                        class_name = classes[idx].replace("#", '')

                        # Find the index of a superclass label
                        superclass_idx = self.classes_name[class_name]['id']

                        # Add the coordinates of each vertex to a list in YOLO format
                        # class, x, y, w, h(Normalize 0–1)
                        yolo_bbox = [str(superclass_idx)] + list(map(str, bbox))
                        file.write(" ".join(yolo_bbox) + "\n")


