from __future__ import annotations

import matplotlib.pyplot as plt

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


class yoloSegConverter(BaseConverter):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 classes_yaml: str,
                 dataset_type: str,
                 patch_size: Optional[int] = None,
                 stride: Optional[int] = None,
                 store_none: bool = False):
        super().__init__(source_dir, output_dir,  classes_yaml)
        self.source_dir = os.path.join(source_dir, dataset_type)
        self.output_dir = output_dir
        self.dataset_type = dataset_type  # train or test
        self.patch_size = patch_size
        self.stride = stride
        self.store_none = store_none
        self.generate_dir()

    def generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'labels'), exist_ok=True)

    def generate_original(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):

            # 解析json
            image_height, image_width, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # <train or test>
            # image
            image_name = Path(image_file).stem
            shutil.copy(image_file,
                        os.path.join(self.output_dir, self.dataset_type, 'images', image_name + '.jpg'))

            # label
            with open(os.path.join(self.output_dir, self.dataset_type, 'labels', image_name + '.txt'), 'w') as file:
                for idx, polygon in enumerate(polygons):
                    # Normalize polygon to be between 0-1
                    normalized_polygon = polygon / np.array([image_width, image_height])

                    # Extract the class label without the '#'
                    class_name = classes[idx].replace("#", '')

                    # Find the index of a class label
                    # class_idx = self.classes_name.index(class_name)

                    # Find the index of a superclass label
                    superclass_idx = self.classes_name[class_name]['id']

                    # Add the coordinates of each vertex to a list in YOLO format
                    # class, x1, y1, x2, y2, …(Normalize 0–1)
                    yolo_coords = [str(superclass_idx)] + normalized_polygon.flatten().astype(str).tolist()
                    file.write(" ".join(yolo_coords) + "\n")

    def generate_patch(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):

            # 解析json
            image_height, image_width, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # 切patch
            results = BaseConverter._divide_to_patch(self,
                                                     image_file,
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
                mask = np.array(results[i]['label']['mask'])
                classes = results[i]['label']['classes']
                bboxes = results[i]['label']['bboxes']
                polygons = results[i]['label']['polygons']

                processed_image_count = results[i]['processed_image_count']

                # <train or test>
                # image
                image_name = f"patch_{processed_image_count}_{i}"
                image_patch.save(os.path.join(self.output_dir, self.dataset_type, 'images', image_name + '.jpg'))

                # label
                if len(classes) != 0:
                    with open(os.path.join(self.output_dir, self.dataset_type, 'labels', image_name + '.txt'), 'w') as file:
                        for idx, polygon in enumerate(polygons):
                            # Normalize polygon to be between 0-1
                            normalized_polygon = polygon / np.array([image_width, image_height])

                            # Extract the class label without the '#'
                            class_name = classes[idx].replace("#", '')

                            # Find the index of a class label
                            # class_idx = self.classes_name.index(class_name)

                            # Find the index of a superclass label
                            superclass_idx = self.classes_name[class_name]['id']
                            # Add the coordinates of each vertex to a list in YOLO format
                            # class, x1, y1, x2, y2, …(Normalize 0–1)
                            yolo_coords = [str(superclass_idx)] + normalized_polygon.flatten().astype(str).tolist()

                            file.write(" ".join(yolo_coords) + "\n")
                else:
                    with open(os.path.join(self.output_dir, self.dataset_type, 'labels', image_name + '.txt'), 'w') as file:
                        pass

