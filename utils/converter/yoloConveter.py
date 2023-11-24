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


class yoloConverter(BaseConverter):
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
        self.dataset_type = dataset_type
        self.generate_dir()

    def generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'label'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'label'), exist_ok=True)

    def generate_original(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            # 解析json
            image_height, image_width, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # <train>
            # image
            shutil.copy(image_file,
                        os.path.join(self.output_dir, self.dataset_type, 'image', str(idx) + '.jpg'))

            # label
            with open(os.path.join(self.output_dir, self.dataset_type, 'label', str(idx) + '.txt'), 'w') as file:
                for idx, polygon in enumerate(polygons):
                    # Normalize the coordinates to be between 0-1
                    normalized_coords = polygon / np.array([image_width, image_height])

                    # Extract the class label without the '#'
                    class_name = classes[idx][1:]

                    # Find the index of a class label
                    class_idx = self.classes_name.index(class_name)

                    # Add the coordinates of each vertex to a list in YOLO format
                    # class, x1, y1, x2, y2, …(Normalize 0–1)
                    yolo_coords = [str(class_idx)] + normalized_coords.flatten().astype(str).tolist()
                    file.write(" ".join(yolo_coords) + "\n")

            categories_count = defaultdict(int)
            categories_filenames = defaultdict(list)

            category = class_name
            categories_count[category] += 1

            filename = str(idx) + '.jpg'
            categories_filenames[category].append(filename)

    def generate_patch(self):
        pass


