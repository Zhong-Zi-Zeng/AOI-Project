from __future__ import annotations
from baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from tqdm import tqdm
from jsonParser import JsonParser
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
                 patch_size: Optional[int] = None):

        super().__init__(source_dir, output_dir, classes_txt)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.classes_txt = classes_txt
        self.patch_size = patch_size
        self.generate_dir()

    def generate_dir(self):
        os.mkdir(os.path.join(self.output_dir, 'train'))
        os.mkdir(os.path.join(self.output_dir, 'test'))
        os.mkdir(os.path.join(self.output_dir, 'train', 'image'))
        os.mkdir(os.path.join(self.output_dir, 'train', 'label'))
        os.mkdir(os.path.join(self.output_dir, 'test', 'image'))
        os.mkdir(os.path.join(self.output_dir, 'test', 'label'))

    def generate_original(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            # 解析json
            image_height, image_width, mask, classes, bboxes, polygons = JsonParser(json_file).parse()

            # <train>
            # image
            shutil.copy(image_file,
                        os.path.join(self.output_dir, 'train', 'image', str(idx) + '.jpg'))

            # label
            with open(os.path.join(self.output_dir, 'train', 'label', str(idx) + '.txt'), 'w') as file:
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

            # <test>
            # image

            # label

            categories_count = defaultdict(int)
            categories_filenames = defaultdict(list)

            category = class_name
            categories_count[category] += 1

            filename = str(idx) + '.jpg'
            categories_filenames[category].append(filename)

    def generate_patch(self):
        pass


a = yoloConverter(source_dir='./source_w', output_dir='./white', classes_txt='classes_w.txt')
