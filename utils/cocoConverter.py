from __future__ import annotations
from baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from copy import deepcopy
from jsonParser import JsonParser
from tqdm import tqdm
from typing import Optional
import cv2
import json
import os
import numpy as np
import argparse
import shutil


class cocoConverter(BaseConverter):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 classes_txt: str,
                 patch_size: Optional[int] = None):
        super().__init__(source_dir, output_dir, classes_txt)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self._generate_dir()

    def _generate_dir(self):
        os.mkdir(os.path.join(self.output_dir, 'train2017'))
        os.mkdir(os.path.join(self.output_dir, 'test2017'))
        os.mkdir(os.path.join(self.output_dir, 'annotations'))

    def generate_original(self):
        images = []
        anns = []
        cats = [{'id': id, 'name': cls} for id, cls in enumerate(self.classes_name)]
        anns_count = 0

        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            h, w, mask, classes, bboxes, polygons = JsonParser(json_file).parse()

            # image
            shutil.copy(image_file,
                        os.path.join(self.output_dir, 'train2017', str(idx) + '.jpg'))

            # Label
            images.append({
                'file_name': str(idx) + '.jpg',
                'height': h,
                'width': w,
                'id': idx
            })

            for cls, bbox, polygon in zip(classes, bboxes, polygons):
                anns.append({
                    'segmentation': polygon.tolist(),
                    'area': 0,
                    'iscrowd': 0,
                    'image_id': idx,
                    'bbox': bbox,
                    'category_id': self.classes_name.index(cls.replace('#', '')),
                    'id': anns_count,
                })
                anns_count += 1

        with open(os.path.join(self.output_dir, 'annotations', 'instances_train2017.json'), 'w') as file:
            json.dump({'images': images,
                       'annotations': anns,
                       'categories': cats}, file)

    def generate_patch(self):
        pass


cocoConverter(source_dir='./source_w', output_dir='./white', classes_txt='./classes_w.txt').generate_original()
