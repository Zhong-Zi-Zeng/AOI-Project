from __future__ import annotations
from .baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from copy import deepcopy
from .jsonParser import jsonParser
from tqdm import tqdm
from typing import Optional
import cv2
import json
import os
import numpy as np
import shutil


class cocoConverter(BaseConverter):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 classes_txt: str,
                 dataset_type: str,
                 patch_size: Optional[int] = None):
        super().__init__(source_dir, output_dir, classes_txt)
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.dataset_type = 'val' if dataset_type == 'test' else dataset_type
        self._generate_dir()

    def _generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'train2017'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'val2017'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations'), exist_ok=True)

    def generate_original(self):
        images = []
        anns = []
        cats = [{'id': id, 'name': cls.replace('#', '')} for id, cls in enumerate(self.classes_name)]
        anns_count = 0

        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            h, w, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # image
            image_name = Path(image_file).stem
            shutil.copy(image_file,
                        os.path.join(self.output_dir, self.dataset_type + '2017', image_name + '.jpg'))

            # Label
            images.append({
                'file_name': image_name + '.jpg',
                'height': h,
                'width': w,
                'id': idx
            })

            for cls, bbox, polygon in zip(classes, bboxes, polygons):
                anns.append({
                    'segmentation': np.reshape(polygon, (1, -1)).tolist(),
                    'area': cv2.contourArea(polygon),
                    'iscrowd': 0,
                    'image_id': idx,
                    'bbox': bbox,
                    'category_id': self.classes_name.index(cls.replace('#', '')),
                    'id': anns_count,
                })
                anns_count += 1

        with open(os.path.join(self.output_dir, 'annotations', 'instances_' + self.dataset_type + '2017.json'),
                  'w') as file:
            json.dump({'images': images,
                       'annotations': anns,
                       'categories': cats}, file)

    def generate_patch(self):
        images = []
        anns = []
        cats = [{'id': id, 'name': cls.replace('#', '')} for id, cls in enumerate(self.classes_name)]
        anns_count = 0
        img_id = 0

        for image_file, json_file in tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path)):
            h, w, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # 切patch
            results = BaseConverter.divide_to_patch(self,
                                                    image_file,
                                                    h,
                                                    w,
                                                    mask,
                                                    classes,
                                                    bboxes,
                                                    polygons, self.patch_size)
            # 取有瑕疵的patch
            for i in range(len(results)):
                image_patch = results[i]['image']

                h = results[i]['label']['image_height'][0]
                w = results[i]['label']['image_width'][0]
                mask = np.array(results[i]['label']['mask'])
                classes = results[i]['label']['classes']
                bboxes = results[i]['label']['bboxes']
                polygons = results[i]['label']['polygons']

                processed_image_count = results[i]['processed_image_count']

                # image
                image_name = f"patch_{processed_image_count}_{i}"
                image_patch.save(os.path.join(self.output_dir, self.dataset_type + '2017', image_name + '.jpg'))

                # Label
                images.append({
                    'file_name': image_name + '.jpg',
                    'height': h,
                    'width': w,
                    'id': img_id
                })

                for cls, bbox, polygon in zip(classes, bboxes, polygons):
                    anns.append({
                        'segmentation': np.reshape(polygon, (1, -1)).tolist(),
                        'area': cv2.contourArea(polygon),
                        'iscrowd': 0,
                        'image_id': img_id,
                        'bbox': bbox,
                        'category_id': self.classes_name.index(cls.replace('#', '')),
                        'id': anns_count,
                    })
                    anns_count += 1
                img_id += 1

        with open(os.path.join(self.output_dir, 'annotations', 'instances_' + self.dataset_type + '2017.json'),
                  'w') as file:
            json.dump({'images': images,
                       'annotations': anns,
                       'categories': cats}, file)