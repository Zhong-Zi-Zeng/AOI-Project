from __future__ import annotations
from .baseConverter import BaseConverter
from patchify import patchify
from pathlib import Path
from copy import deepcopy
from .jsonParser import jsonParser
from collections import OrderedDict
from tqdm import tqdm
from typing import Optional
import cv2
import json
import os
import numpy as np
import shutil


class cocoStuffConverter(BaseConverter):
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
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.dataset_type = 'val' if dataset_type == 'test' else dataset_type
        self.stride = stride
        self.store_none = store_none
        self._generate_dir()

    def _generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'annotations', 'val'), exist_ok=True)

    def generate_original(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            h, w, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            # image
            image_name = Path(image_file).stem
            shutil.copy(image_file,
                        os.path.join(self.output_dir, 'images', self.dataset_type + '2017', image_name + '.jpg'))

            # Label
            mask = np.zeros(shape=(h, w), dtype=np.uint8)
            for cls, bbox, polygon in zip(classes, bboxes, polygons):
                class_name = cls.replace('#', '')
                cls_id = self.classes_name[class_name]['id'] + 1  # 數值0要給無類別使用
                cv2.fillPoly(mask, [polygon], color=[cls_id])

            cv2.imwrite(os.path.join(self.output_dir, 'annotations', self.dataset_type + '2017', image_name + '.jpg'), mask)


    def generate_patch(self):
        pass
        # images = []
        # anns = []
        # cats = []
        # for item in sorted(self.classes_name.values(), key=lambda x: x['id']):
        #     cat_dict = {'id': item['id'], 'name': item['super']}
        #     if cat_dict not in cats:
        #         cats.append(cat_dict)
        # anns_count = 0
        # img_id = 0
        #
        # for image_file, json_file in tqdm(zip(self.image_files_path, self.json_files_path),
        #                                   total=len(self.image_files_path)):
        #     h, w, mask, classes, bboxes, polygons = jsonParser(json_file).parse()
        #
        #     # 切patch
        #     results = BaseConverter._divide_to_patch(self,
        #                                              image_file,
        #                                              h,
        #                                              w,
        #                                              mask,
        #                                              classes,
        #                                              bboxes,
        #                                              polygons,
        #                                              self.patch_size, self.stride, self.store_none)
        #     # 取有瑕疵的patch
        #     for i in range(len(results)):
        #         image_patch = results[i]['image']
        #
        #         h = results[i]['label']['image_height'][0]
        #         w = results[i]['label']['image_width'][0]
        #         mask = np.array(results[i]['label']['mask'])
        #         classes = results[i]['label']['classes']
        #         bboxes = results[i]['label']['bboxes']
        #         polygons = results[i]['label']['polygons']
        #
        #         processed_image_count = results[i]['processed_image_count']
        #
        #         # image
        #         image_name = f"patch_{processed_image_count}_{i}"
        #         image_patch.save(os.path.join(self.output_dir, self.dataset_type + '2017', image_name + '.jpg'))
        #
        #         # Label
        #         images.append({
        #             'file_name': image_name + '.jpg',
        #             'height': h,
        #             'width': w,
        #             'id': img_id
        #         })
        #
        #         if len(classes) != 0:
        #             for cls, bbox, polygon in zip(classes, bboxes, polygons):
        #                 class_name = cls.replace('#', '')
        #                 anns.append({
        #                     'segmentation': np.reshape(polygon, (1, -1)).tolist(),
        #                     'area': cv2.contourArea(polygon),
        #                     'iscrowd': 0,
        #                     'image_id': img_id,
        #                     'bbox': bbox,
        #                     'category_id': self.classes_name[class_name]['id'],
        #                     'id': anns_count,
        #                 })
        #                 anns_count += 1
        #         img_id += 1
        #
        # with open(os.path.join(self.output_dir, 'annotations', 'instances_' + self.dataset_type + '2017.json'),
        #           'w') as file:
        #     json.dump({'images': images,
        #                'annotations': anns,
        #                'categories': cats}, file, indent=2)
