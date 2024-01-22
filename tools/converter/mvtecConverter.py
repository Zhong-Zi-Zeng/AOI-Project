from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
import cv2
from patchify import patchify

from .baseConverter import BaseConverter
from .jsonParser import jsonParser


class mvtecConverter(BaseConverter):
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
        self.dataset_type = dataset_type  # train or test
        self.patch_size = patch_size
        self.stride = stride
        self.store_none = store_none
        self.generate_dir()

    def generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'mask'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'mask', 'defect'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'good'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'good'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'defect'), exist_ok=True)

    def generate_original(self):
            if self.dataset_type == 'train':
                # =====good=====
                for idx, image_file in enumerate(tqdm(self.image_files_path, total=len(self.image_files_path))):
                    image_name = Path(image_file).stem
                    shutil.copy(image_file,
                                os.path.join(self.output_dir, 'train', 'good', image_name + '.jpg'))
            else:   # test
                # =====good=====
                for idx, image_file in enumerate(tqdm(self.good_image_files_path, total=len(self.good_image_files_path))):
                    image_name = Path(image_file).stem
                    shutil.copy(image_file,
                                os.path.join(self.output_dir, 'test', 'good', image_name + '.jpg'))
                # =====defect=====
                # 依 super 分類
                super_info = {entry['super']: {'super': entry['super'], 'id': entry['id']}
                                for entry in self.classes_name.values()}
                super_info_sort = sorted(list(super_info.items()), key=lambda x: x[1]['id'])
                super_cls = [entry[1]['super'] for entry in super_info_sort]    # ['Scratch', 'Friction', 'Dirty', 'Assembly']  # ['Defect']
                # 創建每個瑕疵的folder
                for super_cls_name in super_cls:
                    os.makedirs(os.path.join(self.output_dir, 'mask', 'defect', super_cls_name), exist_ok=True)
                    os.makedirs(os.path.join(self.output_dir, 'test', 'defect', super_cls_name), exist_ok=True)

                for idx, (image_file, json_file) in enumerate(
                        tqdm(zip(self.defect_image_files_path, self.json_files_path), total=len(self.defect_image_files_path))):
                    # 解析json
                    _, _, mask, classes, _, _ = jsonParser(json_file).parse()

                    # 一張圖可能有多個瑕疵
                    for idx, class_name in enumerate(classes):
                        class_name = class_name[1:]
                        super_name = self.classes_name[class_name]['super']

                        if super_name in super_cls:
                            # image
                            image_name = Path(image_file).stem
                            shutil.copy(image_file,
                                        os.path.join(self.output_dir, 'test', 'defect', super_name, image_name + '.jpg'))
                            # mask
                            cv2.imwrite(os.path.join(self.output_dir, 'mask', 'defect', super_name, image_name + '.jpg'), mask)

    def generate_patch(self):
            if self.dataset_type == 'train':
                # =====good=====
                for idx, image_file in enumerate(tqdm(self.image_files_path, total=len(self.image_files_path))):
                    # 切 patch
                    original_image = cv2.imread(image_file)
                    original_patches = patchify(original_image, (self.patch_size, self.patch_size, 3), step=int(self.patch_size / self.stride))
                    original_patches = original_patches.reshape((-1, self.patch_size, self.patch_size, 3))

                    for i, image_patch in enumerate(original_patches):
                        image_name = f"patch_{idx}_{i}"
                        train_path = os.path.join(self.output_dir, 'train', 'good', image_name + '.jpg')
                        cv2.imwrite(train_path, cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB))
            else:   # test
                # =====good=====
                for idx, image_file in enumerate(tqdm(self.good_image_files_path, total=len(self.good_image_files_path))):
                    # 切 patch
                    original_image = cv2.imread(image_file)
                    original_patches = patchify(original_image, (self.patch_size, self.patch_size, 3), step=int(self.patch_size / self.stride))
                    original_patches = original_patches.reshape((-1, self.patch_size, self.patch_size, 3))

                    for i, image_patch in enumerate(original_patches):
                        image_name = f"patch_{idx}_{i}"
                        train_path = os.path.join(self.output_dir, 'test', 'good', image_name + '.jpg')
                        cv2.imwrite(train_path, cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB))
                # =====defect=====
                # 依 super 分類
                super_info = {entry['super']: {'super': entry['super'], 'id': entry['id']}
                                for entry in self.classes_name.values()}
                super_info_sort = sorted(list(super_info.items()), key=lambda x: x[1]['id'])
                super_cls = [entry[1]['super'] for entry in super_info_sort]    # ['Scratch', 'Friction', 'Dirty', 'Assembly']
                # 創建每個瑕疵的folder
                for super_cls_name in super_cls:
                    os.makedirs(os.path.join(self.output_dir, 'mask', 'defect', super_cls_name), exist_ok=True)
                    os.makedirs(os.path.join(self.output_dir, 'test', 'defect', super_cls_name), exist_ok=True)


                for idx, (image_file, json_file) in enumerate(
                        tqdm(zip(self.defect_image_files_path, self.json_files_path), total=len(self.defect_image_files_path))):
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
                        masks = np.array(results[i]['label']['mask'])
                        classes = results[i]['label']['classes']
                        bboxes = results[i]['label']['bboxes']
                        polygons = results[i]['label']['polygons']
                        processed_image_count = results[i]['processed_image_count']

                        # 一張圖可能有多個瑕疵
                        for idx, (class_name, mask) in enumerate(zip(classes, masks)):
                            class_name = class_name[1:]
                            super_name = self.classes_name[class_name]['super']

                            if super_name in super_cls:
                                # image
                                image_name = f"patch_{processed_image_count}_{i}_{idx}"
                                image_patch.save(os.path.join(self.output_dir, 'test', 'defect', super_name, image_name + '.jpg'))

                                # mask
                                mask_path = os.path.join(self.output_dir, 'mask', 'defect', super_name,image_name + '.jpg')
                                cv2.imwrite(mask_path, mask)