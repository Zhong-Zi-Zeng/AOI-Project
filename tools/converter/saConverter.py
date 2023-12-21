from __future__ import annotations
from .baseConverter import BaseConverter
from patchify import patchify
from tqdm import tqdm
from .jsonParser import jsonParser
from typing import Optional
from pathlib import Path
import cv2
import os
import numpy as np
import shutil


class saConverter(BaseConverter):
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
        self.dataset_type = dataset_type

        self._generate_dir()

    def _generate_dir(self):
        os.makedirs(os.path.join(self.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'train', 'label'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'image'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'test', 'label'), exist_ok=True)

    def divide_to_patch(self, large_image: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        """
            將傳入的圖片拆成patch
            Return:
                patches: [H, W, 3] or [H, W], N為patch的數量
        """
        h, w = large_image.shape[:2]

        assert h % self.patch_size == 0 and w % self.patch_size == 0

        if len(large_image.shape) == 3:
            patches = patchify(large_image, patch_size=(self.patch_size, self.patch_size, 3), step=self.patch_size)
            return np.reshape(patches, (-1, self.patch_size, self.patch_size, 3))
        elif len(large_image.shape) == 2:
            patches = patchify(large_image, patch_size=(self.patch_size, self.patch_size), step=self.patch_size)
            return np.reshape(patches, (-1, self.patch_size, self.patch_size))
        else:
            raise ValueError

    def generate_original(self):
        for idx, (image_file, json_file) in enumerate(
                tqdm(zip(self.image_files_path, self.json_files_path), total=len(self.image_files_path))):
            h, w, mask, classes, bboxes, polygons = jsonParser(json_file).parse()
            image_name = Path(image_file).stem

            # image
            shutil.copy(image_file,
                        os.path.join(self.output_dir, self.dataset_type, 'image', image_name + '.jpg'))

            # label
            cv2.imwrite(os.path.join(self.output_dir, self.dataset_type, 'label', image_name + '.jpg'),
                        mask)

    def divide_to_patch(self):
        assert self.patch_size is not None, 'You need to assign the size of patch'
        name = 0

        for (image_file, json_file) in tqdm(zip(self.image_files_path, self.json_files_path),
                                            total=len(self.image_files_path)):
            h, w, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

            patch_mask = self.divide_to_patch(mask)
            patch_image = self.divide_to_patch(cv2.imread(image_file))

            # 將沒有異常mask的地方去掉
            valid_indices = [i for i, mask in enumerate(patch_mask) if mask.max() != 0]
            filtered_image = patch_image[valid_indices]
            filtered_mask = patch_mask[valid_indices]

            for i in range(len(filtered_mask)):
                cv2.imwrite(os.path.join(self.output_dir, self.dataset_type, 'image', str(name) + '.jpg'),
                            filtered_image[i])
                cv2.imwrite(os.path.join(self.output_dir, self.dataset_type, 'label', str(name) + '.jpg'),
                            filtered_mask[i])
                name += 1


# saConverter(source_dir='./source', output_dir='./black', patch_size=1024).generate_patch()
