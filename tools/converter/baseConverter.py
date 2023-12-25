from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image, ImageDraw
from copy import deepcopy
from .jsonParser import jsonParser
import os
import json
from patchify import patchify
from skimage.util import view_as_windows
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

class BaseConverter(ABC):
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 classes_yaml: str):
        # 生成初始的資料夾
        os.makedirs(output_dir, exist_ok=True)

        with open(classes_yaml, 'r') as file:
            # self.classes_name = [cls.rstrip() for cls in file.readlines()]  # txt
            self.classes_name = yaml.safe_load(file)  # yaml
            # self.classes_name = {cls: data.get('super') for cls, data in classes_data.items()}  # {'Border': 'aaa'...}

        self.image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                                 if self.is_image(os.path.join(source_dir, image_name))]  # 儲存所有image路徑

        self.json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                                if self.is_json(os.path.join(source_dir, json_name))]  # 儲存所有image路徑

        self.processed_image_count = 0  # Record the number of times divide_to_patch is called (so that the patch name is not overwritten)


    def _divide_to_patch(self,
                         image_file,
                         image_height,
                         image_width,
                         mask,
                         classes,
                         bboxes,
                         polygons,
                         patch_size):
        """
            Returns:
                h (int): 圖片的高
                w (int): 圖片的寬
                mask (np.ndarray):  0、255的二值化影像 [H, W]
                classes (list): 這個json檔中包含的瑕疵類別 [N, ]
                bboxes (list): 每個瑕疵對應的bbox，格式為x, y, w, h [N, 4]
                polygons (list[np.ndarray]): 每個瑕疵對應的polygon [N, M, 2]
        """
        # Check whether image_h and image_w are divisible by patch_size
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("patch_size cannot divide the original image.")

        # Information about defective patches
        patch_num = (image_height//patch_size) * (image_width//patch_size)
        labels = {idx: {'image_height': [],
                        'image_width': [],
                        'mask': [],
                        'classes': [],
                        'bboxes': [],
                        'polygons': []
                        } for idx in range(patch_num)}

        # Read the original image and cut the patch
        original_image = cv2.imread(image_file)
        original_patches = patchify(original_image, (patch_size, patch_size, 3), step=patch_size)
        original_patches = original_patches.reshape((-1, patch_size, patch_size, 3))

        # Create empty masks
        N = len(polygons)  # Number of defects
        black_canvas = np.zeros(shape=(N, image_height, image_width), dtype=np.uint8)

        # Process one image, multiple defects
        for idx, (polygon, cls) in enumerate(zip(polygons, classes)):
            # Draw each polygon on the corresponding mask
            cv2.fillPoly(black_canvas[idx], [polygon], color=(255, 255, 255))

            # Divide into patch sizes
            patches_mask = view_as_windows(black_canvas[idx], (patch_size, patch_size), step=patch_size)
            patches_mask = patches_mask.reshape((-1, patch_size, patch_size))  # (P, H, W)

            # Find the index of the patch containing the defect and save the information
            flaws_idx = set(np.where(patches_mask > 0)[0])  # 取p
            for patch_idx in flaws_idx:
                # mask
                patch = patches_mask[patch_idx]
                # polygon
                contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 只要外部輪廓，所有輪廓點
                patch_polygon = np.vstack(contours).reshape((-1, 2))
                # bbox
                x, y, w, h = cv2.boundingRect(patch_polygon)

                # info
                labels[patch_idx]['image_height'].append(patch_size)
                labels[patch_idx]['image_width'].append(patch_size)
                labels[patch_idx]['mask'].append(patch)
                labels[patch_idx]['classes'].append(cls)
                labels[patch_idx]['bboxes'].append([x, y, w, h])
                labels[patch_idx]['polygons'].append(patch_polygon)

        # Save all defective patch images
        results = []
        for patch_idx in labels:
            if labels[patch_idx]['image_height']:   # not empty
                patch_image = Image.fromarray(original_patches[patch_idx])
                results.append({'image': patch_image, 'label': labels[patch_idx], 'processed_image_count': self.processed_image_count})

        self.processed_image_count += 1
        return results

    @abstractmethod
    def generate_original(self):
        pass

    @abstractmethod
    def generate_patch(self):
        pass

    @staticmethod
    def is_image(image_path: str) -> bool:
        """
           判斷該路徑是不是圖像
           Arg:
                image_path: 影像路徑
           Return:
                True or False
       """
        _allow_format = ['.jpg', '.png', '.bmp']
        return Path(image_path).suffix in _allow_format

    @staticmethod
    def is_json(json_path: str) -> bool:
        """
            判斷該路徑是不是json檔
            Arg:
                json_path: json檔路徑
            Return:
                True or False
        """
        return Path(json_path).suffix == '.json'
