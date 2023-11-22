from __future__ import annotations
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import *
from typing import Optional
from transformers import SamProcessor
from torch.utils.data import DataLoader
from utils.augmentation import *
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import os
import cv2
import yaml
import torch
import json


class BaseDataset(Dataset):
    def __init__(self, processor, trans: Optional[list] = None):
        self.processor = processor
        self.trans = trans

        self.images_path = []  # 儲存輸入影像的path
        self.masks_path = []  # 儲存gt影像的path

    @staticmethod
    def _is_image(image_path: str) -> bool:
        """
           判斷該路徑是不是圖像
            Arg:
                image_path: 影像路徑
            Return:
                True or False
       """
        allow_format = ['.jpg', '.png', '.bmp']
        return pathlib.Path(image_path).suffix in allow_format

    @staticmethod
    def _is_json(json_path: str) -> bool:
        """
            判斷該路徑是不是json檔
            Arg:
                json_path: json檔路徑
            Return:
                True or False
        """
        return pathlib.Path(json_path).suffix == '.json'

    @staticmethod
    def _get_point_from_json(text: dict) -> list:
        """
            解析json檔中所有的point，並將其存在list中返回
            Args:
                text: json檔的內容，須為字典型態

            Return:
                points = [[x1, y1], [x2, y2], ...]

        """
        assert isinstance(text, dict)

        objects = text['Objects']  # list
        points = []

        for obj in objects:
            points.append(list(map(float, obj['Layers'][0]['Shape']['Point0'].split(','))))

        return points

    @staticmethod
    def _get_point_from_mask(mask: np.ndarray) -> list:
        """
            從mask中生成points
            Args:
                mask: (H, W) 的二值化影像

            Return:
                points = [[x1, y1], [x2, y2], ...]
        """
        points = []

        # 找到輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍歷每個輪廓
        for contour in contours:
            # 計算輪廓的中心點
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append([cX, cY])

        return points


class CustomDataset(BaseDataset):
    def __init__(self,
                 image_path: str,
                 mask_path: str,
                 use_points: bool,
                 use_boxes: bool,
                 processor,
                 trans: Optional[list] = None):
        super().__init__(processor, trans)
        """
            Args:
                image_path: image資料夾路徑，裡面要有rgb圖片
                mask_path: mask資料夾路徑，裡面要有mask圖片
                use_points: 是否使用points作為prompt
                use_boxes: 是否使用boxes作為prompt (box一張圖片只能有一個!!)
                processor: 處理data的processor
                trans: torch中的transforms
        """

        self.use_points = use_points
        self.use_boxes = use_boxes

        # 讀取資料夾下所有的圖片
        self.images_path = [os.path.join(image_path, image_name) for image_name in os.listdir(image_path)
                            if self._is_image(os.path.join(image_path, image_name))]

        # 讀取資料夾下所有的mask
        self.masks_path = [os.path.join(mask_path, mask_name) for mask_name in os.listdir(mask_path)]

        assert len(self.images_path) == len(self.masks_path)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_file = self.images_path[idx]
        mask_file = self.masks_path[idx]

        # 讀取檔案
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_file)

        # augmentation
        tr_image = transforms.ToTensor()(image)
        gt_mask = transforms.ToTensor()(gt_mask)

        if self.trans is not None:
            for aug in self.trans:
                tr_image, gt_mask = aug(tr_image, gt_mask)
        gt_point = self._get_point_from_mask(gt_mask[0].cpu().numpy().astype(np.uint8))

        if self.use_points:
            inputs = self.processor(tr_image, input_points=[gt_point], return_tensors="pt", do_rescale=False)
        else:
            inputs = self.processor(tr_image, return_tensors="pt", do_rescale=False)

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs['ground_truth_mask'] = gt_mask[0]
        inputs['original_image'] = image

        return inputs


# For test
if __name__ == "__main__":
    with open('./configs/config_1.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)


    augmentation = [
        RandomFlip(prob=config['random_flip_prob']),
        ColorEnhance(brightness_gain=config['brightness_gain'],
                     contrast_gain=config['contrast_gain'],
                     saturation_gain=config['saturation_gain'],
                     hue_gain=config['hue_gain']),
        RandomPerspective(degrees=config['degrees'],
                          translate=config['translate'],
                          scale=config['scale'],
                          shear=config['shear'])
    ]

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    train_dataset = CustomDataset(image_path="D:/AOI/ControllerDataset/white/patch-train",
                                  mask_path="D:/AOI/ControllerDataset/white/patch-train-gt",
                                  use_points=False,
                                  use_boxes=False,
                                  processor=processor,
                                  trans=augmentation)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # batch = next(iter(train_dataloader))
    # for k, v in batch.items():
    #     print(k, v.shape)

    for batch in train_dataloader:
        # print(batch['pixel_values'].shape)
        # print(batch['ground_truth_mask'].shape)
        # print(batch['input_points'].shape)
        # print()
        pass
