from __future__ import annotations

import torch
import albumentations as A
from torch.utils.data import Dataset
from torchvision import transforms as T
from pycocotools.coco import COCO
from typing import Optional
from torch.utils.data import DataLoader
from transformers import SamProcessor
from utils.augmentation import *
from pathlib import Path
# from utils.general import collate_fn
import numpy as np
import cv2
import yaml

torch.manual_seed(10)
np.random.seed(10)

class CustomDataset(Dataset):
    def __init__(self,
                 root: Optional[str | Path],
                 ann_file: Optional[str | Path],
                 use_points: bool,
                 use_boxes: bool,
                 trans: A.Compose):
        """
            Args:
                root: coco的image資料夾路徑
                ann_file: coco的標註文件路徑
                use_points: 是否使用points作為prompt
                use_boxes: 是否使用boxes作為prompt (box一張圖片只能有一個!!)
                trans: torch中的transforms
        """
        self.root = root
        self.coco = COCO(ann_file)
        self.use_points = use_points
        self.use_boxes = use_boxes
        self.trans = trans

    @staticmethod
    def _get_points_from_mask(mask: np.ndarray) -> list:
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

    @staticmethod
    def _get_polygon_from_mask(mask: np.ndarray) -> list:
        """
            從mask中生成polygon
            Args:
                mask: (H, W) 的二值化影像

            Return:
                points = [x1, y1, x2, y2, ...]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if contour.size >= 6:
                polygons.append(contour.flatten())

        return polygons

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, idx):
        """
        Returns:
            {
                tr_image (Tensor): [1, C, 1024, 1024]
                original_image (Tensor): [1, H, W, C],
                ground_truth_mask (Tensor): [1, H, W],
                points (Tensor): [1, N, 2],
                boxes (Tensor): [1, N, 4],
            }
        """

        points = None
        boxes = [[0, 0, 0, 0]]

        # =======Input image=======
        image_info = self.coco.loadImgs(ids=[idx])[0]
        image_path = Path(self.root) / image_info['file_name']
        height, width = image_info['height'], image_info['width']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ======= Label =======
        ann_ids = self.coco.getAnnIds(imgIds=[idx])
        ann_info = self.coco.loadAnns(ids=ann_ids)
        has_label = len(ann_info) > 0
        gt_mask = np.zeros(shape=(height, width), dtype=bool)
        polygons = []

        for ann in ann_info:
            mask = self.coco.annToMask(ann).astype(bool)
            polygons.append(self._get_polygon_from_mask(mask.astype(np.uint8)))
            gt_mask = np.logical_or(mask, gt_mask)

        gt_mask = gt_mask.astype(np.float32) * 255

        # If you use points be the prompt
        if self.use_points and has_label:
            points = self._get_points_from_mask(gt_mask.astype(np.uint8))

        # If you use boxes be the prompt
        if self.use_boxes and has_label:
            boxes = []
            for polygon in polygons:
                x, y, w, h = cv2.boundingRect(np.array(polygon, dtype=np.int32).reshape((-1, 2)))
                boxes.append([x, y, x + w, y + h])

        # Augmentation
        if self.use_boxes and self.use_points:
            transformed_result = self.trans(image=image,
                                            bboxes=boxes,
                                            bbox_classes=[1] * len(boxes),
                                            keypoints=points,
                                            mask=gt_mask)
        elif self.use_boxes:
            transformed_result = self.trans(image=image,
                                            bboxes=boxes,
                                            bbox_classes=[1] * len(boxes),
                                            mask=gt_mask)
        elif self.use_points:
            transformed_result = self.trans(image=image,
                                            keypoints=points,
                                            mask=gt_mask)
        else:
            transformed_result = self.trans(image=image,
                                            mask=gt_mask)

        tr_image = torch.from_numpy(transformed_result['image']).permute(2, 0, 1)
        tr_mask = torch.from_numpy(transformed_result['mask'])

        if self.use_points:
            points = torch.from_numpy(np.array(transformed_result.get('keypoints')))
        if self.use_boxes:
            boxes = torch.from_numpy(np.array(transformed_result.get('bboxes')))

        return {
            'tr_image': tr_image,
            'original_image': image,
            'ground_truth_mask': tr_mask,
            'points': points,
            'boxes': boxes
        }

    def collate_fn(self, batch: list):
        def convert_to_array(batch):
            tr_images = torch.stack([item['tr_image'] for item in batch])
            original_images = [item['original_image'] for item in batch]
            ground_truth_masks = torch.stack([item['ground_truth_mask'] for item in batch])
            points = np.stack([np.array(item['points']) for item in batch])
            boxes = np.stack([np.array(item['boxes']) for item in batch])

            return {
                'tr_image': tr_images,
                'original_image': original_images,
                'ground_truth_mask': ground_truth_masks,
                'points': None if all(point is None for point in points) else points,
                'boxes': None if all(bbox is None for bbox in boxes) else boxes,
            }

        if not self.use_points and not self.use_boxes:
            return convert_to_array(batch)

        if self.use_points:
            max_points = max(len(item['points']) for item in batch)

            for item in batch:
                num_point_padding = max_points - len(item['points'])
                if num_point_padding > 0:
                    item['points'] = np.vstack([item['points'], np.zeros((num_point_padding, 2))])

        if self.use_boxes:
            max_boxes = max(len(item['boxes']) for item in batch)

            for item in batch:
                num_box_padding = max_boxes - len(item['boxes'])
                if num_box_padding > 0:
                    item['boxes'] = np.vstack([item['boxes'], np.zeros((num_box_padding, 4))])

        return convert_to_array(batch)


# For test
if __name__ == "__main__":
    with open('./configs/config_1.yaml', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    use_points = False
    use_boxes = True

    if use_boxes and use_points:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
            keypoint_params=A.KeypointParams(format='xy'),
            additional_targets={'mask': 'image'})
    elif use_boxes:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
            additional_targets={'mask': 'image'})
    elif use_points:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], keypoint_params=A.KeypointParams(format='xy'),
            additional_targets={'mask': 'image'})
    else:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], additional_targets={'mask': 'image'})

    coco_root = Path(config['coco_root']) / 'train2017'
    coco_ann_file = Path(config['coco_root']) / "annotations" / "instances_train2017.json"
    train_dataset = CustomDataset(root=coco_root,
                                  ann_file=coco_ann_file,
                                  use_points=use_points,
                                  use_boxes=use_boxes,
                                  trans=trans)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=False,
                                  collate_fn=train_dataset.collate_fn)
    #
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=4,
    #                               shuffle=False)

    for batch in train_dataloader:
        # print(batch['pixel_values'].shape)
        # print(len(batch['original_image']))
        print()
    # pass
