from __future__ import annotations

import random
from typing import Optional
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from utils.augmentation import create_augmentation
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import albumentations as A
import cv2
import yaml


def to_torch(func):
    """
        Change all the output to the tensor type
    """

    def wrapper(*args, **kwargs):
        return [torch.from_numpy(np.array(rs)) for rs in func(*args, **kwargs)]

    return wrapper


class CustomDataset(Dataset):
    def __init__(self,
                 root: Optional[str | Path],
                 ann_file: Optional[str | Path],
                 use_points: bool,
                 use_boxes: bool,
                 trans: A.Compose,
                 trans_with_bboxes: A.Compose,
                 trans_with_points: A.Compose,
                 trans_with_bboxes_points: A.Compose):
        """
            Args:
                root: coco的image資料夾路徑
                ann_file: coco的標註文件路徑
                use_points: 是否使用points作為prompt
                use_boxes: 是否使用boxes作為prompt (box一張圖片只能有一個!!)
                trans: 只針對image和mask做aug
                trans_with_bboxes: 針對image、mask和bboxes做aug
                trans_with_points: 針對image、mask和points做aug
                trans_with_bboxes_points: 針對image、mask、bboxes和points做aug
        """
        self.root = root
        self.coco = COCO(ann_file)
        self.use_points = use_points
        self.use_boxes = use_boxes
        self.trans = trans
        self.trans_with_bboxes = trans_with_bboxes
        self.trans_with_points = trans_with_points
        self.trans_with_bboxes_points = trans_with_bboxes_points
        self.img_ids = self._filter_image()

    def _filter_image(self) -> list[int]:
        """Filter out images that didn't has label"""
        all_img_ids = self.coco.getImgIds()
        result = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ids=ann_ids)
            if len(ann_info):
                result.append(img_id)
        return result

    @staticmethod
    def _get_center_points_from_mask(mask: np.ndarray) -> list:
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

    @staticmethod
    def _get_bbox_from_polygons(polygons: list, category_id: list) -> list:
        """
            從polygon中生成bbox，如果有不符合的bbox則與category_id一起剔除
            Args:
                polygons: [[x1, y1, x2, y2, ....], [x1, y2, x2, y2, ...], ...]
                category_id: [cls1 , cls2, ....]
            Return:
                boxes = [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
        boxes = []
        for idx, polygon in enumerate(polygons):
            x, y, w, h = cv2.boundingRect(np.array(polygon, dtype=np.int32).reshape((-1, 2)))
            x1, y1, x2, y2 = x, y, x + w, y + h

            # check box
            if x2 <= x1 or y2 <= y1:
                del category_id[idx]
                continue

            boxes.append([x1, y1, x2, y2])
        return boxes

    @to_torch
    def augment_with_boxes_and_points(self, image, polygons, category_id, gt_mask):
        boxes = self._get_bbox_from_polygons(polygons, category_id)
        points = self._get_center_points_from_mask(gt_mask.astype(np.uint8))

        transformed_result = self.trans_with_bboxes_points(image=image,
                                                           bboxes=boxes,
                                                           bbox_classes=category_id,
                                                           keypoints=points,
                                                           mask=gt_mask)

        return [transformed_result['image'],
                transformed_result['mask'],
                transformed_result['keypoints'],
                transformed_result['bboxes']]

    @to_torch
    def augment_with_boxes(self, image, polygons, category_id, gt_mask):
        boxes = self._get_bbox_from_polygons(polygons, category_id)
        transformed_result = self.trans_with_bboxes(image=image,
                                                    bboxes=boxes,
                                                    bbox_classes=category_id,
                                                    mask=gt_mask)
        return [transformed_result['image'],
                transformed_result['mask'],
                transformed_result['bboxes']]

    @to_torch
    def augment_with_points(self, image, gt_mask):
        points = self._get_center_points_from_mask(gt_mask.astype(np.uint8))
        transformed_result = self.trans_with_points(image=image,
                                                    keypoints=points,
                                                    mask=gt_mask)
        return [transformed_result['image'],
                transformed_result['mask'],
                transformed_result['keypoints']]

    @to_torch
    def augment(self, image, gt_mask):
        transformed_result = self.trans(image=image,
                                        mask=gt_mask)
        return [transformed_result['image'],
                transformed_result['mask']]

    def __len__(self):
        return len(self.img_ids)

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
        # =========================Input image=========================
        image_info = self.coco.loadImgs(ids=[self.img_ids[idx]])[0]
        image_path = Path(self.root) / image_info['file_name']
        height, width = image_info['height'], image_info['width']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # =========================Label=========================
        ann_ids = self.coco.getAnnIds(imgIds=[self.img_ids[idx]])
        ann_info = self.coco.loadAnns(ids=ann_ids)
        gt_mask = np.zeros(shape=(height, width), dtype=bool)
        polygons = []
        category_id = []

        # Collate polygon、mask、category
        if self.use_boxes or self.use_points:
            ann = random.choice(ann_info)
            mask = self.coco.annToMask(ann).astype(bool)
            polygons.append(self._get_polygon_from_mask(mask.astype(np.uint8)))
            gt_mask = np.logical_or(mask, gt_mask)
            category_id.append(ann['category_id'])
        else:
            for ann in ann_info:
                mask = self.coco.annToMask(ann).astype(bool)
                polygons.append(self._get_polygon_from_mask(mask.astype(np.uint8)))
                gt_mask = np.logical_or(mask, gt_mask)
                category_id.append(ann['category_id'])

        gt_mask = gt_mask.astype(np.uint8) * 255
        points = []
        boxes = []

        if self.use_points and self.use_boxes:
            tr_image, tr_mask, points, boxes = self.augment_with_boxes_and_points(image, polygons, category_id, gt_mask)
        elif self.use_boxes:
            tr_image, tr_mask, boxes = self.augment_with_boxes(image, polygons, category_id, gt_mask)
        elif self.use_points:
            tr_image, tr_mask, points = self.augment_with_points(image, gt_mask)
        else:
            tr_image, tr_mask = self.augment(image, gt_mask)

        # avoid error
        if self.use_boxes and len(boxes) == 0:
            print(f"Could not find label in {image_path}.")
            return self.__getitem__(idx + 1)

        return {
            'tr_image': tr_image,
            'original_image': image,
            'gt_mask': tr_mask,
            'points': points,
            'boxes': boxes
        }

    @staticmethod
    def collate_fn(batch: list):
        try:
            tr_images = torch.stack([item['tr_image'] for item in batch])
            gt_mask = torch.stack([item['gt_mask'] > 0 for item in batch]).to(float)
            original_images = [item['original_image'] for item in batch]
            points = np.stack([np.array(item['points']) for item in batch])
            boxes = np.stack([np.array(item['boxes']) for item in batch])
        except:
            raise ValueError("After augmentation causes bbox or point disappear,"
                             " please adjust 'translate', 'scale', 'shear' and 'perspective' to more small value.")

        return {
            'tr_image': tr_images,
            'original_image': original_images,
            'gt_mask': gt_mask,
            'points': None if all(len(point) == 0 for point in points) else points,
            'boxes': None if all(len(box) == 0 for box in boxes) else boxes,
        }


# For test
if __name__ == "__main__":
    with open('configs/vit-b.yaml', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    use_points = False
    use_boxes = True

    coco_root = Path(config['coco_root']) / 'train2017'
    coco_ann_file = Path(config['coco_root']) / "annotations" / "instances_train2017.json"
    train_dataset = CustomDataset(root=coco_root,
                                  ann_file=coco_ann_file,
                                  use_points=use_points,
                                  use_boxes=use_boxes,
                                  **create_augmentation(hyp=config))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=train_dataset.collate_fn)
    pbar = tqdm(train_dataloader)
    for idx, batch in enumerate(pbar):
        tr_image = batch['tr_image'][0].cpu().numpy()
        original_image = batch['original_image'][0]
        mask = batch['gt_mask'][0].cpu().numpy()
        #
        tr_image = cv2.resize(tr_image, (1024, 1024))
        original_image = cv2.resize(original_image, (1024, 1024))
        # print(batch['boxes'])
        bboxes = batch['boxes'][0]
        # print(bboxes)
        # points = batch['points'][0]
        # print(batch['points'])
        for bbox in bboxes:
            cv2.rectangle(tr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          thickness=1, color=(0, 255, 255))

        # for point in points:
        #     cv2.circle(tr_image, (int(point[0]), int(point[1])), thickness=1, color=(0, 255, 255), radius=3)

        cv2.imshow('', tr_image)
        cv2.imshow('ori', original_image)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
