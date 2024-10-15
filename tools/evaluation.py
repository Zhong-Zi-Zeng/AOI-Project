from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))

import numpy as np
import pandas as pd
import argparse
import platform
from typing import Optional, Tuple, List, Union, Dict
from tqdm import tqdm

import redis
import torch
from pycocotools.coco import COCO
from torchvision.ops import nms
from colorama import Fore, Back, Style, init
from rich.console import Console
from rich.table import Table

from engine.general import (get_work_dir_path, save_json, load_json, xywh_to_xyxy)
from engine.redis_manager import RedisManager
from engine.builder import Builder
from model_zoo import BaseInstanceModel


def get_args_parser():
    parser = argparse.ArgumentParser('Model evaluation script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    parser.add_argument('--excel', '-e', type=str,
                        help='Existing Excel file.'
                             'If given the file, this script will append new value in the given file.'
                             'Otherwise, this script will create a new Excel file depending on the task type.')

    parser.add_argument('--detected_json', '-d', type=str,
                        help='Existing detected file.'
                             'If given the file, this script will be evaluate directly.')

    parser.add_argument('--dir_name', type=str,
                        help='The name of work dir.')
    return parser




class Evaluator:
    def __init__(self,
                 model: BaseInstanceModel,
                 cfg: dict,
                 detected_json: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.detected_json = detected_json

        self.coco_gt = COCO(os.path.join(cfg["coco_root"], 'annotations', 'instances_val.json'))

        self.redis = RedisManager()
        try:
            self.redis.ping()
        except Exception as e:
            print(f"{Fore.RED}Redis connection failed: {e}{Style.RESET_ALL}")
            self.redis = None

    @classmethod
    def build_by_config(cls, cfg: dict):
        _model = Builder.build_model(cfg)
        return cls(model=_model, cfg=cfg)

    @staticmethod
    def remove_pass_images(coco: COCO):
        """
            Removes image IDs that only contain the 'Pass' class from the COCO dataset.
            Args:
                coco (COCO): COCO dataset object
            Returns:
                all_defect_image_ids (list): List of image IDs that only contain the defect class
                pass_category_id (int): The 'Pass' category ID
                defect_category_ids (list): List of defect category IDs
        """
        # Get all image IDs and category IDs
        all_image_ids = coco.getImgIds()
        all_category_ids = coco.getCatIds()
        all_pass_images_ids = []

        # Get all category names
        category_names = [coco.loadCats(cat_id)[0]['name'] for cat_id in all_category_ids]

        if 'Pass' or 'pass' in category_names:
            pass_category_name = 'Pass' if 'Pass' in category_names else 'pass'

            # Get the category ID for the 'Pass' class
            pass_category_id = coco.getCatIds(catNms=[pass_category_name])[0]

            # Get non-pass category IDs
            defect_category_ids = [cat_id for cat_id in all_category_ids if cat_id != pass_category_id]

            # Get image IDs that only contain the 'Pass' class
            for img_id in coco.getImgIds(catIds=[pass_category_id]):
                annIds = coco.getAnnIds(imgIds=[img_id])
                anns = coco.loadAnns(annIds)
                if all(ann['category_id'] == pass_category_id for ann in anns):
                    all_pass_images_ids.append(img_id)

            # Remove image IDs that only contain the 'Pass' class
            all_defect_image_ids = list(set(all_image_ids) - set(all_pass_images_ids))
            all_defect_image_ids.sort()
        else:
            all_defect_image_ids = all_image_ids
            pass_category_id = None
            defect_category_ids = all_category_ids

        return all_image_ids, all_defect_image_ids, all_pass_images_ids, pass_category_id, defect_category_ids

    @staticmethod
    def get_iou(dt_bbox: Union[list | np.ndarray],
                gt_bbox: Union[list | np.ndarray]) -> np.ndarray:
        """
            Calculates the Intersection over Union (IoU) of two bboxes.

            Args:
                dt_bbox (list | np.ndarray): [D, 4], 格式為[x, y, w, h]
                gt_bbox (list | np.ndarray): [G, 4], 格式為[x, y, w, h]
            Returns:
                Ious np.ndarray(float): [D, G]
        """
        dt_bbox = np.array(dt_bbox, dtype=np.float32)
        gt_bbox = np.array(gt_bbox, dtype=np.float32)

        # xywh -> x1y1x2y2
        dt_bbox = xywh_to_xyxy(dt_bbox)
        gt_bbox = xywh_to_xyxy(gt_bbox)

        lt = np.maximum(dt_bbox[:, None, :2], gt_bbox[:, :2])  # [N,M,2]
        rb = np.minimum(dt_bbox[:, None, 2:], gt_bbox[:, 2:])  # [N,M,2]

        wh = np.maximum(0, rb - lt)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area_1 = (dt_bbox[:, 2] - dt_bbox[:, 0]) * (dt_bbox[:, 3] - dt_bbox[:, 1])
        area_2 = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])

        union = area_1[:, None] + area_2 - inter

        iou = inter / union

        return iou

    def _generate_det(self) -> None:
        """
        Generate detected json file.
        """
        img_id_list = self.coco_gt.getImgIds()
        detected_result = []

        for i, img_id in enumerate(tqdm(img_id_list)):
            if self.redis is not None:
                device_id = int(self.cfg['device'])
                self.redis.set_value(f"GPU:{device_id}_eval_progress", (i + 1) / len(img_id_list) * 100)
                if self.redis.get_value(f"GPU:{device_id}_stop_eval"):
                    sys.exit()

            # Load image
            img_info = self.coco_gt.loadImgs([img_id])[0]
            img_file = os.path.join(self.cfg['coco_root'], 'val', img_info['file_name'])

            # Inference
            result = self.model.predict(img_file,
                                        conf_thres=0.3,
                                        nms_thres=self.cfg['nms_thres'],
                                        return_vis=False)

            class_list = result['class_list']
            score_list = result['score_list']
            bbox_list = result['bbox_list']

            for cls, score, bbox in zip(class_list, score_list, bbox_list):
                detected_result.append({
                    'image_id': img_id,
                    'category_id': cls,
                    'bbox': bbox,
                    'score': round(score, 5),
                })

        # Save detected file
        self.detected_json = os.path.join(get_work_dir_path(self.cfg), 'detected.json')
        save_json(self.detected_json, detected_result, indent=2)

    def _filter_result(self, score_threshold: float) -> List[dict]:
        """
            Filter the result of detection model by score threshold

        Args:
            score_threshold (float): Confidence threshold

        Returns:
            result (list[dict]): [{'image_id': int, 'category_id': int, 'bbox': list, 'score': float}]
                - 'image_id'（int）： Image ID
                - 'category_id'（int）： Category ID
                - 'bbox'（list）： [x, y, w, h]
                - 'score'（float）： Confidence
        """

        detected_results = load_json(self.detected_json)
        result = []

        # Filter out results with low confidence
        filtered_results = [result for result in detected_results if result['score'] >= score_threshold]

        for img_id in self.coco_gt.getImgIds():
            xyxy_bbox_list, xywh_bbox_list, score_list, class_list = [], [], [], []

            for rs in filtered_results:
                if rs['image_id'] == img_id:
                    xyxy_bbox_list.append(xywh_to_xyxy(rs['bbox'])[0])
                    xywh_bbox_list.append(rs['bbox'])
                    score_list.append(rs['score'])
                    class_list.append(rs['category_id'])

            # NMS
            if class_list and xywh_bbox_list and score_list:
                indices = nms(boxes=torch.FloatTensor(np.array(xyxy_bbox_list)),
                              scores=torch.FloatTensor(np.array(score_list)),
                              iou_threshold=self.cfg['nms_thres']).cpu().numpy()

                class_list = np.array(class_list)[indices].tolist()
                xywh_bbox_list = np.array(xywh_bbox_list)[indices].tolist()
                score_list = np.array(score_list)[indices].tolist()

            # Save result
            for cls, score, bbox in zip(class_list, score_list, xywh_bbox_list):
                result.append({
                    'image_id': img_id,
                    'category_id': cls,
                    'bbox': bbox,
                    'score': round(score, 5),
                })

        return result

    def eval(self, return_detail=False) -> list | tuple:
        # Generate detected json
        if self.detected_json is None:
            self._generate_det()

        print('=' * 40)
        print(Fore.BLUE + "Confidence score: {:.1}".format(self.cfg['conf_thres']) + Fore.WHITE)
        print('=' * 40)

        # Get all defect image ids
        all_image_ids, all_defect_images, all_pass_images_ids, pass_category_id, defect_category_ids = \
            self.remove_pass_images(self.coco_gt)

        # Number of defects
        all_category_ids = self.coco_gt.getCatIds()
        all_category_names = [cat['name'] for cat in self.coco_gt.loadCats(all_category_ids)]
        all_defects = sum([len(self.coco_gt.getAnnIds(catIds=[cat_id]))
                           for cat_id in all_category_ids if cat_id != pass_category_id])

        # Filter detection result
        dt_result = self._filter_result(score_threshold=self.cfg["conf_thres"])
        if len(dt_result) == 0:
            print(Fore.RED + 'Can not detect anything! All of the values are zero.' + Fore.WHITE)
            return [0] * 4

        # Load detection result
        coco_dt = self.coco_gt.loadRes(dt_result)

        # Calculate the recall rate and false positive rate
        nf_recall_img = 0
        nf_fpr_img = 0
        nf_recall_ann = 0
        nf_fpr_ann = 0
        fpr_image_name = []
        undetected_image_name = []
        each_detect_score = {defect_category_id: 0 for defect_category_id in defect_category_ids}

        # For each defected image
        for i, img_id in enumerate(all_defect_images):
            gt_ann = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=[img_id]))
            dt_ann = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))

            # Extract predictions that are classified as defect and ground truth is also defect
            defected_gt_bboxes = np.array([gt['bbox'] for gt in gt_ann if gt['category_id'] != pass_category_id])
            defected_dt_bboxes = np.array([dt['bbox'] for dt in dt_ann if dt['category_id'] != pass_category_id])

            # Check if there is any prediction
            if len(defected_dt_bboxes) == 0 and len(defected_gt_bboxes) != 0:
                undetected_image_name.append(self.coco_gt.loadImgs(ids=[img_id])[0]['file_name'])
                continue
            elif len(defected_dt_bboxes) == 0 or len(defected_gt_bboxes) == 0:
                continue

            # Calculate iou
            dt_gt_iou = self.get_iou(defected_dt_bboxes, defected_gt_bboxes)

            # Filter iou
            dt_gt_iou[dt_gt_iou < self.cfg["iou_thres"]] = 0

            # ==========To detected all classes recall and fpr==========
            # Calculate the recall
            nf_recall = np.any(dt_gt_iou, axis=0).sum()
            if nf_recall < len(defected_gt_bboxes):
                undetected_image_name.append(self.coco_gt.loadImgs(ids=[img_id])[0]['file_name'])
            nf_recall = nf_recall if nf_recall < len(defected_gt_bboxes) else len(defected_gt_bboxes)
            nf_recall_ann += nf_recall
            nf_recall_img += 1 if nf_recall > 0 else 0

            # Calculate the fpr
            nf_fpr = np.sum(~np.any(dt_gt_iou != 0, axis=1))
            nf_fpr_ann += nf_fpr
            if nf_fpr > 0:
                nf_fpr_img += 1
                fpr_image_name.append(self.coco_gt.loadImgs(ids=[img_id])[0]['file_name'])

            # ==========To calculate each class recall and average confidence==========
            gt_class_id = np.array([gt['category_id'] for gt in gt_ann if gt['category_id'] != pass_category_id])

            class_id = gt_class_id[np.sum(dt_gt_iou, axis=0) != 0]
            for cls_id in class_id:
                each_detect_score[cls_id] += 1

        # For each pass image
        for i, img_id in enumerate(all_pass_images_ids):
            dt_ann = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[img_id]))
            defected_dt_bboxes = np.array([dt['bbox'] for dt in dt_ann if dt['category_id'] != pass_category_id])

            nf_fpr_img += 1 if len(defected_dt_bboxes) > 0 else 0
            nf_fpr_ann += len(defected_dt_bboxes)
            if len(defected_dt_bboxes) > 0:
                fpr_image_name.append(self.coco_gt.loadImgs(ids=[img_id])[0]['file_name'])

        result = [
            round((nf_recall_img / len(all_image_ids)) * 100, 2),  # 檢出率 (圖片)
            round((nf_fpr_img / len(all_image_ids)) * 100, 2),  # 過殺率 (圖片)
            nf_recall_ann,  # 檢出數 (瑕疵)
            nf_fpr_ann,  # 過殺數 (瑕疵)
        ]

        # ==========Print information==========
        console = Console()

        # Print result recall and fpr
        table = Table(title="Metrics for all classes", title_justify="left")

        for title, value in zip(self.cfg['metrics_for_all'], result):
            table.add_column(title, justify="center", style="cyan", no_wrap=True)

        table.add_row(*[str(value) for value in result])
        console.print(table)

        # Print each class recall and average confidence
        table = Table(title="Metrics for each class", title_justify="left", show_header=True,
                      header_style="bold magenta")
        table.add_column("Category")
        table.add_column("Total", justify='center', style="cyan", no_wrap=True)
        table.add_column("Recall", justify='center', style="cyan", no_wrap=True)
        table.add_column("Recall rate", justify='center', style="cyan", no_wrap=True)

        for cls_id, recall in each_detect_score.items():
            cat_name = all_category_names[cls_id]
            total = len(self.coco_gt.getAnnIds(catIds=[cls_id]))
            recall_rate = f"{recall / total:.3f}" if total > 0 else "-"

            table.add_row(
                cat_name, str(total), str(recall), recall_rate
            )

        console.print(table)

        # Print information
        table = Table(title="Result Analysis", show_lines=True, title_justify="left")
        table.add_column("Information")
        table.add_column("Value")

        table.add_row("Number of defect image", str(len(all_defect_images)))
        table.add_row("Number of defect", str(all_defects))
        table.add_row("FPR images", ", ".join(fpr_image_name))
        table.add_row("Undetected images", ", ".join(undetected_image_name))

        console.print(table)

        if return_detail:
            return result, fpr_image_name, undetected_image_name
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # Create builder
    builder = Builder(config_path=args.config, task='eval', work_dir_name=args.dir_name)

    # Build config
    cfg = builder.build_config()

    # Build model
    model = builder.build_model(cfg)

    # Use multi confidence threshold
    detected_json = args.detected_json
    confidences = np.arange(0.3, 1.0, 0.1)

    results_df = pd.DataFrame(
        columns=['Confidence Threshold', 'Recall (Image)', 'FPR (Image)', 'Recall (Defect)', 'FPR (Defect)'])

    for idx, conf in enumerate(confidences):
        cfg['conf_thres'] = conf

        # Build evaluator
        evaluator = Evaluator(model=model, cfg=cfg, detected_json=detected_json)
        _result = evaluator.eval()

        results_df.loc[idx] = [conf, _result[0], _result[1], _result[2], _result[3]]

        # Save detected result
        if idx == 0:
            detected_json = evaluator.detected_json

    results_df['Recall (Image)'] = results_df['Recall (Image)'].apply(lambda x: f"{x}%")
    results_df['FPR (Image)'] = results_df['FPR (Image)'].apply(lambda x: f"{x}%")
    transposed_results_df = results_df.set_index('Confidence Threshold').T
    transposed_results_df.to_excel(os.path.join(get_work_dir_path(cfg), 'result_analysis.xlsx'))
