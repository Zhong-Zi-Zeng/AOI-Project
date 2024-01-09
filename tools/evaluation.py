from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from typing import Optional
from pycocotools.coco import COCO
from colorama import Fore, Back, Style, init
from model_zoo import BaseInstanceModel
from faster_coco_eval.extra import PreviewResults
from engine.general import (get_work_dir_path, save_json)
from engine.timer import TIMER
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import openpyxl
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('Model evaluation script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    parser.add_argument('--excel', '-e', type=str,
                        help='Existing Excel file.'
                             'If given the file, this script will append new value in the given file.'
                             'Otherwise, this script will create a new Excel file depending on the task type.')

    parser.add_argument('--dir_name', type=str,
                        help='The name of work dir.')

    return parser


class Logger:
    def __init__(self,
                 cfg: dict):
        self.cfg = cfg
        self._start = False

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Print process time
        print('\n\n')
        for timer in TIMER:
            print(f"{timer.name:15s} {timer.dt:.3f}", end=' | ')
        print('\n\n')

        self._start = False

    def print_message(self,
                      value_for_all: list,
                      value_for_each: dict):
        # 對於所有類別
        data = {title: [value]
                for title, value in zip(self.cfg['metrics_for_all'], value_for_all)}

        df = pd.DataFrame(data)
        print('For all classes:')
        print(df.to_string(index=False))
        print('\n' * 2)

        # 對於每一個類別
        data = {"Metrics": self.cfg['metrics_for_each']}
        data.update(value_for_each)

        df = pd.DataFrame(data)
        print('For each class:')
        print(df.to_string(index=False))


class Writer:
    def __init__(self, cfg: dict, excel_path: Optional[str] = None):
        self.cfg = cfg
        self.excel_path = excel_path

        if excel_path is None:
            self.wb = openpyxl.Workbook()
            self.wb.remove(self.wb['Sheet'])
            self.excel_path = os.path.join(get_work_dir_path(cfg), 'result.xlsx')
            self._create_sheet()
            self._append_title()
        else:
            self.wb = openpyxl.load_workbook(self.excel_path)
            self.excel_path = os.path.join(get_work_dir_path(cfg), 'result.xlsx')
            self._append_title()

    def __enter__(self):
        # For common metrics
        self.common_metrics = [self.cfg['model_name'],
                               self.cfg['optimizer'],
                               self.cfg['imgsz'][0],
                               self.cfg['use_patch'],
                               self.cfg['number_of_class'],
                               round(TIMER[2].dt, 3),
                               round(TIMER[3].dt, 3),
                               self.cfg['nms_thres'],
                               self.cfg['conf_thres'],
                               self.cfg['iou_thres'],]

    def __exit__(self, exc_type, exc_val, exc_tb):
        """自動儲存"""

        self.wb.save(self.excel_path)
        self._save_to_json()

        print('Excel file has been saved to the {}'.format(self.excel_path))

    def _check_has_title(self, sheet_name: str):
        """檢查excel檔是否已經有標題了"""
        work_sheet = self.wb[sheet_name]
        if work_sheet.max_row == 1:
            return False
        return True

    def _save_to_json(self):
        """將excel檔轉換成json後儲存"""

        # TODO: 目前json只有存全部類別的指標，沒有各別的
        path = os.path.join(get_work_dir_path(self.cfg), 'result.json')
        excel = pd.read_excel(self.excel_path)
        data_dict = excel.to_dict()
        save_json(path, data_dict, indent=2)

        print('Json file has been saved to the {}'.format(path))

    def _append_title(self):
        """從config中的common、metrics、class_names添加到對應的sheet"""
        for idx, sheet_name in enumerate(self.cfg['sheet_names']):
            if self._check_has_title(sheet_name):
                continue

            if idx == 0:
                self.write_col(self.cfg['common'] + self.cfg['metrics_for_all'], sheet_name, row=1)
            else:
                self.write_col(self.cfg['common'] + self.cfg['class_names'], sheet_name, row=1)

    def _create_sheet(self):
        """依照config中的sheet_names去創建新的work sheet"""
        for idx, sheet_name in enumerate(self.cfg['sheet_names']):
            self.wb.create_sheet(sheet_name)

    def write_col(self,
                  data_list: list[str | float | int],
                  sheet_name: str,
                  row: Optional[int] = None):
        """將資料寫入到excel中"""
        worksheet = self.wb[sheet_name]

        if row is None:
            real_max_row = worksheet.max_row
            while real_max_row > 0:
                if all(cell.value is None for cell in worksheet[real_max_row]):
                    real_max_row -= 1
                else:
                    break
            row = real_max_row + 1

        for col_idx, value in enumerate(data_list, start=1):
            worksheet.cell(row=row, column=col_idx, value=value)


class Evaluator:
    def __init__(self,
                 model: BaseInstanceModel,
                 cfg: dict,
                 excel_path: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.coco_root = cfg["coco_root"]
        self.nms_thres = cfg["nms_thres"]
        self.conf_thres = cfg["conf_thres"]
        self.coco_gt = COCO(os.path.join(cfg["coco_root"], 'annotations', 'instances_val2017.json'))
        self.writer = Writer(cfg, excel_path=excel_path)
        self.logger = Logger(cfg)

    def _generate_det(self):
        """
            對給定的model輸入測試圖片，並將結果轉換成coco eval格式的json檔
        """
        img_id_list = self.coco_gt.getImgIds()
        detected_result = []

        for img_id in tqdm(img_id_list):
            # Load image
            img_info = self.coco_gt.loadImgs([img_id])[0]
            img_file = img_info['file_name']
            image = cv2.imread(os.path.join(self.coco_root, 'val2017', img_file))

            # Inference
            result = self.model.predict(image, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

            class_list = result['class_list']
            score_list = result['score_list']
            bbox_list = result['bbox_list']

            # When not detect anything, append null list into detected_result
            if not (class_list or score_list or bbox_list):
                detected_result.append({
                    'image_id': img_id,
                    'category_id': None,
                    'bbox': [],
                    'score': None,
                    'segmentation': []
                })
                continue

            # Analyze result
            if self.cfg['task'] == 'instance_segmentation':
                rle_list = result['rle_list']
                for cls, score, bbox, rle in zip(class_list, score_list, bbox_list, rle_list):
                    detected_result.append({
                        'image_id': img_id,
                        'category_id': cls,
                        'bbox': bbox,
                        'score': round(score, 5),
                        'segmentation': rle
                    })

            elif self.cfg['task'] == 'object_detection':
                for cls, score, bbox in zip(class_list, score_list, bbox_list):
                    detected_result.append({
                        'image_id': img_id,
                        'category_id': cls,
                        'bbox': bbox,
                        'score': round(score, 5),
                    })
        # Save
        save_json(os.path.join(get_work_dir_path(self.cfg), 'detected.json'), detected_result, indent=2)

    @staticmethod
    def calculate_metrics(eval: PreviewResults,
                          img_ids: list,
                          cat_ids: list):
        """
        Args:
            eval (PreviewResults): 以計算完畢的faster-coco-eval
            img_ids (list):　存放dataset中所有image的id
            cat_ids (list):　存放dataset中所有class的id

        Returns:
             sum_tp_for_image (float): 所有類別, 以圖片為單位的TP
             sum_fp_for_image (float): 所有類別, 以圖片為單位的TP
             each_class_tp_for_image list[float]: 各個類別, 以圖片為單位的TP
             each_class_fp_for_image list[float]: 各個類別, 以圖片為單位的FP
             each_class_fn_for_image list[float]: 各個類別, 以圖片為單位的FN
             each_class_tp_for_defect list[float]: 各個類別, 以瑕疵為單位的TP
             each_class_fp_for_defect list[float]: 各個類別, 以瑕疵為單位的FP
             each_class_fn_for_defect list[float]: 各個類別, 以瑕疵為單位的FN
        """

        sum_tp_for_image = 0
        sum_fp_for_image = 0

        each_class_tp_for_image = np.zeros(len(cat_ids), dtype=np.float32)
        each_class_fp_for_image = np.zeros(len(cat_ids), dtype=np.float32)
        each_class_fn_for_image = np.zeros(len(cat_ids), dtype=np.float32)

        each_class_tp_for_defect = np.zeros(len(cat_ids), dtype=np.float32)
        each_class_fp_for_defect = np.zeros(len(cat_ids), dtype=np.float32)
        each_class_fn_for_defect = np.zeros(len(cat_ids), dtype=np.float32)

        for img_id in img_ids:
            tp, fp, fn = eval.get_tp_fp_fn(image_id=img_id)

            sum_tp = np.sum(tp)
            sum_fp = np.sum(fp)

            each_class_tp_for_image += (tp > 0).astype(np.float32)
            each_class_fp_for_image += (fp > 0).astype(np.float32)
            each_class_fn_for_image += (fn > 0).astype(np.float32)

            each_class_tp_for_defect += tp
            each_class_fp_for_defect += fp
            each_class_fn_for_defect += fn

            sum_tp_for_image += (sum_tp > 0).astype(np.float32)
            sum_fp_for_image += (sum_fp > 0).astype(np.float32)

        return sum_tp_for_image, sum_fp_for_image, each_class_tp_for_image, each_class_fp_for_image, each_class_fn_for_image, each_class_tp_for_defect, each_class_fp_for_defect, each_class_fn_for_defect

    @staticmethod
    def calculate_metrics_percentage(denominator,
                                     sum_tp,
                                     sum_fp,
                                     each_class_tp,
                                     each_class_fp,
                                     each_class_fn):
        # 檢出率 (全部類別)
        recall_all_classes = round((sum_tp / denominator) * 100, 2)

        # 過殺率 (全部類別)
        fpr_all_classes = round((sum_fp / denominator) * 100, 2)

        # 檢出率 (各個類別)
        recall_each_class = [round(v * 100, 2) for v in np.divide(each_class_tp, each_class_tp + each_class_fn,
                                                                  out=np.zeros_like(each_class_tp),
                                                                  where=(each_class_tp + each_class_fn) != 0)]

        # 過殺率 (各個類別)
        fpr_each_class = [round(v * 100, 2) for v in np.divide(each_class_fp, each_class_tp + each_class_fp,
                                                               out=np.zeros_like(each_class_fp),
                                                               where=(each_class_tp + each_class_fp) != 0)]

        return recall_all_classes, fpr_all_classes, recall_each_class, fpr_each_class

    def _get_recall_fpr(self,
                        coco_de: COCO,
                        iou_type: str,
                        threshold_iou: float = 0.3) -> dict:
        """
            計算檢出率、過殺率

            Args:
                coco_de (COCO): 由_generate_det()函數所生成的json檔，經過loadRes()所得到的
                iou_type (str): 指定評估的型態 ex['bbox', 'segm']
                threshold_iou (float): IoU 閥值

            Returns:
                Image 為以圖片為單位 Defect 為以瑕疵為單位

                {
                    "All": [Recall(image)、FPR(image)、Recall(defect)、FPR(defect)]
                    "Each": {
                                "Class 1": [Recall(image)、FPR(image)、Recall(defect)、FPR(defect)]
                                "Class 2": [Recall(image)、FPR(image)、Recall(defect)、FPR(defect)]
                                ...
                            }
                }
        """
        eval = PreviewResults(self.coco_gt, coco_de, iou_tresh=threshold_iou, iouType=iou_type, useCats=False)
        cat_ids = self.coco_gt.getCatIds()
        img_ids = self.coco_gt.getImgIds()
        ann_ids = self.coco_gt.getAnnIds()

        # ==========不使用patch==========
        if not self.cfg['use_patch']:
            # 計算tp、fp、fn
            sum_tp_for_image, \
                sum_fp_for_image, \
                each_class_tp_for_image, \
                each_class_fp_for_image, \
                each_class_fn_for_image, \
                each_class_tp_for_defect, \
                each_class_fp_for_defect, \
                each_class_fn_for_defect = self.calculate_metrics(eval, img_ids, cat_ids)

        # ==========使用patch==========
        else:
            # 找尋原始圖片與patch對應的id
            mother_to_sub_images = dict()
            for img_id in img_ids:
                image_name = self.coco_gt.loadImgs(img_id)[0]['file_name']
                parts = image_name.split('_')
                mother_image_id = parts[1]

                if mother_image_id in mother_to_sub_images:
                    mother_to_sub_images[mother_image_id].append(img_id)
                else:
                    mother_to_sub_images[mother_image_id] = [img_id]

            img_ids = mother_to_sub_images

            # 計算tp、fp、fn
            sum_tp_for_image = 0
            sum_fp_for_image = 0
            each_class_tp_for_image = np.zeros(len(cat_ids), dtype=np.float32)
            each_class_fp_for_image = np.zeros(len(cat_ids), dtype=np.float32)
            each_class_fn_for_image = np.zeros(len(cat_ids), dtype=np.float32)
            each_class_tp_for_defect = np.zeros(len(cat_ids), dtype=np.float32)
            each_class_fp_for_defect = np.zeros(len(cat_ids), dtype=np.float32)
            each_class_fn_for_defect = np.zeros(len(cat_ids), dtype=np.float32)

            for patch_ids in mother_to_sub_images.values():
                sum_tp_for_image_, \
                    sum_fp_for_image_, \
                    each_class_tp_for_image_, \
                    each_class_fp_for_image_, \
                    each_class_fn_for_image_, \
                    each_class_tp_for_defect_, \
                    each_class_fp_for_defect_, \
                    each_class_fn_for_defect_ = self.calculate_metrics(eval, patch_ids, cat_ids)

                sum_tp_for_image += (sum_tp_for_image_ > 0).astype(np.float32)
                sum_fp_for_image += (sum_fp_for_image_ > 0).astype(np.float32)
                each_class_tp_for_image += (each_class_tp_for_image_ > 0).astype(np.float32)
                each_class_fp_for_image += (each_class_fp_for_image_ > 0).astype(np.float32)
                each_class_fn_for_image += (each_class_fn_for_image_ > 0).astype(np.float32)
                each_class_tp_for_defect += each_class_tp_for_defect_
                each_class_fp_for_defect += each_class_fp_for_defect_
                each_class_fn_for_defect += each_class_fn_for_defect_

        # 以圖片為單位的結果
        image_recall_all_classes, \
            image_fpr_all_classes, \
            image_recall_each_class, \
            image_fpr_each_class = self.calculate_metrics_percentage(len(img_ids),
                                                                     sum_tp_for_image,
                                                                     sum_fp_for_image,
                                                                     each_class_tp_for_image,
                                                                     each_class_fp_for_image,
                                                                     each_class_fn_for_image)
        # 以瑕疵為單位的結果
        defect_recall_all_classes, \
            defect_fpr_all_classes, \
            defect_recall_each_class, \
            defect_fpr_each_class = self.calculate_metrics_percentage(len(ann_ids),
                                                                      np.sum(each_class_tp_for_defect),
                                                                      np.sum(each_class_fp_for_defect),
                                                                      each_class_tp_for_defect,
                                                                      each_class_fp_for_defect,
                                                                      each_class_fn_for_defect)
        return {
            "All": [image_recall_all_classes,
                    image_fpr_all_classes,
                    defect_recall_all_classes,
                    defect_fpr_all_classes],
            "Each": {cls_name:
                         [image_recall_each_class[idx],
                          image_fpr_each_class[idx],
                          defect_recall_each_class[idx],
                          defect_fpr_each_class[idx]]
                     for idx, cls_name in
                     zip(range(len(self.cfg['metrics_for_each'])), self.cfg['class_names'])}
        }

    def _instance_segmentation_eval(self, predicted_coco: COCO):
        with self.logger:
            # Evaluate
            recall_and_fpr = self._get_recall_fpr(coco_de=predicted_coco, iou_type='bbox',
                                                  threshold_iou=self.cfg['iou_thres'])

            # Print information
            self.logger.print_message(recall_and_fpr['All'], recall_and_fpr['Each'])

            # Store value
            self.writer.write_col(self.writer.common_metrics +
                                  recall_and_fpr['All'],
                                  sheet_name=self.cfg['sheet_names'][0])

            each_value = [value for value in recall_and_fpr['Each'].values()]
            each_value = np.array(each_value).T

            for idx, sheet_name in enumerate(self.cfg['sheet_names'][1:]):
                self.writer.write_col(self.writer.common_metrics +
                                      each_value[idx].tolist(),
                                      sheet_name=sheet_name)

        if (np.array(recall_and_fpr['All']) == 0).all():
            print(Fore.RED + 'Can not detect anything! All of the values are zero.' + Fore.WHITE)

    def eval(self):
        # Generate detected json
        self._generate_det()

        with self.writer:
            # Load json
            predicted_coco = self.coco_gt.loadRes(os.path.join(get_work_dir_path(self.cfg), 'detected.json'))

            if self.cfg['task'] == 'instance_segmentation' or \
                    self.cfg['task'] == 'object_detection':
                self._instance_segmentation_eval(predicted_coco)
            elif self.cfg['task'] == 'semantic_segmentation':
                # TODO: segmentation evaluation
                pass


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

    # Build evaluator
    evaluator = Evaluator(model=model, cfg=cfg, excel_path=args.excel)
    evaluator.eval()
