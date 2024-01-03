from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from typing import Optional
from pycocotools.coco import COCO
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
                               round(TIMER[2].dt, 3),
                               round(TIMER[3].dt, 3)]

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

            # Analyze result
            if self.cfg['task'] == 'instance_segmentation':
                class_list = result['class_list']
                score_list = result['score_list']
                bbox_list = result['bbox_list']
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

        # Save
        save_json(os.path.join(get_work_dir_path(self.cfg), 'detected.json'), detected_result, indent=2)

    def _get_recall_fpr(self,
                        coco_de: COCO,
                        iouType: str,
                        threshold_iou: float = 0.3) -> dict:
        """
            計算檢出率、過殺率

            Args:
                coco_de (COCO): 由_generate_det()函數所生成的json檔，經過loadRes()所得到的
                iouType (str): 指定評估的型態 ex['bbox', 'segm']
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
        eval = PreviewResults(self.coco_gt, coco_de, iou_tresh=threshold_iou, iouType=iouType, useCats=False)

        cat_ids = self.coco_gt.getCatIds()
        img_ids = self.coco_gt.getImgIds()
        ann_ids = self.coco_gt.getAnnIds()

        # ==========不使用patch==========
        if not self.cfg['use_patch']:
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

                # 針對每一個類別
                for cat_id in cat_ids:
                    each_class_tp_for_image[cat_id] += 1 if tp[cat_id] > 0 else 0
                    each_class_fp_for_image[cat_id] += 1 if fp[cat_id] > 0 else 0
                    each_class_fn_for_image[cat_id] += 1 if fn[cat_id] > 0 else 0

                each_class_tp_for_defect += tp
                each_class_fp_for_defect += fp
                each_class_fn_for_defect += fn

                # 針對全部類別
                sum_tp_for_image += 1 if sum_tp > 0 else 0
                sum_fp_for_image += 1 if sum_fp > 0 else 0

            sum_tp_for_defect = np.sum(each_class_tp_for_defect)
            sum_fp_for_defect = np.sum(each_class_fp_for_defect)

            # ==========依圖片為單位==========
            # 檢出率 (全部類別)
            image_recall_all_classes = round((sum_tp_for_image / len(img_ids)) * 100, 2)

            # 過殺率 (全部類別)
            image_fpr_all_classes = round((sum_fp_for_image / len(img_ids)) * 100, 2)

            # 檢出率 (各個類別)
            image_recall_each_class = [round(v * 100, 2) for v in
                                       np.divide(each_class_tp_for_image,
                                                 each_class_tp_for_image + each_class_fn_for_image,
                                                 out=np.zeros_like(each_class_tp_for_image),
                                                 where=(each_class_tp_for_image + each_class_fn_for_image) != 0)]
            # 過殺率 (各個類別)
            image_fpr_each_class = [round(v * 100, 2) for v in
                                    np.divide(each_class_fp_for_image,
                                              each_class_fp_for_image + each_class_tp_for_image,
                                              out=np.zeros_like(each_class_fp_for_image),
                                              where=(each_class_fp_for_image + each_class_tp_for_image) != 0)]

            # ==========依瑕疵為單位==========
            # 檢出率 (全部類別)
            defect_recall_all_classes = round((sum_tp_for_defect / len(ann_ids)) * 100, 2)  # 檢出率(全部類別)

            # TODO: 會大於1!!!
            # 過殺率 (全部類別)
            defect_fpr_all_classes = round((sum_fp_for_defect / len(ann_ids)) * 100, 2)  # 檢出率(全部類別)

            # 檢出率 (各個類別)
            defect_recall_each_class = [round(v * 100, 2) for v in
                                        np.divide(each_class_tp_for_defect,
                                                  each_class_tp_for_defect + each_class_fn_for_defect,
                                                  out=np.zeros_like(each_class_tp_for_defect),
                                                  where=(each_class_tp_for_defect + each_class_fn_for_defect,) != 0)]
            # 過殺率 (各個類別)
            defect_fpr_each_class = [round(v * 100, 2) for v in
                                     np.divide(each_class_fp_for_defect,
                                               each_class_fp_for_defect + each_class_tp_for_defect,
                                               out=np.zeros_like(each_class_fp_for_defect),
                                               where=(each_class_fp_for_defect + each_class_tp_for_defect) != 0)]

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

        # ==========使用patch==========
        else:
            pass

    def _instance_segmentation_eval(self, predicted_coco: COCO):
        with self.logger:
            # Evaluate
            recall_and_fpr = self._get_recall_fpr(coco_de=predicted_coco, iouType='bbox')

            # Print information
            self.logger.print_message(recall_and_fpr['All'], recall_and_fpr['Each'])

            # Store value
            self.writer.write_col(self.writer.common_metrics +
                                  recall_and_fpr['All'],
                                  sheet_name=self.cfg['sheet_names'][0])

            for cls_name, sheet_name in zip(self.cfg['class_names'], self.cfg['sheet_names'][1:]):
                self.writer.write_col(self.writer.common_metrics +
                                      recall_and_fpr['Each'][cls_name],
                                      sheet_name=sheet_name)

    def eval(self):
        # Generate detected json
        self._generate_det()

        with self.writer:
            # Load json
            predicted_coco = self.coco_gt.loadRes(os.path.join(get_work_dir_path(self.cfg), 'detected.json'))

            if self.cfg['task'] == 'instance_segmentation' or\
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
    builder = Builder(config_path=args.config, task='eval')

    # Build config
    cfg = builder.build_config()

    # Build model
    model = builder.build_model(cfg)

    # Build evaluator
    evaluator = Evaluator(model=model, cfg=cfg, excel_path=args.excel)
    evaluator.eval()
