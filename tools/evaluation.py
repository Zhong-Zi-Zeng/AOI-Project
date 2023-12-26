from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from typing import Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
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

    def print_for_all(self,
                      all_boxes_result: Optional[list] = None,
                      all_masks_result: Optional[list] = None):
        print("For all classes:")

        if all_boxes_result is not None and \
                all_masks_result is not None:
            length = len(all_boxes_result) + len(all_masks_result)
            print(("%20s" * length) % (tuple(self.cfg['metrics_for_all'])))
            print(("%20.3f" * length) % (*all_boxes_result, *all_masks_result))

        elif all_boxes_result is not None:
            length = len(all_boxes_result)
            print(("%20s" * length) % (tuple(self.cfg['metrics_for_all'])))
            print(("%20.3f" * length) % (tuple(all_boxes_result)))

        elif all_masks_result is not None:
            length = len(all_masks_result)
            print(("%20s" * length) % (tuple(self.cfg['metrics_for_all'])))
            print(("%20.3f" * length) % (tuple(all_masks_result)))

    def print_for_each(self,
                       cls_name: str,
                       box_result: list = [],
                       mask_result: list = []):
        length = len(box_result) + len(mask_result)

        if not self._start:
            print("\n\nFor each class:")
            print(("%25s" + " %20s" * length) % ("Class", *self.cfg['metrics_for_each']))
            self._start = True

        if box_result is not None and \
                mask_result is not None:
            print(("%25s" + " %20.3f" * length) % (cls_name, *box_result, *mask_result))
        elif box_result is not None:
            print(("%25s" + " %20.3f" * length) % (cls_name, *box_result))
        elif mask_result is not None:
            print(("%25s" + " %20.3f" * length) % (cls_name, *mask_result))


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

    def _get_precision_recall(self,
                              coco_de: COCO,
                              iouType: str,
                              threshold_iou: float = 0.5,
                              mode: str = "micro") -> dict:
        """
            產生confusion matrix後計算整體和各個類別的Precision和Recall


            Args:
                coco_de (COCO): 由_generate_det()函數所生成的json檔，經過loadRes()所得到的
                iouType (str): 指定評估的型態 ex['bbox', 'segm']
                threshold_iou (float): IoU 閥值
                mode (str): ["macro", "micro"] 使用宏平均還是微平均，預設使用微平均

            Returns:

                precision_recall (dict[list]): 整體和個別的precision、recall、tp、fp、fn
                    {
                        "All" : [total_precision, total_recall, total_tp, total_fp, total_fn],

                        "Class_1": [precision, recall, tp, fp, fn],
                        "Class_2": [precision, recall, tp, fp, fn],
                        "Class_3": [precision, recall, tp, fp, fn]
                        ...
                    }
        """
        results = PreviewResults(
            self.coco_gt, coco_de, iou_tresh=threshold_iou, iouType=iouType, useCats=False
        )

        confusion_matrix_with_fp_fn = results.compute_confusion_matrix()  # (Number of Class, Number of Class + 2)
        confusion_matrix = confusion_matrix_with_fp_fn[..., :-2]  # (Number of Class, Number of Class)

        tp = np.diag(confusion_matrix)  # ((Number of Class, )
        fp = confusion_matrix_with_fp_fn[:, -2]  # ((Number of Class, )
        fn = confusion_matrix_with_fp_fn[:, -1]  # ((Number of Class, )

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)  # ((Number of Class, )
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)  # ((Number of Class, )

        if mode == "micro":
            total_precision = np.sum(tp) / np.sum(tp + fp)
            total_recall = np.sum(tp) / np.sum(tp + fn)
        elif mode == "macro":
            total_precision = np.mean(precision)
            total_recall = np.mean(recall)
        else:
            ValueError("Mode argument error. You only can choose micro or macro.")

        precision_recall = {
            "All": [np.round(total_precision, 3),
                    np.round(total_recall, 3),
                    np.round(np.sum(tp), 3),
                    np.round(np.sum(fp), 3),
                    np.round(np.sum(fn), 3)],
        }

        for idx, cls_name in enumerate(self.cfg['class_names']):
            precision_recall.update({cls_name:
                                         [np.round(precision[idx], 3),
                                          np.round(recall[idx], 3),
                                          np.round(tp[idx], 3),
                                          np.round(fp[idx], 3),
                                          np.round(fn[idx], 3)]}
                                    )

        return precision_recall

    def _coco_eval(self,
                   coco_de: COCO,
                   iouType: str,
                   class_id: Optional[int] = None) -> list:
        """
            依照給定的task使用coco內建的eval api進行評估，並返回各項評估數值

            Args:
                coco_de (COCO): 由_generate_det()函數所生成的json檔，經過loadRes()所得到的
                iouType (str): 指定評估的型態 ex['bbox', 'segm']
                class_id (int): 如果有給定的話，則是返回給定類別的評估結果，用於想得知每一個類別的評估數據
            Return:
                stats[0] : AP at IoU=.50:.05:.95
                stats[1] : AP at IoU=.50
                stats[2] : AP at IoU=.75
                stats[3] : AP for small objects: area < 32^2
                stats[4] : AP for medium objects: 32^2 < area < 96^2
                stats[5] : AP for large objects: area > 96^2
                stats[6] : AR given 1 detection per image
                stats[7] : AR given 10 detections per image
                stats[8] : AR given 100 detections per image
                stats[9] : AR for small objects: area < 32^2
                stats[10] : AR for medium objects: 32^2 < area < 96^2
                stats[11] : AR for large objects: area > 96^2
        """
        coco_eval = COCOeval(self.coco_gt, coco_de, iouType)

        if class_id is not None:
            coco_eval.params.catIds = class_id

        coco_eval.evaluate()
        coco_eval.accumulate()
        stats = coco_eval.summarize(show=False)
        stats = [round(val, 3) for val in stats]

        # 依照不同的task取出需要的metrics
        if self.cfg['task'] == 'instance_segmentation' or self.cfg['task'] == 'object_detection':
            return [stats[1], stats[2], stats[0]]
        elif self.cfg['task'] == 'semantic_segmentation':
            # TODO: semantic_segmentation evaluation value
            pass
        else:
            raise ValueError('Can not find the task type of {}'.format(self.cfg['task']))

        return stats

    def _instance_segmentation_eval(self, predicted_coco: COCO):
        with self.logger:
            # =========Evaluate all classes=========
            all_boxes_result = self._coco_eval(predicted_coco, iouType='bbox')
            bbox_precision_recall = self._get_precision_recall(coco_de=predicted_coco, iouType='bbox')
            all_masks_result = self._coco_eval(predicted_coco, iouType='segm')
            mask_precision_recall = self._get_precision_recall(coco_de=predicted_coco, iouType='segm')

            # Print information
            self.logger.print_for_all(bbox_precision_recall["All"][:2] + all_boxes_result,
                                      mask_precision_recall["All"][:2] + all_masks_result)

            # Store value
            self.writer.write_col(self.writer.common_metrics +
                                  bbox_precision_recall["All"][:2] + all_boxes_result +
                                  mask_precision_recall["All"][:2] + all_masks_result,
                                  sheet_name=self.cfg['sheet_names'][0])

            # =========Evaluate each class=========
            _value = {cls_name: [] for cls_name in self.cfg["class_names"]}

            for cls_id, cls_name in enumerate(self.cfg["class_names"]):
                # box_result = self._coco_eval(predicted_coco, iouType='bbox', class_id=cls_id)
                # mask_result = self._coco_eval(predicted_coco, iouType='segm', class_id=cls_id)
                box_result = bbox_precision_recall[cls_name][:2]
                mask_result = mask_precision_recall[cls_name][:2]
                _value[cls_name] = box_result + mask_result

                # Print information
                self.logger.print_for_each(cls_name, box_result, mask_result)

            # Store value
            for idx, sheet_name in enumerate(self.cfg['sheet_names'][1:]):
                self.writer.write_col(self.writer.common_metrics + [val[idx] for val in _value.values()],
                                      sheet_name=sheet_name)

    def _object_detection_eval(self, predicted_coco: COCO):
        with self.logger:
            # =========Evaluate all classes=========
            all_boxes_result = self._coco_eval(predicted_coco, iouType='bbox')
            bbox_precision_recall = self._get_precision_recall(coco_de=predicted_coco, iouType='bbox')

            # Print information
            self.logger.print_for_all(bbox_precision_recall["All"][:2] + all_boxes_result)

            # Store value
            self.writer.write_col(self.writer.common_metrics +
                                  bbox_precision_recall["All"][:2] + all_boxes_result,
                                  sheet_name=self.cfg['sheet_names'][0])

            # =========Evaluate per class=========
            _value = {cls_name: [] for cls_name in self.cfg["class_names"]}

            for cls_id, cls_name in enumerate(self.cfg["class_names"]):
                # box_result = self._coco_eval(predicted_coco, iouType='bbox', class_id=cls_id)
                box_result = bbox_precision_recall[cls_name][:2]
                _value[cls_name] = box_result
                self.logger.print_for_each(cls_name, box_result)

            # Store value
            for idx, sheet_name in enumerate(self.cfg['sheet_names'][1:]):
                self.writer.write_col(self.writer.common_metrics + [val[idx] for val in _value.values()],
                                      sheet_name=sheet_name)

    def eval(self):
        # Generate detected json
        self._generate_det()

        with self.writer:
            # Load json
            predicted_coco = self.coco_gt.loadRes(os.path.join(get_work_dir_path(self.cfg), 'detected.json'))

            if self.cfg['task'] == 'instance_segmentation':
                self._instance_segmentation_eval(predicted_coco)
            elif self.cfg['task'] == 'object_detection':
                self._object_detection_eval(predicted_coco)
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
