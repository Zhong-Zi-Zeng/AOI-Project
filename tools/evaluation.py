from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from typing import Optional
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model_zoo.base.BaseModel import BaseModel
from tqdm import tqdm
import pycocotools.mask as ms
import json
import numpy as np
import argparse
import openpyxl
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('Model evaluation script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    return parser

class Writer:
    def __init__(self,
                 excel_path: str,
                 class_names: list[str]):

        self.default_template = {
            'All': {'titles': ["Optimizer",
                               "Image Size",
                               "Inference Time(ms) ↓",
                               "NMS(ms) ↓",
                               "mAP_0.5(Box) ↑",
                               "mAP_0.5:0.95(Box) ↑",
                               "mAR_0.5:0.95(Box) ↑",
                               "mAP_0.5(Mask) ↑",
                               "mAP_0.5:0.95(Mask) ↑",
                               "mAR_0.5:0.95(Mask) ↑"]},
            'Each class(box, mAP)': {'titles': ["Optimizer", "Image Size"] + class_names},
            'Each class(box, mAR)': {'titles': ["Optimizer", "Image Size"] + class_names},
            'Each class(mask, mAP)': {'titles': ["Optimizer", "Image Size"] + class_names},
            'Each class(mask, mAR)': {'titles': ["Optimizer", "Image Size"] + class_names}
        }
        self.excel_path = excel_path
        self._create_or_load_workbook(excel_path)

    def _create_or_load_workbook(self, excel_path: str):
        if not os.path.isfile(excel_path):
            self.wb = openpyxl.Workbook()
            self.wb.remove(self.wb.active)
            self._initial_sheets()
            self.wb.save(excel_path)
        else:
            self.wb = openpyxl.load_workbook(excel_path)

    def _initial_sheets(self):
        """
            針對不同的task初始化不同的excel檔案和sheet name
        """
        # create sheet (for instance segmentation)
        for sheet_name, titles in self.default_template.items():
            self.wb.create_sheet(sheet_name)
            self.write_col(data_list=titles['titles'], sheet_name=sheet_name, row=1)

        self.wb.save(self.excel_path)

    def write_col(self,
                  data_list: list[str | float | int],
                  sheet_name: str,
                  row: Optional[int] = None):
        worksheet = self.wb[sheet_name]

        if row is None:
            row = worksheet.max_row + 1

        for col_idx, value in enumerate(data_list, start=1):
            worksheet.cell(row=row, column=col_idx, value=value)

        self.wb.save(self.excel_path)


class Evaluator:
    def __init__(self, model: BaseModel, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.coco_root = cfg["coco_root"]
        self.nms_thres = cfg["nms_thres"]
        self.conf_thres = cfg["conf_thres"]
        self.coco_gt = COCO(os.path.join(cfg["coco_root"], 'annotations', 'instances_val2017.json'))

        # excel_path = os.path.join(cfg['work_dir'], 'evaluation.xlsx')
        # class_ids = self.coco_gt.getCatIds()
        # class_names = self.coco_gt.loadCats(class_ids)
        # class_names = [cls['name'] for cls in class_names]
        # self.writer = Writer(excel_path=excel_path, class_names=class_names)

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
            height, width = image.shape[:2]

            # Inference
            result = self.model.predict(image, conf_thres=self.conf_thres, nms_thres=self.nms_thres)

            # Analyze result
            class_list = result['class_list']
            score_list = result['score_list']
            bbox_list = result['bbox_list']
            polygon_list = result['polygon_list']

            for cls, score, bbox, polygon in zip(class_list, score_list, bbox_list, polygon_list):
                if len(bbox) == 0 or len(polygon) < 2:
                    continue

                # Convert polygon to RLE format
                mask = Image.new('L', (width, height), 0)
                ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
                rle = ms.encode(np.asfortranarray(np.array(mask)))
                rle['counts'] = str(rle['counts'], encoding='utf-8')

                # Save predicted bbox result
                detected_result.append({
                    'image_id': img_id,
                    'category_id': cls,
                    'bbox': bbox,
                    'score': round(score, 5),
                    'segmentation': rle
                })

        # Save
        with open('./result.json', 'w') as file:
            json.dump(detected_result, file)

    def _coco_eval(self,
                   coco_de: COCO,
                   task: str,
                   class_id: Optional[int] = None) -> np.ndarray:
        """
            依照給定的task使用coco內建的eval api進行評估，並返回各項評估數值

            Args:
                coco_de (COCO): 由_generate_det()函數所生成的json檔，經過loadRes()所得到的
                task (str): 指定評估的型態 ex['bbox', 'segm']
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
        coco_eval = COCOeval(self.coco_gt, coco_de, task)

        if class_id is not None:
            coco_eval.params.catIds = class_id

        coco_eval.evaluate()
        coco_eval.accumulate()
        stats = coco_eval.summarize(show=False)
        return stats

    def eval(self):
        # Generate detected json
        self._generate_det()

        # Load json
        predicted_json = self.coco_gt.loadRes('./result.json')

        # Evaluate all classes
        all_boxes_result = self._coco_eval(predicted_json, task='bbox')
        all_masks_result = self._coco_eval(predicted_json, task='segm')
        print("For all classes:")
        print("%15s" * 6 % ("(Box mAP_0.5", " mAP_0.5:0.95", " mAR_0.5:0.95)",
                            " (Mask mAP_0.5", " mAP_0.5:0.95", " mAR_0.5:0.95)"))
        print("%15.3f" * 6 % (all_boxes_result[1], all_boxes_result[0], all_boxes_result[8],
                              all_masks_result[1], all_masks_result[0], all_masks_result[8]))

        # TODO: Record value

        # Evaluate per class
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        class_name = [cat['name'] for cat in cats]

        print("For each class:")
        print(("%25s" + " %15s" * 4) % ("Class", "(Box mAP_0.5", " mAR_0.5:0.95)",
                                        " (Mask mAP_0.5", " mAR_0.5:0.95)"))
        for cls_id, name in enumerate(class_name):
            box_result = self._coco_eval(predicted_json, task='bbox', class_id=cls_id)
            mask_result = self._coco_eval(predicted_json, task='segm', class_id=cls_id)
            print(("%25s" + " %15.3f" * 4) % (name, box_result[1], box_result[8],
                                              mask_result[1], mask_result[8]))

            # TODO: Record value

        # Print process time
        for key, value in self.model.timer().items():
            print(f"{key:15s} {value:4.3f}", end=' | ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # Create builder
    builder = Builder(config_path=r"D:\Heng_shared\AOI-Project\configs\yolov7_seg.yaml")

    # Build config
    cfg = builder.build_config()

    # Create work dir
    root = os.getcwd()
    os.makedirs(os.path.join(root, 'work_dirs', cfg['name']), exist_ok=True)

    # Build model
    model = builder.build_model(cfg)

    # Build evaluator
    evaluator = Evaluator(model=model, cfg=cfg)
    evaluator.eval()
