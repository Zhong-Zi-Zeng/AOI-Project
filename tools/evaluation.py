from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))

from moodle_zoo import Yolov7
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from moodle_zoo.base.baseInference import baseInference
import json
from tqdm import tqdm
import numpy as np
import argparse
import cv2


class Evaluator:
    def __init__(self,
                 model: baseInference,
                 coco_root: str,
                 output_path: str = './coco_dt.json'):

        self.model = model
        self.coco_root = coco_root
        self.output_path = output_path
        self.coco_gt = COCO(os.path.join(coco_root, 'annotations', 'instances_val2017.json'))

    def _generate_det(self, task="bbox"):
        img_id_list = self.coco_gt.getImgIds()
        detected_result = []

        for img_id in tqdm(img_id_list):
            img_info = self.coco_gt.loadImgs([img_id])[0]
            img_file = img_info['file_name']
            image = cv2.imread(os.path.join(self.coco_root, 'val2017', img_file))

            result = yolov7.run(image, conf_thres=0.001, nms_thres=0.5)

            class_list = result['class_list']
            score_list = result['score_list']
            bbox_list = result['bbox_list']
            polygon_list = result['polygon_list']

            for cls, score, bbox, polygon in zip(class_list, score_list, bbox_list, polygon_list):
                if task == "bbox":
                    detected_result.append({
                        'image_id': img_id,
                        'category_id': cls,
                        'bbox': bbox,
                        'score': round(score, 3)
                    })

        with open(self.output_path, 'w') as file:
            json.dump(detected_result, file)

    def _coco_eval(self, coco_de, task: str, class_id=None):
        coco_eval = COCOeval(self.coco_gt, coco_de, task)

        if class_id is not None:
            coco_eval.params.catIds = class_id

        coco_eval.evaluate()
        coco_eval.accumulate()
        return coco_eval.summarize(show=False)

    def eval(self):
        # Generate detected json
        # self._generate_det(task)

        # Load json
        coco_de = self.coco_gt.loadRes(self.output_path)

        # All classes
        all_boxes_result = self._coco_eval(coco_de, task='bbox')
        all_masks_result = self._coco_eval(coco_de, task='segm')
        print("For all classes:")
        print("%15s" * 6 % ("(Box mAP_0.5", " mAP_0.5:0.95", " mAR_0.5:0.95)",
                            " (Mask mAP_0.5", " mAP_0.5:0.95", " mAR_0.5:0.95)"))
        print("%15.3f" * 6 % (all_boxes_result[1], all_boxes_result[0], all_boxes_result[8],
                              all_masks_result[1], all_masks_result[0], all_masks_result[8]))

        # Per class
        cats = self.coco_gt.loadCats(self.coco_gt.getCatIds())
        class_name = [cat['name'] for cat in cats]

        print("For each class:")
        print(("%30s" + " %15s" * 6) % ("Class", "(Box mAP_0.5", " mAP_0.5:0.95", " mAR_0.5:0.95)",
                                     " (Mask mAP_0.5", " mAP_0.5:0.95", " mAR_0.5:0.95)"))
        for cls_id, name in enumerate(class_name):
            box_result = self._coco_eval(coco_de, task='bbox', class_id=cls_id)
            mask_result = self._coco_eval(coco_de, task='segm', class_id=cls_id)
            print(("%30s" + " %15.3f" * 6) % (name, box_result[1], box_result[0], box_result[8],
                                            mask_result[1], mask_result[0], mask_result[8]))


if __name__ == "__main__":
    yolov7 = Yolov7(
        weights=r"\\DESKTOP-PPOB8AK\share\AOI_result\Instance Segmentation\yolov7\1024_SGD_202312061402\weights\best.pt",
        data=r"D:\Heng_shared\yolov7-segmentation\data\custom.yaml",
        imgsz=(1024, 1024)
    )

    Evaluator(model=yolov7,
              coco_root=r"C:\Users\鐘子恒\Desktop\Side-Project\AOI-Project\tools\coco").eval()

# Load data
# coco_root =
# coco_gt =
#
# coco_de = coco_gt.loadRes('./coco_dt.json')
# cocoEval = COCOeval(coco_gt, coco_de, 'bbox')
# for id in range(16):
#     cocoEval.params.catIds = id
# cocoEval.evaluate()
# cocoEval.accumulate()
# s = cocoEval.summarize()

# Evaluation
