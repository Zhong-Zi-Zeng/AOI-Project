from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from engine.general import (get_work_dir_path, load_yaml)
import numpy as np
import argparse
import torch
import subprocess

torch.manual_seed(10)
np.random.seed(10)

model_base_config = {
    "Yolov7inSeg": "./base/model/YOLO-v7/yolov7_inSeg_base.yaml",
    "Yolov7Obj": "./base/model/YOLO-v7/yolov7_obj_base.yaml",
    "CascadeMaskRCNN": "./base/model/Cascade-Mask RCNN/r50.yaml",
    "Mask2Former": "./base/model/Mask2Former/r50.yaml",
}

model_config_template = {
    "Yolov7inSeg": "./configs/template/yolov7_inSeg_custom.yaml",
    "Yolov7Obj": "./configs/template/yolov7_obj_custom.yaml",
    "CascadeMaskRCNN": "./configs/template/cascade_mask_rcnn_custom.yaml",
    "Mask2Former": "./configs/template/mask2former_custom.yaml"
}

experiment_order = {
    "Yolov7inSeg": [
        # 0 ~ 7
        {"coco_root": "./data/white_controller/coco/original_class_4",
         "train_dir": "./data/white_controller/yoloSeg/original_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/original_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/original_class_1",
         "train_dir": "./data/white_controller/yoloSeg/original_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/original_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_4",
         "train_dir": "./data/white_controller/yoloSeg/patch_1024_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_1024_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_1",
         "train_dir": "./data/white_controller/yoloSeg/patch_1024_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_1024_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_4",
         "train_dir": "./data/white_controller/yoloSeg/patch_512_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_512_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_1",
         "train_dir": "./data/white_controller/yoloSeg/patch_512_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_512_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_4",
         "train_dir": "./data/white_controller/yoloSeg/patch_256_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_256_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_1",
         "train_dir": "./data/white_controller/yoloSeg/patch_256_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_256_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

        # 8 ~ 15
        {"coco_root": "./data/white_controller/coco/original_class_4",
         "train_dir": "./data/white_controller/yoloSeg/original_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/original_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/original_class_1",
         "train_dir": "./data/white_controller/yoloSeg/original_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/original_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_4",
         "train_dir": "./data/white_controller/yoloSeg/patch_1024_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_1024_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_1",
         "train_dir": "./data/white_controller/yoloSeg/patch_1024_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_1024_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_4",
         "train_dir": "./data/white_controller/yoloSeg/patch_512_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_512_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_1",
         "train_dir": "./data/white_controller/yoloSeg/patch_512_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_512_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_4",
         "train_dir": "./data/white_controller/yoloSeg/patch_256_class_4/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_256_class_4/test",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_1",
         "train_dir": "./data/white_controller/yoloSeg/patch_256_class_1/train",
         "val_dir": "./data/white_controller/yoloSeg/patch_256_class_1/test",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

    ],
    "Yolov7Obj": [
        # 0 ~ 7
        {"coco_root": "./data/white_controller/coco/original_class_4",
         "train_txt": "./data/white_controller/yoloBbox/original_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/original_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/original_class_1",
         "train_txt": "./data/white_controller/yoloBbox/original_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/original_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_4",
         "train_txt": "./data/white_controller/yoloBbox/patch_1024_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_1024_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_1",
         "train_txt": "./data/white_controller/yoloBbox/patch_1024_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_1024_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_4",
         "train_txt": "./data/white_controller/yoloBbox/patch_512_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_512_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_1",
         "train_txt": "./data/white_controller/yoloBbox/patch_512_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_512_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_4",
         "train_txt": "./data/white_controller/yoloBbox/patch_256_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_256_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_1",
         "train_txt": "./data/white_controller/yoloBbox/patch_256_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_256_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "SGD",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

        # 8 ~ 15
        {"coco_root": "./data/white_controller/coco/original_class_4",
         "train_txt": "./data/white_controller/yoloBbox/original_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/original_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/original_class_1",
         "train_txt": "./data/white_controller/yoloBbox/original_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/original_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": False},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_4",
         "train_txt": "./data/white_controller/yoloBbox/patch_1024_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_1024_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_1024_class_1",
         "train_txt": "./data/white_controller/yoloBbox/patch_1024_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_1024_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [1024, 1024],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_4",
         "train_txt": "./data/white_controller/yoloBbox/patch_512_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_512_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_512_class_1",
         "train_txt": "./data/white_controller/yoloBbox/patch_512_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_512_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [512, 512],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_4",
         "train_txt": "./data/white_controller/yoloBbox/patch_256_class_4/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_256_class_4/val_list.txt",
         "number_of_class": 4,
         "class_names": ["Scratch", "Friction", "Dirty", "Assembly"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

        {"coco_root": "./data/white_controller/coco/patch_256_class_1",
         "train_txt": "./data/white_controller/yoloBbox/patch_256_class_1/train_list.txt",
         "val_txt": "./data/white_controller/yoloBbox/patch_256_class_1/val_list.txt",
         "number_of_class": 1,
         "class_names": ["Defect"],
         "optimizer": "AdamW",
         "weight": " ",
         "imgsz": [256, 256],
         "use_patch": True},

    ]
}


def get_args_parser():
    parser = argparse.ArgumentParser('Model auto training script.', add_help=False)

    parser.add_argument('--model_name', '-m', type=str, required=True,
                        choices=["Yolov7inSeg", "Yolov7Obj", "CascadeMaskRCNN", "Mask2Former"],
                        help='The name of model.')

    parser.add_argument('--start_id', type=int, default=0,
                        help='Start from which experiment id.')

    parser.add_argument('--end_id', type=int, default=15,
                        help='Stop from which experiment id.')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    start_id = args.start_id
    end_id = args.end_id
    model_name = args.model_name
    experiment_length = len(experiment_order[model_name])

    assert end_id <= experiment_length

    for idx in range(start_id, end_id + 1):
        # Load template config
        template_config = load_yaml(model_config_template[model_name])

        # Update config
        template_config.update(experiment_order[model_name][idx])

        # Create builder
        builder = Builder(yaml_dict=template_config, task='train')

        # Build config
        cfg = builder.build_config()

        # Build model
        model = builder.build_model(cfg)

        # Training
        model.train()

        # Generate training curve
        subprocess.run(['python', "./tools/training_curve.py",
                        '--model_type', cfg['model_name'],
                        '--result_path', get_work_dir_path(cfg),
                        '--output_path', get_work_dir_path(cfg),
                        ])

        torch.cuda.empty_cache()
