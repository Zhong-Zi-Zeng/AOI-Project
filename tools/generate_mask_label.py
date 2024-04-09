from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from engine.general import get_work_dir_path, rle_to_polygon
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import argparse
import cv2
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Model training script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    parser.add_argument('--source', '-s', type=str, required=True,
                        help='An image path or a directory path')

    parser.add_argument('--conf_thres', type=float, default=0.5,
                        help='Confidence threshold.')

    parser.add_argument('--nms_thres', type=float, default=0.5,
                        help='NMS threshold.')

    parser.add_argument('--dir_name', type=str,
                        help='The name of work dir.')

    return parser

def run():
    if os.path.isdir(args.source):
        file_list = os.listdir(args.source)
        source = [os.path.join(args.source, item) for item in file_list]
    elif os.path.isfile(args.source):
        source = [args.source]
    else:
        ValueError("Cannot find the source file {}".format(args.source))

    images = []
    anns = []
    cats = []
    anns_count = 0

    for cat_id, cls_name in enumerate(cfg['class_names']):
        cat_dict = {'id': cat_id, 'name': cls_name}
        cats.append(cat_dict)

    for img_id, image_file in enumerate(tqdm(source, total=len(source))):
        result = model.predict(image_file,
                               conf_thres=args.conf_thres,
                               nms_thres=args.nms_thres)

        bboxes = result['bbox_list']
        rle_list = result['rle_list']
        classes = result['class_list']

        # save to coco format
        image_name = Path(image_file).name
        # image = cv2.imread(image_file)
        # h, w, _ = image.shape
        images.append({
            'file_name': image_name,
            'height': 2048,
            'width': 3072,
            'id': img_id
        })

        # rle to polygon
        polygons = [rle_to_polygon(rle) for rle in rle_list]
        for cls, bbox, polygon in zip(classes, bboxes, polygons):
            anns.append({
                'segmentation': np.reshape(polygon, (1, -1)).tolist(),
                'area': -1,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': cls,
                'id': anns_count,
            })
            anns_count += 1

    work_dir_path = get_work_dir_path(cfg)
    with open(os.path.join(work_dir_path, 'friction.json'), 'w') as file:
        json.dump({'images': images,
                   'annotations': anns,
                   'categories': cats}, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # Create builder
    builder = Builder(config_path=args.config, task='predict', work_dir_name=args.dir_name)

    # Build config
    cfg = builder.build_config()

    # Build model
    model = builder.build_model(cfg)

    run()