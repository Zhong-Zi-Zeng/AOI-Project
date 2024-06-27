from __future__ import annotations
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from engine.general import get_work_dir_path
from tqdm import tqdm
from pathlib import Path
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

    parser.add_argument('--show', action='store_true',
                        help='Whether show the resulting image.')

    return parser


def run():
    if os.path.isdir(args.source):
        file_list = os.listdir(args.source)
        source = [os.path.join(args.source, item) for item in file_list]
    elif os.path.isfile(args.source):
        source = [args.source]
    else:
        ValueError("Cannot find the source file {}".format(args.source))

    image_1 = cv2.imread("./data/WC-100/val2017/0.jpg")
    image_2 = cv2.imread("./data/WC-100/val2017/1.jpg")
    image_3 = cv2.imread("./data/WC-100/val2017/2.jpg")

    # input = np.stack((image_1, image_2, image_3), axis=0)
    input = ["./data/WC-100/val2017/0.jpg",
             "./data/WC-100/val2017/1.jpg",
             "./data/WC-100/val2017/2.jpg",
             "./data/WC-100/val2017/3.jpg",
             "./data/WC-100/val2017/4.jpg"]

    for image_file in tqdm(source, total=len(source)):
        result = model.predict(input,
                               conf_thres=args.conf_thres,
                               nms_thres=args.nms_thres)

    # result_image = result['result_image']


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
