from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import cv2
import matplotlib.pyplot as plt


def main(args):
    assert len(args.source_list) == len(args.title_list)

    # 創建資料夾
    os.makedirs(args.output_dir, exist_ok=True)

    # 從json檔中取出每個類別的第一張照片
    with open(args.test_detail, 'r') as file:
        text = json.loads(file.read())

    image_name_list = [[k, Path(v['file_name'][0]).name] for k, v in text.items()]


    number_of_image = len(args.source_list)

    for class_name, image_name in tqdm(image_name_list):
        plt.figure(figsize=(10, 5), dpi=600)
        for i in range(number_of_image):
            image = cv2.imread(os.path.join(args.source_list[i], image_name))
            plt.subplot(1, number_of_image, i + 1)
            plt.axis('off')
            plt.imshow(image)
            plt.title(args.title_list[i], fontsize=12)

        plt.savefig(os.path.join(args.output_dir, class_name + '.jpg'), pad_inches=0, bbox_inches='tight')

def get_args_parser():
    parser = argparse.ArgumentParser('Combine predict script', add_help=False)

    parser.add_argument('-s', '--source', dest='source_list', nargs='+', required=True,
                        help="The path of predicted image.")
    parser.add_argument('-t', '--title', dest='title_list', nargs='+', required=True,
                        help="The name corresponding to each source dir.")
    parser.add_argument('--test_detail', type=str, required=True,
                        help="The json file of testing. It includes what type of class in each image."
                             "Please run the ``slice_data.py`` that can get this json file.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Save the result in this directory.")

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combine predict script. ',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)