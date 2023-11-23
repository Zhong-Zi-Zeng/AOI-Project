from __future__ import annotations
import copy
from patchify import patchify
from pathlib import Path
from copy import deepcopy
import cv2
import json
import os
import numpy as np
import argparse
import shutil


def get_args_parser():
    parser = argparse.ArgumentParser('DeformableVit', add_help=False)
    parser.add_argument('--source', type=str,
                        help="The dataset's path includes image files and json files.")
    parser.add_argument('--output', type=str,
                        help="Save the result in this directory. Recommend to use .../color to name")
    parser.add_argument('--training_classes', type=str,
                        help='The category of training needs a Txt file.')
    parser.add_argument('--testing_classes', type=str,
                        help='The category of testing needs a Txt file. '
                             'If not specified, then same as the category of training.')
    parser.add_argument('--quantity', type=int, default=3,
                        help='Each test\'s category quantity. The default is 3.')
    parser.add_argument('--dataset', type=str,
                        help="Have been processed datasets for generating others type of image.")
    parser.add_argument('--patch_size', type=int,
                        help='The size of the patch needs to be divisible by width and height. '
                             'If you assign the value, the script will generate a patch dataset')

    return parser


def is_image(image_path: str) -> bool:
    """
       判斷該路徑是不是圖像
       Arg:
            image_path: 影像路徑
       Return:
            True or False
   """
    assert os.path.isfile(image_path), 'Can\'t find the file {}.'.format(image_path)

    _allow_format = ['.jpg', '.png', '.bmp']
    return Path(image_path).suffix in _allow_format


def is_json(json_path: str) -> bool:
    """
        判斷該路徑是不是json檔
        Arg:
            json_path: json檔路徑
        Return:
            True or False
    """
    assert os.path.isfile(json_path), 'Can\'t find the file {}.'.format(json_path)
    return Path(json_path).suffix == '.json'


def divide_to_patch(large_image: np.ndarray[np.uint8 | np.float],
                    patch_size: int,
                    step: int) -> np.ndarray:
    """
        將影像劃分為patch, 需要注意輸入的影像大小要能被patch_size整除
        Arg:
            large_image: [H, W, C] for rgb or [H, W] for mask
            patch_size: patch的長寬
            step: 如果等於patch_size，則不會有重疊的部分
        Return:
            patches_img: [N, patch_size, patch_size, C], N為patch的數量
    """

    assert large_image.shape[0] % patch_size == 0 and \
           large_image.shape[1] % patch_size == 0

    if len(large_image.shape) == 3:
        patches_img = patchify(large_image, (patch_size, patch_size, 3), step=step)
        patches_img = np.reshape(patches_img, (-1, patch_size, patch_size, 3))
    else:
        patches_img = patchify(large_image, (patch_size, patch_size), step=step)
        patches_img = np.reshape(patches_img, (-1, patch_size, patch_size))

    return patches_img



def get_mask(json_file_path: str) -> [np.ndarray[np.float] | None]:
    """
        將傳入的json檔進行解析，將其中的polygon資訊轉換為一張mask

        Args:
            json_file_path: json檔的位置
        Returns:
            mask: 二值化的影像 [H, W, 3], 若有錯誤則返回None
    """
    with open(json_file_path, 'r') as file:
        text = json.loads(file.read())

    objects = text['Objects']  # list
    h = int(text['Background']['Height'])
    w = int(text['Background']['Width'])

    mask = np.zeros(shape=(h, w))

    try:
        for obj in objects:
            polygon = obj['Layers'][0]['Shape']['Points']
            polygon = np.array([list(map(float, poly.split(','))) for poly in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], (255, 255, 255))

        return mask
    except KeyError as key:
        print('{} can\'t find the key of {}'.format(json_file_path, key))
        return None


class DataProcess:
    def __init__(self, args):
        self.args = args
        self.color = Path(args.output).name

        # 新增資料夾
        if not os.path.isdir(self.args.output):
            os.mkdir(self.args.output)  # 新增根目錄
            os.mkdir(os.path.join(self.args.output, 'images'))  # 新增images資料夾
            os.mkdir(os.path.join(self.args.output, 'images', 'train'))  # 新增train資料夾
            os.mkdir(os.path.join(self.args.output, 'images', 'test'))  # 新增test資料夾
            os.mkdir(os.path.join(self.args.output, 'json'))  # 新增json資料夾
            os.mkdir(os.path.join(self.args.output, 'delete'))  # 新增delete資料夾

        # 檢查是否要生成original資料夾
        if not os.path.isdir(os.path.join(self.args.output, 'images', 'train', 'original')):
            # 讀取training classes
            with open(self.args.training_classes, 'r') as file:
                self.training_classes = file.readlines()
                self.training_classes = {cls.rstrip(): {'Number': 0, 'Name': []} for cls in self.training_classes}

            # 讀取testing classes
            if self.args.testing_classes is None:
                self.testing_classes = copy.deepcopy(self.training_classes)
            else:
                with open(self.args.testing_classes, 'r') as file:
                    self.testing_classes = file.readlines()
                    self.testing_classes = {cls.rstrip(): {'Number': 0, 'Name': []} for cls in self.testing_classes}

            self._generate_mode_dir('original')
            self._generate_original()

        # 生成patch
        if self.args.patch_size is not None:
            self._generate_mode_dir('patch_' + str(self.args.patch_size))
            # TODO: 從original中去生成patch

    def _generate_mode_dir(self, mode: str):
        """
            生成指定mode的資料夾，包括img、label、bbox、polygon、mask

            Args:
                mode: 資料夾名稱
        """
        tasks = ['train', 'test']

        for task in tasks:
            mode_dir = os.path.join(self.args.output, 'images', task, mode)
            os.makedirs(os.path.join(mode_dir, 'img'))
            label_dir = os.path.join(mode_dir, 'label')
            os.makedirs(label_dir)
            os.makedirs(os.path.join(label_dir, 'bbox'))
            os.makedirs(os.path.join(label_dir, 'polygon'))
            os.makedirs(os.path.join(label_dir, 'mask'))

    def _generate_original(self):
        # 把source中所有的image和json取出
        image_files = [os.path.join(self.args.source, image_name) for image_name in os.listdir(self.args.source)
                       if is_image(os.path.join(self.args.source, image_name))]

        json_files = [os.path.join(self.args.source, json_name) for json_name in os.listdir(self.args.source)
                      if is_json(os.path.join(self.args.source, json_name))]

        image_files_copy = deepcopy(image_files)
        json_files_copy = deepcopy(json_files)

        # 對檔案進行匹配
        for idx, image_file in enumerate(image_files):
            # 尋找對應的json檔
            correspond_json_file = image_file + '.json'

            # 如果沒有對應的json檔則繼續尋找
            if correspond_json_file not in json_files:
                continue

            # 如果json檔中找不到mask則續續尋找
            mask = get_mask(correspond_json_file)
            if mask is None:
                continue

            # 將mask存在mask資料夾下
            cv2.imwrite(
                os.path.join(self.args.output, 'images', 'train', 'original', 'label', 'mask',
                             self.color + "_" + str(idx) + '.jpg'),
                mask)

            # 讀取class

            # TODO: 將bbox的txt存在bbox資料夾下

            # TODO: 將polygon的txt存在polygon資料夾下

            # TODO: 生成描述檔

            # TODO: 取指定類別的資料到test資料夾下

            # 將json檔複製一份到json資料夾下
            shutil.copy(correspond_json_file,
                        os.path.join(self.args.output, 'json',
                                     self.color + "_" + str(idx) + '.json'))

            # 將image複製一份到img資料夾下
            shutil.copy(image_file,
                        os.path.join(self.args.output, 'images', 'train', 'original', 'img',
                                     self.color + "_" + str(idx) + '.jpg'))

            # 將對應到的檔案從copy list中移除
            image_files_copy.remove(image_file)
            json_files_copy.remove(correspond_json_file)

        # 把剩下在image_files_copy裡的檔案複製一份到delete資料夾下
        for except_image_file in image_files_copy:
            shutil.copy(except_image_file, os.path.join(self.args.output, 'delete', Path(except_image_file).name))

        # 把剩下在json_files_copy裡的檔案複製一份到delete資料夾下
        for except_json_file in json_files_copy:
            shutil.copy(except_json_file, os.path.join(self.args.output, 'delete', Path(except_json_file).name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset process script. '
                                     'If you run this script first time,'
                                     'you need to fill out both --source and --output parameters.\n'
                                     'Otherwise, you can specify assign that both --dataset and --patch_size '
                                     'so that can generate a patch dataset.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    DataProcess(args)
