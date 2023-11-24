from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from converter import jsonParser
import argparse
import os
import json


def get_args_parser():
    parser = argparse.ArgumentParser('DeformableVit', add_help=False)

    parser.add_argument('--source_dir', type=str, required=True,
                        help="The dataset's path includes image files and json files.")
    parser.add_argument('--classes_txt', type=str, required=True,
                        help='The category of training needs a Txt file.')
    parser.add_argument('--assign_number', type=int, default=3,
                        help="Each number of categories in the test dataset. The default is 3.")

    return parser


def is_image(image_path: str) -> bool:
    """
       判斷該路徑是不是圖像
       Arg:
            image_path: 影像路徑
       Return:
            True or False
   """
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
    return Path(json_path).suffix == '.json'


def check_file(source_dir: str):
    """
        對source資料夾下的圖片和json進行配對, 若有問題的檔案則會被移動到
        source_dir/delete 資料夾下
    """
    # 把source中所有的image和json取出
    image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                        if is_image(os.path.join(source_dir, image_name))]

    json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                       if is_json(os.path.join(source_dir, json_name))]

    image_files_copy = deepcopy(image_files_path)
    json_files_copy = deepcopy(json_files_path)

    # 對檔案進行匹配
    for image_file in image_files_path:
        # 尋找對應的json檔
        correspond_json_file = image_file + '.json'

        # 檢查對應的json檔是否有在source資料夾下
        if correspond_json_file not in json_files_path:
            continue

        # 檢查對應的json檔內容是否正確
        if not jsonParser(correspond_json_file).check_json():
            continue

        # 將對應到的檔案從copy list中移除
        image_files_copy.remove(image_file)
        json_files_copy.remove(correspond_json_file)

    # 把剩下在的檔案移動到delete資料夾下
    for except_image_file in image_files_copy:
        os.rename(except_image_file, os.path.join(source_dir, 'delete', Path(except_image_file).name))
    for except_json_file in json_files_copy:
        os.rename(except_json_file, os.path.join(source_dir, 'delete', Path(except_json_file).name))


def slice_file(source_dir: str, assign_number: int, classes_name: list):
    test = {cls: {'number': 0, 'file_name': []} for cls in classes_name}

    image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                        if is_image(os.path.join(source_dir, image_name))]

    json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                       if is_json(os.path.join(source_dir, json_name))]

    for image_file, json_file in zip(image_files_path, json_files_path):
        image_height, image_width, mask, classes, bboxes, polygons = jsonParser(json_file).parse()

        cls = classes[0].replace('#', '')
        if test[cls]['number'] != assign_number and len(set(classes)) == 1:
            test[cls]['number'] += 1
            test[cls]['file_name'].append(image_file)

            # 照片移到source/test資料夾下
            os.rename(image_file, os.path.join(source_dir, 'test', Path(image_file).name))
            os.rename(json_file, os.path.join(source_dir, 'test', Path(json_file).name))
        else:
            # 照片移到source/train資料夾下
            os.rename(image_file, os.path.join(source_dir, 'train', Path(image_file).name))
            os.rename(json_file, os.path.join(source_dir, 'train', Path(json_file).name))

    # 把描述檔存在source資料夾下
    with open(os.path.join(source_dir, 'detail.json'), 'w') as file:
        json.dump(test, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset slice script. ',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.classes_txt, 'r') as file:
        classes_name = [cls.rstrip() for cls in file.readlines()]  # 儲存所有類別名稱

    os.mkdir(os.path.join(args.source_dir, 'train'))
    os.mkdir(os.path.join(args.source_dir, 'test'))
    os.mkdir(os.path.join(args.source_dir, 'delete'))

    check_file(args.source_dir)
    slice_file(source_dir=args.source_dir,
               assign_number=args.assign_number,
               classes_name=classes_name)
