from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from converter import jsonParser
from tqdm import tqdm
import argparse
import os
import json
import yaml


def get_args_parser():
    parser = argparse.ArgumentParser('DeformableVit', add_help=False)

    parser.add_argument('--source_dir', type=str, required=True,
                        help="The dataset's path includes image files and json files.")
    parser.add_argument('--classes_yaml', type=str, required=True,
                        help='The category of training needs a YAML file.')
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


def slice_file(source_dir: str, assign_number: int, classes_name: list) -> object:    # classes_name = {'Border': 'aaa'...}
    test = {cls: {'super': classes_name[cls], 'number': 0, 'file_name': []} for cls in classes_name}
    total = {cls: {'super': classes_name[cls], 'number': 0, 'file_name': []} for cls in classes_name}

    image_files_path = [os.path.join(source_dir, image_name) for image_name in os.listdir(source_dir)
                        if is_image(os.path.join(source_dir, image_name))]

    json_files_path = [os.path.join(source_dir, json_name) for json_name in os.listdir(source_dir)
                       if is_json(os.path.join(source_dir, json_name))]

    for idx, (old_image_file, old_json_file) in tqdm(enumerate(zip(image_files_path, json_files_path)),
                                                     total=len(image_files_path)):
        image_height, image_width, mask, classes, bboxes, polygons = jsonParser(old_json_file).parse()

        new_image_file = os.path.join(str(Path(old_image_file).parent), str(idx) + '.jpg')

        # total
        for cls in set(classes):
            total[cls.replace('#', '')]['number'] += 1
            total[cls.replace('#', '')]['file_name'].append(new_image_file)

        # for testing
        cls = classes[0].replace('#', '')
        if test[cls]['number'] != assign_number and len(set(classes)) == 1:
            test[cls]['number'] += 1
            test[cls]['file_name'].append(new_image_file)

            # 照片移到source/test資料夾下
            os.rename(old_image_file, os.path.join(source_dir, 'test', str(idx) + '.jpg'))
            os.rename(old_json_file, os.path.join(source_dir, 'test', str(idx) + '.json'))
        else:
            # 照片移到source/train資料夾下
            os.rename(old_image_file, os.path.join(source_dir, 'train', str(idx) + '.jpg'))
            os.rename(old_json_file, os.path.join(source_dir, 'train', str(idx) + '.json'))

    # 把描述檔存在source資料夾下
    with open(os.path.join(source_dir, 'test_detail.json'), 'w') as file:
        json.dump(test, file, indent=2)
    with open(os.path.join(source_dir, 'all_detail.json'), 'w') as file:
        json.dump(total, file, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset slice script. ',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.classes_yaml, 'r') as file:
        # classes_name = [cls.rstrip() for cls in file.readlines()]  # txt
        classes_data = yaml.safe_load(file)     # yaml
        classes_name = {cls: data.get('super') for cls, data in classes_data.items()}  # {'Border': 'aaa'...}

    os.mkdir(os.path.join(args.source_dir, 'train'))
    os.mkdir(os.path.join(args.source_dir, 'test'))
    os.mkdir(os.path.join(args.source_dir, 'delete'))

    check_file(args.source_dir)
    slice_file(source_dir=args.source_dir,
               assign_number=args.assign_number,
               classes_name=classes_name)
