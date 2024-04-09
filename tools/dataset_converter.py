# 只需輸入 coco dataset路徑
# 轉換成 yoloBbox 或 yoloSeg
import json
import os
import shutil

import numpy as np
from tqdm import tqdm


class coco2yoloBbox():
    def __init__(self, coco_path):
        self.coco_path = coco_path
        self.coco_train = self.coco_path + '/annotations/instances_train2017.json'
        self.coco_val = self.coco_path + '/annotations/instances_val2017.json'

        # The storage path of the yoloBbox dataset
        folder_name = os.path.basename(self.coco_path)
        self.yoloBbox_save_path = os.path.join(os.getcwd(), 'data', 'yoloBbox', folder_name)
        print(self.yoloBbox_save_path)
        yoloBbox_train_images = os.path.join(self.yoloBbox_save_path, 'images', 'train')
        yoloBbox_val_images = os.path.join(self.yoloBbox_save_path, 'images', 'val')

        # Check
        if not os.path.exists(self.yoloBbox_save_path):
            self._generate_dir()

        if not any(file.endswith('.jpg') or file.endswith('.png') for file in os.listdir(yoloBbox_train_images)):
            self._copy_images(os.path.join(self.coco_path, 'train2017'), yoloBbox_train_images, 'training images')
        else:
            print('Training images already exist.')

        if not any(file.endswith('.jpg') or file.endswith('.png') for file in os.listdir(yoloBbox_val_images)):
            self._copy_images(os.path.join(self.coco_path, 'val2017'), yoloBbox_val_images, 'validation images')
        else:
            print('Validation images already exist.')

    def _generate_dir(self):
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'labels', 'val'), exist_ok=True)

    def _copy_images(self, source_dir, dest_dir, task):
        for filename in tqdm(os.listdir(source_dir), desc='Processing %s' % task):
            shutil.copy(os.path.join(source_dir, filename), dest_dir)

    def _conv_box(self, size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]

        x = round(x * dw, 11)
        w = round(w * dw, 11)
        y = round(y * dh, 11)
        h = round(h * dh, 11)
        return (x, y, w, h)

    def convert(self):
        tasks = []
        yoloBbox_train_labels = os.path.join(self.yoloBbox_save_path, 'labels', 'train')
        yoloBbox_val_labels = os.path.join(self.yoloBbox_save_path, 'labels', 'val')

        # Check
        if os.path.exists(yoloBbox_train_labels) and os.listdir(yoloBbox_train_labels):
            print('Training labels already exists.')
        else:
            tasks.append('train')

        if os.path.exists(yoloBbox_val_labels) and os.listdir(yoloBbox_val_labels):
            print('Validation labels already exists.')
        else:
            tasks.append('val')

        # Converter
        for task in tasks:
            data = json.load(open(getattr(self, f'coco_{task}'), 'r'))
            id_map = {}
            for i, category in enumerate(data['categories']):
                id_map[category['id']] = i

            # 寫入image的相對路徑
            list_file = open(os.path.join(self.yoloBbox_save_path, f'{task}_list.txt'), 'w')

            for img in tqdm(data['images'], desc='Processing %s labels' % task):
                filename = img["file_name"]
                img_width = img["width"]
                img_height = img["height"]
                img_id = img["id"]
                head, tail = os.path.splitext(filename)
                ana_txt_name = head + ".txt"

                # 寫入每個image的txt (label)
                f_txt = open(os.path.join(self.yoloBbox_save_path, 'labels', f'{task}', ana_txt_name), 'w')
                for ann in data['annotations']:
                    if ann['image_id'] == img_id:
                        box = self._conv_box((img_width, img_height), ann["bbox"])
                        f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
                f_txt.close()

                # 寫入image的相對路徑
                list_file.write(f'./images/{task}/%s.jg\n' % (head))
            list_file.close()


class coco2yoloSeg():
    def __init__(self, coco_path):

        self.coco_path = coco_path
        self.coco_train = self.coco_path + '/annotations/instances_train2017.json'
        self.coco_test = self.coco_path + '/annotations/instances_val2017.json'

        # The storage path of the yoloSeg dataset
        folder_name = os.path.basename(self.coco_path)
        self.yoloSeg_save_path = os.path.join(os.getcwd(), 'data', 'yoloSeg', folder_name)
        yoloSeg_train_images = os.path.join(self.yoloSeg_save_path, 'train', 'images')
        yoloSeg_test_images = os.path.join(self.yoloSeg_save_path, 'test', 'images')

        # Check
        if not os.path.exists(self.yoloSeg_save_path):
            self._generate_dir()

        if not any(file.endswith('.jpg') or file.endswith('.png') for file in os.listdir(yoloSeg_train_images)):
            self._copy_images(os.path.join(self.coco_path, 'train2017'), yoloSeg_train_images, 'training images')
        else:
            print('Training images already exist.')

        if not any(file.endswith('.jpg') or file.endswith('.png') for file in os.listdir(yoloSeg_test_images)):
            self._copy_images(os.path.join(self.coco_path, 'val2017'), yoloSeg_test_images, 'testing images')
        else:
            print('Validation images already exist.')

    def _generate_dir(self):
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'test', 'labels'), exist_ok=True)

    def _copy_images(self, source_dir, dest_dir, task):
        for filename in tqdm(os.listdir(source_dir), desc='Processing %s' % task):
            shutil.copy(os.path.join(source_dir, filename), dest_dir)

    def _conv_polygon(self, size, seg):  # normalize
        '''
            seg = [[637, 446, 636, 447,...]]
        '''

        dw = 1. / (size[0])
        dh = 1. / (size[1])

        seg = np.array(seg, dtype=np.float64).reshape((-1, 2))
        seg[:, 0] *= dw
        seg[:, 1] *= dh

        # polygons = []
        # for i, point in enumerate(seg[0]):
        #     polygons.append(point * dw if i % 2 == 0 else point * dh)
        # return polygons
        return seg.reshape((-1,)).tolist()


    def convert(self):
        tasks = []
        yoloSeg_train_labels = os.path.join(self.yoloSeg_save_path, 'train', 'labels')
        yoloSeg_test_labels = os.path.join(self.yoloSeg_save_path, 'test', 'labels')

        # Check
        if os.path.exists(yoloSeg_train_labels) and os.listdir(yoloSeg_train_labels):
            print('Training labels already exists.')
        else:
            tasks.append('train')

        if os.path.exists(yoloSeg_test_labels) and os.listdir(yoloSeg_test_labels):
            print('Testing labels already exists.')
        else:
            tasks.append('test')

        # Converter
        for task in tasks:
            data = json.load(open(getattr(self, f'coco_{task}'), 'r'))

            id_map = {}
            for i, category in enumerate(data['categories']):
                id_map[category['id']] = i

            for img in tqdm(data['images'], desc='Processing %sing labels' % task):
                filename = img["file_name"]
                img_width = img["width"]
                img_height = img["height"]
                img_id = img["id"]
                head, tail = os.path.splitext(filename)
                ana_txt_name = head + ".txt"

                # 寫入每個image的txt (label)
                f_txt = open(os.path.join(self.yoloSeg_save_path, f'{task}', 'labels', ana_txt_name), 'w')
                for ann in data['annotations']:
                    if ann['image_id'] == img_id and ann["segmentation"]:   # normal data直接跳過
                        polygons = self._conv_polygon((img_width, img_height), ann["segmentation"])  # list
                        flattened_polygons = ' '.join(str(coord) for coord in polygons)
                        f_txt.write("%s %s\n" % (id_map[ann["category_id"]], flattened_polygons))
                f_txt.close()


if __name__ == '__main__':
    coco_path = './data/test/coco/original_class_2'  # input
    task = 'yolov7_inSeg'

    if task == 'yolov7_obj':
        conv = coco2yoloBbox(coco_path)
        conv.convert()
    elif task == 'yolov7_inSeg':
        conv = coco2yoloSeg(coco_path)
        conv.convert()
    else:  # 不轉換
        pass
