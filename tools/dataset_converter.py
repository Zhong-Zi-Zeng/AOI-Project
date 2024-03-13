# 只需輸入 coco dataset路徑
# 轉換成 yoloBbox 或 yoloSeg
import os
import sys
import shutil
import json
import itertools

class coco2yoloBbox():
    def __init__(self, coco_path):
        # The storage path of the yoloBbox dataset
        self.yoloBbox_save_path = coco_path.replace('coco', 'yoloBbox')
        if not os.path.exists(self.yoloBbox_save_path):
            os.makedirs(self.yoloBbox_save_path)
        else:   # Check
            print("yolobbox dataset already exists.")
            sys.exit()
        # The path to the json file in the coco dataset
        self.coco_train = coco_path + '/annotations/instances_train2017.json'
        self.coco_val = coco_path + '/annotations/instances_val2017.json'
        # Create folder
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloBbox_save_path, 'labels', 'val'), exist_ok=True)
        # Copy coco images to yoloBbox
        self.copy_images(os.path.join(coco_path, 'train2017'), os.path.join(self.yoloBbox_save_path, 'images', 'train'))
        self.copy_images(os.path.join(coco_path, 'val2017'), os.path.join(self.yoloBbox_save_path, 'images', 'val'))

    def copy_images(self, source_dir, dest_dir):
        for filename in os.listdir(source_dir):
            shutil.copy(os.path.join(source_dir, filename), dest_dir)

    def conv_box(self, size, box):
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
        tasks = ['train', 'val']
        for task in tasks:

            data = json.load(open(getattr(self, f'coco_{task}'), 'r'))

            id_map = {}
            for i, category in enumerate(data['categories']):
                id_map[category['id']] = i

            # 寫入image的相對路徑
            list_file = open(os.path.join(self.yoloBbox_save_path, f'{task}_list.txt'), 'w')

            for img in data['images']:
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
                        box = self.conv_box((img_width, img_height), ann["bbox"])
                        f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
                f_txt.close()

                # 寫入image的相對路徑
                list_file.write(f'./images/{task}/%s.jpg\n' % (head))
            list_file.close()
        print('finish!')


class coco2yoloSeg():
    def __init__(self, coco_path):
        # The storage path of the yoloSeg dataset
        self.yoloSeg_save_path = coco_path.replace('coco', 'yoloSeg')
        if not os.path.exists(self.yoloSeg_save_path):
            os.makedirs(self.yoloSeg_save_path)
        else:   # Check
            print("yoloSeg dataset already exists.")
            sys.exit()
        # The path to the json file in the coco dataset
        self.coco_train = coco_path + '/annotations/instances_train2017.json'
        self.coco_test = coco_path + '/annotations/instances_val2017.json'
        # Create folder
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'test', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.yoloSeg_save_path, 'test', 'labels'), exist_ok=True)
        # Copy coco images to yolobbox
        self.copy_images(os.path.join(coco_path, 'train2017'), os.path.join(self.yoloSeg_save_path, 'train', 'images'))
        self.copy_images(os.path.join(coco_path, 'val2017'), os.path.join(self.yoloSeg_save_path, 'test', 'images'))

    def copy_images(self, source_dir, dest_dir):
        for filename in os.listdir(source_dir):
            shutil.copy(os.path.join(source_dir, filename), dest_dir)

    def conv_polygon(self, size, seg):  # normalize
        '''
            seg = [[637, 446, 636, 447,...]]
        '''

        dw = 1. / (size[0])
        dh = 1. / (size[1])

        polygons = []
        for i, point in enumerate(seg[0]):
            polygons.append(point * dw if i % 2 == 0 else point * dh)

        return polygons

    def convert(self):
        tasks = ['train', 'test']
        for task in tasks:

            data = json.load(open(getattr(self, f'coco_{task}'), 'r'))

            id_map = {}
            for i, category in enumerate(data['categories']):
                id_map[category['id']] = i

            for img in data['images']:
                filename = img["file_name"]
                img_width = img["width"]
                img_height = img["height"]
                img_id = img["id"]
                head, tail = os.path.splitext(filename)
                ana_txt_name = head + ".txt"

                # 寫入每個image的txt (label)
                f_txt = open(os.path.join(self.yoloSeg_save_path, f'{task}', 'labels', ana_txt_name), 'w')
                for ann in data['annotations']:
                    if ann['image_id'] == img_id:
                        polygons = self.conv_polygon((img_width, img_height), ann["segmentation"])  # list
                        flattened_polygons = ' '.join(str(coord) for coord in polygons)
                        f_txt.write("%s %s\n" % (id_map[ann["category_id"]], flattened_polygons))
                f_txt.close()
        print('finish!')


if __name__ == '__main__':
    coco_path = './data/test/coco/original_class_2'  # input
    task = 'yolov7_inSeg'

    if task == 'yolov7_obj':
        conv = coco2yoloBbox(coco_path)
        conv.convert()
    elif task == 'yolov7_inSeg':
        conv = coco2yoloSeg(coco_path)
        conv.convert()
    else:   # 不轉換
        pass




