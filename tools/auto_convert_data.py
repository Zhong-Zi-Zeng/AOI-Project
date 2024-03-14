import subprocess
import os
from threading import Thread


training_dataset = r"D:\AOI-New-Dataset\500_training"
testing_dataset = r"D:\AOI-New-Dataset\testing_dataset"
output_root = r"D:\AOI-New-Dataset\500\coco"

class_yaml_1 = r"D:\AOI\solovision_vs_yolov8\class_1_white.yaml"
class_yaml_4 = r"D:\AOI\solovision_vs_yolov8\class_2_white.yaml"

format = 'coco'
type = [
    # original_class_4
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'original_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'train',
     'format': format},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'original_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'test',
     'format': format},

    # original_class_1
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'original_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'train',
     'format': format},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'original_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'test',
     'format': format},

    # patch_1024_class_4
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'patch_1024_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'train',
     'format': format,
     'patch_size': "1024",
     'stride': "2"},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'patch_1024_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'test',
     'format': format,
     'patch_size': "1024",
     'stride': "2",
     'store_none': True},

    # patch_1024_class_1
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'patch_1024_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'train',
     'format': format,
     'patch_size': "1024",
     'stride': "2"},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'patch_1024_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'test',
     'format': format,
     'patch_size': "1024",
     'stride': "2",
     'store_none': True},

    # patch_512_class_4
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'patch_512_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'train',
     'format': format,
     'patch_size': "512",
     'stride': "2"},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'patch_512_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'test',
     'format': format,
     'patch_size': "512",
     'stride': "2",
     'store_none': True},

    # patch_512_class_1
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'patch_512_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'train',
     'format': format,
     'patch_size': "512",
     'stride': "2"},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'patch_512_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'test',
     'format': format,
     'patch_size': "512",
     'stride': "2",
     'store_none': True},

    # patch_256_class_4
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'patch_256_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'train',
     'format': format,
     'patch_size': "256",
     'stride': "2"},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'patch_256_class_2'),
     'classes_yaml': class_yaml_4,
     'dataset_type': 'test',
     'format': format,
     'patch_size': "256",
     'stride': "2",
     'store_none': True},

    # patch_256_class_1
    {'source_dir': training_dataset,
     'output_dir': os.path.join(output_root, 'patch_256_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'train',
     'format': format,
     'patch_size': "256",
     'stride': "2"},

    {'source_dir': testing_dataset,
     'output_dir': os.path.join(output_root, 'patch_256_class_1'),
     'classes_yaml': class_yaml_1,
     'dataset_type': 'test',
     'format': format,
     'patch_size': "256",
     'stride': "2",
     'store_none': True},
]


def convert(setting: dict):
    commend = ['python', "./tools/create_data.py",
               '--source_dir', setting['source_dir'],
               '--output_dir', setting['output_dir'],
               '--classes_yaml', setting['classes_yaml'],
               '--dataset_type', setting['dataset_type'],
               '--format', format,
               ]

    if setting.get('patch_size') is not None:
        commend.append('--patch_size')
        commend.append(setting['patch_size'])
        commend.append('--stride')
        commend.append(setting['stride'])

        if setting.get('store_none') is not None:
            commend.append('--store_none')
    subprocess.run(commend)


# Thread
for i in range(16):
    Thread(target=convert, args=(type[i], )).start()
    # time.sleep(1)