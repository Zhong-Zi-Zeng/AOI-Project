from __future__ import annotations
from engine.general import get_model_path, get_work_dir_path, load_python, update_python_file
from mmdet.apis import DetInferencer
import os
import subprocess
import platform


class BaseMMdetection:
    def __init__(self, cfg: dict, optimizer: dict, transforms: list):
        self.cfg = cfg
        self.optimizer = optimizer
        self.transforms = transforms

    def _config_transform(self):
        config_dict = load_python(self.cfg['cfg_file'])

        # Update base file path
        new_base = []
        for base in config_dict['_base_']:
            new_base.append(os.path.join(get_model_path(self.cfg), 'configs', base))

        # Update config file
        variables = {
            '_base_': new_base,
            'data_root': self.cfg['coco_root'],
            'classes': self.cfg['class_names'],
            'batch_size': self.cfg['batch_size'],
            'epochs': self.cfg['end_epoch'],
            'height': self.cfg['imgsz'][0],
            'width': self.cfg['imgsz'][1],
            'num_classes': self.cfg['number_of_class'],
            'lr': self.cfg['lr'],
            'start_factor': self.cfg['initial_lr'] / self.cfg['lr'],
            'minimum_lr': self.cfg['minimum_lr'],
            'warmup_begin': 0,
            'warmup_end': self.cfg['warmup_epoch'],
            'optim_wrapper': self.optimizer,
            'albu_train_transforms': self.transforms,
            'check_interval': self.cfg['save_period'],
            'nms_threshold': self.cfg['nms_thres'],
        }

        update_python_file(self.cfg['cfg_file'], os.path.join(get_work_dir_path(self.cfg), 'cfg.py'), variables)
        self.cfg['cfg_file'] = os.path.join(get_work_dir_path(self.cfg), 'cfg.py')

    def _build_optimizer(self,
                         weight_decay: float = 0.0005,
                         **kwargs) -> dict:

        optimizer = dict(_delete_=True, type='OptimWrapper')
        optimizer.update(kwargs)

        if self.cfg['optimizer'] == 'SGD':
            optimizer['optimizer'] = dict(type='SGD', lr=self.cfg['lr'], momentum=0.937,
                                          weight_decay=weight_decay)
        elif self.cfg['optimizer'] == 'Adam':
            optimizer['optimizer'] = dict(type='Adam', lr=self.cfg['lr'], betas=(0.937, 0.999),
                                          weight_decay=weight_decay)
        elif self.cfg['optimizer'] == 'AdamW':
            optimizer['optimizer'] = dict(type='AdamW', lr=self.cfg['lr'], betas=(0.937, 0.999),
                                          weight_decay=weight_decay)

        return optimizer

    def _build_augmentation(self) -> list:
        transforms = [
            dict(
                type='ColorJitter',
                hue=self.cfg['hsv_h'],
                saturation=self.cfg['hsv_s'],
                brightness=self.cfg['hsv_v']),
            dict(
                type='Affine',
                scale=self.cfg['scale'],
                translate_percent=self.cfg['translate'],
                shear=self.cfg['shear'],
                rotate=self.cfg['degrees']
            ),
            dict(type='Perspective', scale=self.cfg['perspective']),
            dict(type='HorizontalFlip', p=self.cfg['fliplr']),
            dict(type='VerticalFlip', p=self.cfg['flipud']),
        ]
        return transforms

    def train(self):
        system = platform.system()
        if system == 'Linux':
            env_vars = {'CUDA_VISIBLE_DEVICES': self.cfg['device'], 'PORT': '29500'}
            dist_train_sh = os.path.join(get_model_path(self.cfg), 'tools', 'dist_train.sh')

            with open(dist_train_sh, 'rb') as file:
                script_content = file.read().replace(b'\r\n', b'\n')

            with open(dist_train_sh, 'wb') as file:
                file.write(script_content)

            # TODO: resume from pretrained weight
            command = f'{dist_train_sh} {self.cfg["cfg_file"]} 1 --work-dir {get_work_dir_path(self.cfg)}'
            proc = subprocess.Popen(command, shell=True, env={**os.environ, **env_vars},
                                    stderr=subprocess.PIPE, executable='/bin/bash')
            proc.communicate()
        else:
            command = [
                'python',
                os.path.join(get_model_path(self.cfg), 'tools', 'train.py'),
                self.cfg['cfg_file'],
                '--work-dir', get_work_dir_path(self.cfg)
            ]
            if os.path.exists(self.cfg['weight']):
                command.extend(['--resume', self.cfg['weight']])

            subprocess.run(command)

    def _load_model(self):
        self.model = DetInferencer(model=self.cfg['cfg_file'],
                                   weights=self.cfg['weight'],
                                   show_progress=False)
