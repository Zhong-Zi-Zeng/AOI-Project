from .yolov7_obj import Yolov7Obj
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, check_path)
import os
import subprocess

class Yolov7W6Obj(Yolov7Obj):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def train(self):
        subprocess.run(['python',
                        os.path.join(get_model_path(self.cfg), 'train_aux.py'),
                        '--data', self.cfg['data_file'],
                        '--cfg', self.cfg['cfg_file'],
                        '--hyp', self.cfg['hyp_file'],
                        '--batch-size', str(self.cfg['batch_size']),
                        '--weights', self.cfg['weight'] if check_path(self.cfg['weight']) else " ",
                        '--epochs', str(self.cfg['end_epoch']),
                        '--project', get_work_dir_path(self.cfg),
                        '--optimizer', self.cfg['optimizer'],
                        '--device', self.cfg['device'],
                        '--name', './',
                        '--save_period', str(self.cfg['save_period']),
                        '--eval_period', str(self.cfg['eval_period']),
                        '--img-size', str(self.cfg['imgsz'][0]),
                        '--exist-ok',
                        ]
                       )