import os
import subprocess

from .yolov7_obj import Yolov7Obj
from engine.general import (get_work_dir_path, load_yaml, save_yaml, get_model_path, check_path)


class Yolov7W6Obj(Yolov7Obj):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg

    def train(self):
        command = ['python',
                   os.path.join(get_model_path(self.cfg), 'train_aux.py'),
                   '--data', self.cfg['data_file'],
                   '--cfg', self.cfg['cfg_file'],
                   '--hyp', self.cfg['hyp_file'],
                   '--batch-size', str(self.cfg['batch_size']),
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

        if self.cfg['weight'] is not None:
            assert check_path(self.cfg['weight']), "The weight file does not exist."

            if self.cfg['resume_training']:
                self._check_valid_epoch()
            else:
                self._reset_epoch_and_iter()

            command.append('--weights')
            command.append(self.cfg['weight'])

        subprocess.run(command)
