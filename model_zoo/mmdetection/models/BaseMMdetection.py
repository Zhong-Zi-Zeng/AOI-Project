from __future__ import annotations
from engine.general import get_model_path, get_work_dir_path
from mmdet.apis import DetInferencer
import os
import subprocess
import platform


class BaseMMdetection:
    def train(self):
        system = platform.system()

        if system == 'Linux':
            env_vars = {'CUDA_VISIBLE_DEVICES': self.cfg['device'], 'PORT': '29500'}
            dist_train_sh = os.path.join(get_model_path(self.cfg), 'tools', 'dist_train.sh')

            with open(dist_train_sh, 'rb') as file:
                script_content = file.read().replace(b'\r\n', b'\n')

            with open(dist_train_sh, 'wb') as file:
                file.write(script_content)

            command = f'{dist_train_sh} {self.cfg["cfg_file"]} 1 --work-dir {get_work_dir_path(self.cfg)}'
            proc = subprocess.Popen(command, shell=True, env={**os.environ, **env_vars},
                                    stderr=subprocess.PIPE, executable='/bin/bash')
            proc.communicate()
        else:
            subprocess.run([
                'python',
                os.path.join(get_model_path(self.cfg), 'tools', 'train.py'),
                self.cfg['cfg_file'],
                '--work-dir', get_work_dir_path(self.cfg)
            ])

    def _load_model(self):
        self.model = DetInferencer(model=self.cfg['cfg_file'],
                                   weights=self.cfg['weight'],
                                   show_progress=False)
