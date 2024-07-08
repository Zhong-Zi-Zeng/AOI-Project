import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import subprocess
import platform
from threading import Thread

import numpy as np
import redis

from engine.general import get_work_dir_path


class TrainingManager:
    def __init__(self):
        self.training_thread = None
        self.tensorboard_proc = None
        self.training_complete = False

        os_name = platform.system()
        if os_name == "Windows":
            print("Running on Windows")
            redis_host = '127.0.0.1'
        elif os_name == "Linux":
            print("Running on Linux")
            redis_host = 'redis'
        else:
            print(f"Running on {os_name}")
            redis_host = 'redis'

        self.r = redis.Redis(host=redis_host, port=6379, db=0)

    def _train_wrapper(self, train_func):
        train_func()
        self.training_complete = True
        if self.tensorboard_proc:
            self.tensorboard_proc.terminate()
            self.tensorboard_proc = None

    def _get_redis_value(self, key: str):
        value = self.r.get(key)
        if value is None:
            return None
        try:
            return np.around(float(value), 2)
        except ValueError:
            return value.decode('utf-8')  # 如果不能轉換為float，就解碼為字串

    def _clear_redis_key(self):
        self.r.flushdb()

    def _open_tensorboard(self, final_config):
        self.tensorboard_proc = subprocess.Popen(['tensorboard',
                                                  '--logdir', get_work_dir_path(final_config),
                                                  '--host', '0.0.0.0',
                                                  '--port', '1000'])

    def start_training(self, train_func, final_config):
        self._clear_redis_key()
        self.training_complete = False
        self._open_tensorboard(final_config)
        self.training_thread = Thread(target=self._train_wrapper, args=(train_func,))
        self.training_thread.start()

    def stop_training(self):
        self.r.set("stop_training", 1)

    def get_status(self) -> dict:
        status = {}

        if self.training_thread is not None and self.training_thread.is_alive():
            status['status_msg'] = "Training in progress"
        elif self.training_complete:
            status['status_msg'] = "Training completed"
        else:
            status['status_msg'] = "No training in progress"

        status['remaining_time'] = self._get_redis_value("remaining_time")
        status['progress'] = self._get_redis_value("progress")

        return status
