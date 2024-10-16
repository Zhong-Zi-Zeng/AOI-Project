import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import platform
from threading import Thread

import numpy as np
import redis

from engine.general import (get_work_dir_path, copy_logfile_to_work_dir, clear_cache,
                            load_json, check_path, ROOT, TEMP_DIR)


class TrainingManager:
    def __init__(self):
        self.training_thread = None
        self.final_config = None

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

        # ========= After Training =========
        copy_logfile_to_work_dir(self.final_config)
        self._clear_redis_key()
        clear_cache()

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

    def start_training(self, train_func, final_config):
        # ========= Before Training =========
        self.final_config = final_config
        self._clear_redis_key()
        clear_cache()

        self.training_thread = Thread(target=self._train_wrapper, args=(train_func,))
        self.training_thread.start()

    def stop_training(self):
        self.r.set("stop_training", 1)

    def get_status(self) -> dict:
        status = {}

        if self.training_thread is not None and self.training_thread.is_alive():
            status['status_msg'] = "Training in progress"
        else:
            status['status_msg'] = "No training in progress"

        status['remaining_time'] = self._get_redis_value("remaining_time")
        status['progress'] = self._get_redis_value("progress")

        loss_json_path = os.path.join(ROOT, TEMP_DIR, 'loss.json')
        status['loss'] = load_json(loss_json_path) if check_path(loss_json_path) else None

        return status
