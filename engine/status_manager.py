import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import platform
import json

import redis
import numpy as np

from training_manager import TrainingManager


class StatusManager:
    def __init__(self, training_manager: TrainingManager):
        self.training_manager = training_manager

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

    def _get_redis_value(self, key: str):
        value = self.r.get(key)
        if value is None:
            return None
        try:
            return np.around(float(value), 2)
        except ValueError:
            return value.decode('utf-8')  # 如果不能轉換為float，就解碼為字串

    def get_status(self) -> dict:
        self.training_manager.update_status()

        status = dict()

        status['status_msg'] = self._get_redis_value("status_msg")
        status['remaining_time'] = self._get_redis_value("remaining_time")
        status['progress'] = self._get_redis_value("progress")
        status['eval_progress'] = self._get_redis_value("eval_progress")
        status['loss'] = json.loads(self._get_redis_value("loss"))

        return status
