from threading import Thread

import numpy as np
import redis


class TrainingManager:
    def __init__(self):
        self.thread = None
        self.complete = False
        self.r = redis.Redis(host='redis', port=6379, db=0)

    def _train_wrapper(self, train_func):
        train_func()
        self.complete = True

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

    def start_training(self, train_func):
        self._clear_redis_key()
        self.complete = False
        self.thread = Thread(target=self._train_wrapper, args=(train_func,))
        self.thread.start()

    def stop_training(self):
        self.r.set("stop_training", 1)

    def get_status(self) -> dict:
        status = {}

        if self.thread is not None and self.thread.is_alive():
            status['status_msg'] = "Training in progress"
        elif self.complete:
            status['status_msg'] = "Training completed"
        else:
            status['status_msg'] = "No training in progress"

        status['remaining_time'] = self._get_redis_value("remaining_time")
        status['progress'] = self._get_redis_value("progress")

        return status
