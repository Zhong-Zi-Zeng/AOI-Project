from threading import Thread

import numpy as np
import redis

class TrainingManager:
    def __init__(self):
        self.thread = None
        self.complete = False
        self.r = redis.Redis(host='redis', port=6379, db=0)

    def start_training(self, train_func):
        self.complete = False
        self.thread = Thread(target=self._train_wrapper, args=(train_func,))
        self.thread.start()

    def _train_wrapper(self, train_func):
        train_func()
        self.complete = True
        self._clear_redis_key()

    def _get_redis_value(self, key: str):
        value = self.r.get(key)
        if value is None:
            return None
        try:
            return np.around(float(value), 2)
        except ValueError:
            return value.decode('utf-8')  # 如果不能轉換為float，就解碼為字串

    def _clear_redis_key(self):
        self.r.delete("remaining_time")
        self.r.delete("progress")

    def get_remaining_time(self):
        return self._get_redis_value("remaining_time")

    def get_progress(self):
        return self._get_redis_value("progress")

    def is_training(self):
        return self.thread is not None and self.thread.is_alive()

    def is_complete(self):
        return self.complete
