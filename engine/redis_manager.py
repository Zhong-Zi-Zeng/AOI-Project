from typing import Optional, Union
import platform

import redis
import numpy as np


class RedisManager:
    DEFAULT_KEYS = [
        "status_msg",
        "remaining_time",
        "progress",
        "eval_progress",
        "loss",
        "stop_training",
        "stop_eval"
    ]

    def __init__(self):
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

    def ping(self):
        self.r.ping()
    def clear(self):
        self.r.flushall()

    def delete_key(self, key: str):
        self.r.delete(key)

    def set_value(self, key: str, value: Union[int, float, str]) -> None:
        self.r.set(key, value)

    def get_value(self, key: str):
        value = self.r.get(key)
        if value is None:
            return None
        try:
            return np.around(float(value), 2)
        except ValueError:
            return value.decode('utf-8')  # 如果不能轉換為float，就解碼為字串
