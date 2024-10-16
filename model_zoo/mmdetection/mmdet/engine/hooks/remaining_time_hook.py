import time
import platform
from typing import Optional, Union

import redis
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class RemainingTimeHook(Hook):
    def __init__(self):
        self.start_time = None
        self.iter_times = []

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

    def before_train(self, runner) -> None:
        self.start_time = time.time()
        self.iter_times = []

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:

        iter_time = time.time() - self.start_time
        self.iter_times.append(iter_time)

        avg_iter_time = sum(self.iter_times) / len(self.iter_times)

        total_iters = runner.max_iters
        remaining_iters = total_iters - runner.iter

        remaining_time = avg_iter_time * remaining_iters
        progress = ((runner.iter + 1) / total_iters) * 100

        self.r.set("remaining_time", remaining_time / 60)
        self.r.set("progress", progress)

        self.start_time = time.time()