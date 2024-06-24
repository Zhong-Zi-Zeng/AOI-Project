import time
from typing import Optional, Union

import redis
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from torch.utils.tensorboard import SummaryWriter

@HOOKS.register_module()
class RemainingTimeHook(Hook):
    def __init__(self):
        self.start_time = None
        self.iter_times = []
        self.r = redis.Redis(host='redis', port=6379, db=0)

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
        self.r.set("remaining_time", remaining_time / 60)
        self.start_time = time.time()