import time
import os
from typing import Optional

from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from engine.redis_manager import RedisManager
from engine.general import load_yaml


@HOOKS.register_module()
class RemainingTimeHook(Hook):
    def __init__(self):
        self.start_time = None
        self.iter_times = []

        self.redis = RedisManager()

    def before_train(self, runner) -> None:
        self.start_time = time.time()
        self.iter_times = []

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:
        if not hasattr(self, "final_config"):
            setattr(self, "final_config",
                    load_yaml(os.path.join(runner.work_dir.replace(" ", ""), 'final_config.yaml')))

        iter_time = time.time() - self.start_time
        self.iter_times.append(iter_time)

        avg_iter_time = sum(self.iter_times) / len(self.iter_times)

        total_iters = runner.max_iters
        remaining_iters = total_iters - runner.iter

        remaining_time = avg_iter_time * remaining_iters
        progress = ((runner.iter + 1) / total_iters) * 100

        device_id = int(self.final_config['device'])
        self.redis.set_value(f"GPU:{device_id}_remaining_time", remaining_time / 60)
        self.redis.set_value(f"GPU:{device_id}_progress", progress)

        self.start_time = time.time()
