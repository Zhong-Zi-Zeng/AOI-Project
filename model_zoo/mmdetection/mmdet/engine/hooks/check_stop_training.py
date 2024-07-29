import sys
import os
import platform
from typing import Optional

import redis
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class CheckStopTrainingHook(Hook):
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

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:

        stop_training = bool(self.r.get("stop_training"))

        if not stop_training:
            return

        # Save last epoch
        last_weight_path = os.path.join(runner.work_dir, "last.pt")
        torch.save(runner.model.state_dict(), last_weight_path)
        sys.exit()

