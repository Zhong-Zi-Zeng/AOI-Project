import sys
import os
from typing import Optional

import redis
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class CheckStopTrainingHook(Hook):
    def __init__(self):
        self.r = redis.Redis(host='redis', port=6379, db=0)

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

