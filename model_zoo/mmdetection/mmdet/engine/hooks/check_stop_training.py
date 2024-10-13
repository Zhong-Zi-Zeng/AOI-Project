import sys
import os
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from engine.redis_manager import RedisManager
from engine.general import load_yaml

@HOOKS.register_module()
class CheckStopTrainingHook(Hook):
    def __init__(self):
        self.redis = RedisManager()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:
        if not hasattr(self, "final_config"):
            setattr(self, "final_config",
                    load_yaml(os.path.join(runner.work_dir.replace(" ", ""), 'final_config.yaml')))

        device_id = int(self.final_config['device'])
        stop_training = bool(self.redis.get_value(f"GPU:{device_id}_stop_training"))

        if not stop_training:
            return

        # Save last epoch
        last_weight_path = os.path.join(runner.work_dir, "last.pt")
        torch.save(runner.model.state_dict(), last_weight_path)
        sys.exit()

