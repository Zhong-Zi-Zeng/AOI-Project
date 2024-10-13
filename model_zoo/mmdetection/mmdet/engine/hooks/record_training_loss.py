import os
import sys
import json

sys.path.append(os.path.join(os.getcwd()))

from typing import Optional
from datetime import datetime

from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from engine.redis_manager import RedisManager
from engine.general import load_yaml


@HOOKS.register_module()
class RecordTrainingLossHook(Hook):
    def __init__(self):
        self.loss_list = []
        self.redis = RedisManager()

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:
        if not hasattr(self, "final_config"):
            setattr(self, "final_config",
                    load_yaml(os.path.join(runner.work_dir.replace(" ", ""), 'final_config.yaml')))

        loss = outputs['loss'].item()
        iter = runner.iter

        if iter % 5 != 0:
            return

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {'iter': iter, 'loss': loss, 'time': current_time}
        self.loss_list.append(data)

        device_id = int(self.final_config['device'])
        loss_list_json = json.dumps(self.loss_list)
        self.redis.set_value(f"GPU:{device_id}_loss", loss_list_json)
