import os
import sys
import json

sys.path.append(os.path.join(os.getcwd()))

from training_manager import TrainingManager
from engine.general import get_gpu_count
from engine.redis_manager import RedisManager


class StatusManager:
    def __init__(self, training_manager: TrainingManager):
        self.training_manager = training_manager
        self.redis = RedisManager()

    def get_status(self) -> dict:
        self.training_manager.update_status()

        status = {f"GPU:{device_id}": dict()
                  for device_id in range(get_gpu_count())}

        for device_id in range(get_gpu_count()):
            for key in self.redis.DEFAULT_KEYS:
                value = self.redis.get_value(f"GPU:{device_id}_{key}")
                if key == "loss":
                    status[f"GPU:{device_id}"][key] = None if value is None else json.loads(value)
                else:
                    status[f"GPU:{device_id}"][key] = value

        return status
