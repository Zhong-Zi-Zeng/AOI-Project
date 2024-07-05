import os
import sys

import torch
import redis


class CheckStopTrainingHook:
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.r = redis.Redis(host='redis', port=6379, db=0)

    def update(self, model):
        stop_training = bool(self.r.get("stop_training"))

        if not stop_training:
            return

        # Save last epoch
        last_weight_path = os.path.join(self.work_dir, "last.pt")
        torch.save(model.state_dict(), last_weight_path)
        sys.exit()
