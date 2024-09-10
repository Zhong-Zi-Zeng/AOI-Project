import os
import sys
import platform

import torch
import redis


class CheckStopTrainingHook:
    def __init__(self, work_dir: str):
        self.work_dir = work_dir

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

    def update(self, model):
        stop_training = bool(self.r.get("stop_training"))

        if not stop_training:
            return

        # Save last epoch
        last_weight_path = os.path.join(self.work_dir, "last.pt")
        torch.save(model.state_dict(), last_weight_path)
        sys.exit()
