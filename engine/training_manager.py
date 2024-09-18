import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import platform
import json
from threading import Thread

import numpy as np
import redis

from engine.general import (get_work_dir_path, copy_logfile_to_work_dir, clear_cache,
                            load_json, check_path, ROOT, TEMP_DIR)
from tools.evaluation import Evaluator


class TrainingManager:
    def __init__(self):
        self.training_thread = None

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
        self._clear_data()

    def _train_wrapper(self, train_func, final_config):
        train_func()

        # ========= After Training =========
        copy_logfile_to_work_dir(final_config)
        self._clear_data()

    def _clear_data(self):
        self.r.flushdb()
        clear_cache()

    def start_training(self, train_func, final_config):
        # ========= Before Training =========
        self._clear_data()

        self.training_thread = Thread(target=self._train_wrapper, args=(train_func, final_config))
        self.training_thread.start()

    def start_eval(self, model, final_config):
        self._clear_data()

        final_config['conf_thres'] = 0.3
        evaluator = Evaluator(model=model, cfg=final_config)
        evaluator.eval()

        detected_json = evaluator.detected_json
        if detected_json is not None:
            confidences = np.arange(0.3, 1.0, 0.1)
            result = {}

            for conf in confidences:
                final_config['conf_thres'] = conf
                evaluator = Evaluator(model=model, cfg=final_config, detected_json=detected_json)
                recall_and_fpr_for_all, fpr_image_name, undetected_image_name = evaluator.eval(return_detail=True)

                result[conf] = {
                    'recall_for_image': recall_and_fpr_for_all[0],
                    'fpr_for_image': recall_and_fpr_for_all[1],
                    'recall_for_defect': int(recall_and_fpr_for_all[2]),
                    'fpr_for_defect': int(recall_and_fpr_for_all[3]),
                    'fpr_image_name': fpr_image_name,
                    'undetected_image_name': undetected_image_name
                }
            return result
        return None

    def stop_training(self) -> None:
        self.r.set("stop_training", 1)

    def stop_eval(self) -> None:
        self.r.set("stop_eval", 1)

    def update_status(self) -> None:
        if self.training_thread is not None and self.training_thread.is_alive():
            status_msg = "Training in progress"
        else:
            status_msg = "No training in progress"
        self.r.set('status_msg', status_msg)

        loss_json_path = os.path.join(ROOT, TEMP_DIR, 'loss.json')
        loss = load_json(loss_json_path) if check_path(loss_json_path) else None
        loss_str = json.dumps(loss)
        self.r.set('loss', loss_str)
