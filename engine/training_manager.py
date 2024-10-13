import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import platform
import json
from threading import Thread

import numpy as np

from engine.redis_manager import RedisManager
from engine.general import (get_work_dir_path, copy_logfile_to_work_dir, clear_cache,
                            load_json, check_path, ROOT, TEMP_DIR, get_gpu_count)
from tools.evaluation import Evaluator


class TrainingManager:
    def __init__(self):
        self.training_thread_dict = {f"GPU:{device_id}": None for device_id in range(get_gpu_count())}

        self.redis = RedisManager()
        self._clear_status()

    def _train_wrapper(self, train_func, final_config, device_id):
        train_func()

        # ========= After Training =========
        copy_logfile_to_work_dir(final_config)
        self._clear_status(device_id)

    def _clear_status(self, device_id=None):
        if device_id is None:
            self.redis.clear()
            return

        for key in self.redis.DEFAULT_KEYS:
            self.redis.delete_key(f"GPU:{device_id}_{key}")

    def start_training(self, train_func, final_config):
        # ========= Before Training =========
        device_id = int(final_config['device'])
        self._clear_status(device_id)

        self.training_thread_dict[f"GPU:{device_id}"] = Thread(target=self._train_wrapper,
                                                               args=(train_func, final_config, device_id))
        self.training_thread_dict[f"GPU:{device_id}"].start()

    def start_eval(self, model, final_config):
        device_id = int(final_config['device'])
        self._clear_status(device_id)

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

    def stop_training(self, device_id: int) -> None:
        self.redis.set_value(f"GPU:{device_id}_stop_training", 1)

    def stop_eval(self, device_id: int) -> None:
        self.redis.set_value(f"GPU:{device_id}_stop_eval", 1)

    def update_status(self) -> None:
        for device_id in range(get_gpu_count()):
            training_thread = self.training_thread_dict[f"GPU:{device_id}"]
            if training_thread is not None and training_thread.is_alive():
                status_msg = f"GPU:{device_id} Training in progress"
            else:
                status_msg = f"GPU:{device_id} No training in progress"

            self.redis.set_value(f"GPU:{device_id}_status_msg", status_msg)

