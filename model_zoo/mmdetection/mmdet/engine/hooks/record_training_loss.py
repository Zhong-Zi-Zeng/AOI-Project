import os
import sys

sys.path.append(os.path.join(os.getcwd()))

from typing import Optional
from datetime import datetime

from mmengine.hooks import Hook
from mmdet.registry import HOOKS

from engine.general import (save_json, TEMP_DIR, ROOT, load_json)

@HOOKS.register_module()
class RecordTrainingLossHook(Hook):

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs: Optional[dict] = None) -> None:

        loss = outputs['loss'].item()
        iter = runner.iter
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {'iter': iter, 'loss': loss, 'time': current_time}

        # If iter == 0, create loss.json
        # Else, append data to loss.json
        if iter == 0:
            save_json(os.path.join(ROOT, TEMP_DIR, 'loss.json'),
                      [data])
            return

        elif iter % 5 == 0:
            history_data = load_json(os.path.join(ROOT, TEMP_DIR, 'loss.json'))
            history_data.append(data)
            save_json(os.path.join(ROOT, TEMP_DIR, 'loss.json'),
                      history_data)

