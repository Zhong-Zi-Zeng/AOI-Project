import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.getcwd()))

from engine.general import (ROOT, TEMP_DIR, save_json, load_json, check_path)


class RecordTrainingLossHook:
    def update(self,
               iter: int,
               loss: float):

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        loss_json_path = os.path.join(ROOT, TEMP_DIR, 'loss.json')
        data = {'iter': iter, 'loss': loss, 'time': current_time}

        # If iter == 0, create loss.json
        # Else, append data to loss.json
        if iter == 0 or \
                not check_path(loss_json_path):
            save_json(loss_json_path, [data])

        elif iter % 5 == 0:
            history_data = load_json(loss_json_path)
            if isinstance(history_data, list):
                history_data.append(data)
                save_json(loss_json_path, history_data)
            else:
                save_json(loss_json_path, [data])
