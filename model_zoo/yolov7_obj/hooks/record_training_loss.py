import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.getcwd()))

from engine.general import ROOT, TEMP_DIR, save_json, load_json


class RecordTrainingLossHook:
    def update(self,
               iter: int,
               loss: float):

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
