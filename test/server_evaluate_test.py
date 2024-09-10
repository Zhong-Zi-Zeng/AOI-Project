import os
import webbrowser
import requests
import json
from pathlib import Path

URL = "http://localhost:5000/"


def evaluate():
    # Step 1:
    # ========= 取得可使用模型的預設config檔 =========
    response = requests.get(URL + "get_model_list")
    model_dict = response.json()

    # Step 2:
    # =========  選擇其中一個模型 =========
    model_name = 'Cascade-Mask-RCNN_wc_500'
    final_config = model_dict[model_name]['final_config']
    weight_list = model_dict[model_name]['weight_list']

    # Step 3:
    # =========  更改weight和dataset =========
    final_config['weight'] = weight_list[-1]
    final_config['coco_root'] = "./data/WC-100"

    json_data = {
        "final_config": json.dumps(final_config),
        "work_dir_name": "Test"  # 可選擇專案名稱, 不提供則使用模型名稱作為專案名稱
    }
    response = requests.post(URL + "evaluate", data=json_data)
    print(response.json())


if __name__ == '__main__':
    evaluate()
