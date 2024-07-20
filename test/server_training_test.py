import os
import webbrowser
import requests
import json
from pathlib import Path

URL = "http://localhost:5000/"


def training():
    # Step 1:
    # ========= 取得可使用模型的預設config檔 =========
    response = requests.get(URL + "get_template")
    model_dict = response.json()

    # Step 2:
    # ========= 上傳dataset =========
    # test_dataset = r"D:\AOI\0625Dataset\test"
    # foldername = os.path.basename(test_dataset)
    # files = []
    # data = {'paths[]': []}
    #
    # for root, dirs, filenames in os.walk(test_dataset):
    #     for filename in filenames:
    #         filepath = os.path.join(root, filename)
    #         relative_path = os.path.join(foldername, os.path.relpath(filepath, test_dataset))
    #         files.append(('files[]', (filename, open(filepath, 'rb'), 'application/octet-stream')))
    #         data['paths[]'].append(relative_path)
    #
    # requests.post(URL + "upload_dataset", files=files, data=data)

    # Step 3:
    # =========  選擇其中一個模型 =========
    config = model_dict['Cascade-Mask-RCNN-ResNet50']['config']

    # Step 4:
    # =========  更改超參數 =========
    config['batch_size'] = 1
    config['end_epoch'] = 10
    config['coco_root'] = "./data/WC-100"

    # Step 5:
    # =========  儲存自定義的config =========
    # json_data = {
    #     "config_name": "test",
    #     "config": json.dumps(config)
    # }
    # response = requests.post(URL + "save_config", data=json_data)

    # Step 6:
    # =========  訓練模型 =========
    webbrowser.open("http://localhost:1000/")

    json_data = {
        "config": json.dumps(config),
        "work_dir_name": "Test"  # 可選擇專案名稱, 不提供則使用模型名稱作為專案名稱
    }
    response = requests.post(URL + "train", data=json_data)
    print(response.json())


if __name__ == '__main__':
    training()
