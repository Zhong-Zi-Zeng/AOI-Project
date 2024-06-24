import webbrowser
import requests
import json

URL = "http://localhost:5000/"


def training():
    # Step 1:
    # ========= 取得可使用模型的預設config檔 =========
    response = requests.get(URL + "get_template")
    model_dict = response.json()

    # Step 2:
    # =========  選擇其中一個模型 =========
    config = model_dict['Cascade-Mask-RCNN-ResNet50']['config']

    # Step 3:
    # =========  更改超參數 =========
    config['batch_size'] = 1
    config['end_epoch'] = 3
    config['coco_root'] = "./data/WC-100"

    # Step 4:
    # =========  訓練模型 =========
    webbrowser.open("http://localhost:1000/")

    json_data = {"config": json.dumps(config)}
    response = requests.post(URL + "train", data=json_data)
    print(response.json())

if __name__ == '__main__':
    training()

