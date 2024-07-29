import os

import requests
import cv2
import base64
import json
import numpy as np
from io import BytesIO

URL = "http://localhost:5000/"


def predict():
    # Step 1:
    # ========= 取得當前已訓練好的模型 =========
    response = requests.get(URL + "get_model_list")
    model_dict = response.json()

    # Step 2:
    # ========= 選擇其中一個模型 =========
    model_name = list(model_dict.keys())[0]

    final_config = model_dict[model_name]['final_config']
    weight_list = model_dict[model_name]['weight_list']

    final_config['weight'] = weight_list[-1]

    # Step 3:
    # ========= 初始化模型 (如果後續沒有要改變模型或是weight，則不需要呼叫) =========
    json_data = {"final_config": json.dumps(final_config)}
    requests.post(URL + "initialize_model", data=json_data)

    # Step 4:
    # ========= 讀取圖片並轉為二進位後進行預測 =========
    image = cv2.imread(r"D:\Heng_shared\AOI-Project\data\Synth-6000\val\0.jpg")
    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = BytesIO(buffer.tobytes())

    image_data = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(URL + 'predict', files=image_data)

    # Step 5:
    # ========= 解析預測結果 =========
    data = response.json()
    img_data = base64.b64decode(data['result_image'])
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 縮小 5 倍
    img = cv2.resize(img, (int(img.shape[1] / 2.3), int(img.shape[0] / 2.3)))

    print("Class List:", data['class_list'])
    print("Score List:", data['score_list'])
    print("BBox List:", data['bbox_list'])
    print("RLE List:", data['rle_list'])

    cv2.imshow("Result Image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    predict()
