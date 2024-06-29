import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import cv2
import json
import base64
import subprocess
from threading import Thread

from flask import Flask, request, jsonify

from engine.general import (get_work_dir_path, get_works_dir_path, load_yaml,
                            allowed_file, convert_image_to_numpy)
from model_manager import ModelManager
from training_manager import TrainingManager

APP = Flask(__name__)


# =======================================
# =============For Training==============
# =======================================
@APP.route('/get_dataset_list', methods=['GET'])
def get_dataset_list():
    """
    返回在./data資料夾中可使用的Dataset
    :return:
        ['WC-500', 'WC-1500', ...]
    """

    DATASET_DIR = "./data"

    dataset_list = [os.path.join(DATASET_DIR, dataset_name)
                    for dataset_name in os.listdir(DATASET_DIR)
                    if os.path.isdir(os.path.join(DATASET_DIR, dataset_name)) and
                    dataset_name != 'yoloBbox' and
                    dataset_name != 'yoloSeg'
                    ]

    return jsonify(dataset_list), 200


@APP.route('/get_template', methods=['GET'])
def get_template():
    """
    取得所有模型預設的config
    return:
        {
            "model_name":
                {
                    "config": 每個model預設的config
                }
        }
    """

    TEMPLATE_DIR = "./configs"

    model_dict = {model_name.replace(".yaml", ""): {'config': None}
                  for model_name in os.listdir(TEMPLATE_DIR) if model_name.endswith(".yaml")
                  }

    for model_name in model_dict:
        # 取出每個model預設的config
        model_dict[model_name]['config'] = load_yaml(os.path.join(TEMPLATE_DIR, model_name + ".yaml"))

    return jsonify(model_dict), 200


@APP.route('/get_status', methods=['GET'])
def get_status():
    """
    回傳目前模行訓練狀態
    return:
        {
            'status_msg': 狀態訊息
            'remaining_time': 剩餘時間
            'progress': 進度
        }
    """
    status = training_manager.get_status()

    return jsonify(status), 200

@APP.route('/stop_training', methods=['GET'])
def stop_training():
    training_manager.stop_training()

    return jsonify({"message": "success"}), 200

@APP.route('/train', methods=['POST'])
def train():
    """
    給定config後執行training
    """

    def open_tensorboard(final_config):
        subprocess.run(['tensorboard',
                        '--logdir', get_work_dir_path(final_config),
                        '--host', '0.0.0.0',
                        '--port', '1000'])

    def train_model():
        model.train()

    if 'config' not in request.form:
        return jsonify({"message": "config is required"}), 400

    config = request.form.get('config')
    config = json.loads(config)

    final_config = model_manager.initialize_model(config, task='train')
    model = model_manager.get_model()

    # Open tensorboard
    Thread(target=open_tensorboard, args=[final_config]).start()

    # Training
    training_manager.start_training(train_model)

    return jsonify({"message": "success"}), 200


# =======================================
# =============For Inference=============
# =======================================
@APP.route('/get_model_list', methods=['GET'])
def get_model_list():
    """
    搜尋當前work_dirs下，已training好的model所有的weight檔名稱與final_config.yaml

    return:
        {
            "model_name":
                {
                    "weight_list": [weight1, weight2, ...],
                    "final_config": final_config
                }
        }
    """
    train_dir_path = os.path.join(get_works_dir_path(), "train")
    model_dict = {model_name: {'weight_list': [], 'final_config': None}
                  for model_name in os.listdir(train_dir_path)}

    for model_name in model_dict:
        # 取出每個model可用的weight
        model_dict[model_name]['weight_list'] = [os.path.join(train_dir_path, model_name, weight) for weight in
                                                 os.listdir(os.path.join(train_dir_path, model_name)) if
                                                 weight.endswith(".pth") or weight.endswith(".pt")]

        # 取出final_config.yaml的所有設定值
        model_dict[model_name]['final_config'] = load_yaml(
            os.path.join(train_dir_path, model_name, "final_config.yaml"))

    return jsonify(model_dict), 200


@APP.route('/initialize_model', methods=['POST'])
def initialize_model():
    """
    給定final_config，初始化model，用於predict之前
    """
    if 'final_config' not in request.form:
        return jsonify({"message": "final_config is required"}), 400

    final_config = request.form.get('final_config')
    final_config = json.loads(final_config)

    model_manager.initialize_model(final_config, task='predict')
    return jsonify({"message": "success"}), 200


@APP.route('/predict', methods=['POST'])
def predict():
    """
    給定圖片進行預測，並回傳結果

    Returns:
        {
            "result_image": Base64的編碼圖片
            "class_list" (list[int]): (M, ) 檢測到的類別編號，M為檢測到的物體數量
            "score_list" (list[float]): (M, ) 每個物體的信心值
            "bbox_list" (list[int]): (M, 4) 物體的bbox, 需為 x, y, w, h
            "rle_list" (list[dict["size":list, "counts": str]]): (M, ) RLE編碼格式
        }
    """
    # 检查是否包含圖片
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    # 取出image
    image_file = request.files['image']
    if allowed_file(image_file.filename):
        image = convert_image_to_numpy(image_file)
    else:
        return jsonify({"error": "Unsupported image type"}), 400

    # 確認是否已經初始化model
    model = model_manager.get_model()
    if model is None:
        return jsonify({"error": "Model is not initialized. Please call /initialize_model first."}), 400

    # 預測
    result = model.predict(image)
    _, buffer = cv2.imencode('.jpg', result['result_image'])
    jpg_as_text = base64.b64encode(buffer).decode()
    data = {
        'result_image': jpg_as_text,
        'class_list': result.get('class_list'),
        'score_list': result.get('score_list'),
        'bbox_list': result.get('bbox_list'),
        'rle_list': result.get('rle_list'),
    }

    return jsonify(data), 200


if __name__ == '__main__':
    model_manager = ModelManager()
    training_manager = TrainingManager()
    APP.run(debug=False, host='0.0.0.0', port=5000)
