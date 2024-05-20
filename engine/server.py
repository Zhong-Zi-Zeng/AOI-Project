import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import cv2
import json
import base64

from flask import Flask, request, jsonify

from engine.general import get_works_dir_path, load_yaml, allowed_file, convert_image_to_numpy
from model_manager import ModelManager

app = Flask(__name__)


@app.route('/get_model_list', methods=['GET'])
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
    model_list = {model_name: {'weight_list': [], 'final_config': None}
                  for model_name in os.listdir(train_dir_path)}

    for model_name in model_list:
        # 取出每個model可用的weight
        model_list[model_name]['weight_list'] = [os.path.join(train_dir_path, model_name, weight) for weight in
                                                 os.listdir(os.path.join(train_dir_path, model_name)) if
                                                 weight.endswith(".pth") or weight.endswith(".pt")]

        # 取出final_config.yaml的所有設定值
        model_list[model_name]['final_config'] = load_yaml(
            os.path.join(train_dir_path, model_name, "final_config.yaml"))

    return jsonify(model_list), 200


@app.route('/initialize_model', methods=['POST'])
def initialize_model():
    """
    給定final_config，初始化model
    """
    if 'final_config' not in request.form:
        return jsonify({"message": "final_config is required"}), 400

    final_config = request.form.get('final_config')
    final_config = json.loads(final_config)

    model_manager.initialize_model(final_config)
    return jsonify({"message": "success"}), 200


@app.route('/predict', methods=['POST'])
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
    app.run(debug=True, host='0.0.0.0', port=5000)
