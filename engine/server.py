import os
import sys

sys.path.append(os.path.join(os.getcwd()))
import cv2
import json
import base64
import numpy as np

from io import BytesIO
from flask import Flask, request, jsonify

from engine.builder import Builder
from engine.general import get_works_dir_path, load_yaml

app = Flask(__name__)
UPLOAD_FOLDER = '/path/to/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_image_to_numpy(image) -> np.ndarray:
    in_memory_file = BytesIO()
    image.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return image


@app.route('/model_list', methods=['GET'])
def get_model_list():
    """
    搜尋當前work_dirs下，已training好的model所有的weight檔名稱與final_config.yaml

    return:
        {
            model_name:
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


@app.route('/predict', methods=['POST'])
def predict():
    """
    給定圖片與final_config，對圖片進行預測，並回傳結果
    :return:
    """
    global model

    # 检查是否有圖片
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    # 取出image
    image_file = request.files['image']
    if allowed_file(image_file.filename):
        image = convert_image_to_numpy(image_file)
    else:
        return jsonify({"error": "Unsupported image type"}), 400

    # 檢查是否有final_config
    if 'final_config' not in request.form:
        return jsonify({"error": "No final_config part"}), 400

    # 取出final_config
    final_config = request.form.get('final_config')
    final_config = json.loads(final_config)

    # 建立模型
    if model is None:
        builder = Builder(yaml_dict=final_config, task='predict')
        cfg = builder.build_config()
        model = builder.build_model(cfg)

    result = model.predict(image)
    _, buffer = cv2.imencode('.jpg', result['result_image'])
    jpg_as_text = base64.b64encode(buffer).decode()
    data = {
        'result_image': jpg_as_text,
        'class_list': result['class_list'],
        'score_list': result['score_list'],
        'bbox_list': result['bbox_list'],
    }

    return jsonify(data), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
