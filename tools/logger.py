import tensorflow as tf
import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from logs.build_web_app import app

# def log_write(tag, data_dict, step):
#     writer = tf.summary.create_file_writer("logs")
#     with writer.as_default():
#         for data_name, data_value in data_dict.items():
#             tf.summary.scalar(f"{tag}/{data_name}", float(data_value), step=step)
#         writer.flush()
#     '''
#     # 看tensorboard
#     tensorboard --logdir=D:\YA_share\AOI-Project\tools\logs
#     '''

def save_to_json(filename, tag, data_dict, step):
    # 處理路徑
    if not filename.endswith(".json"):
        filename += ".json"
    file_path = os.path.join("logs", filename)
    # 處理資料
    log_data = {
        'Epoch': step,
        tag: data_dict,
    }

    # 讀取檔案內容
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            all_data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        all_data = []

    # 將相同step的資料放一起
    existing_step_idx = next((idx for idx, entry in enumerate(all_data) if entry['step'] == step), None)
    if existing_step_idx is not None:
        all_data[existing_step_idx][tag] = data_dict
    else:
        all_data.append(log_data)

    # 寫入檔案
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(all_data, file, ensure_ascii=False, indent=4)


def read_file_contents(filename, tag):
    if not filename.endswith(".json"):
        filename += ".json"
    file_path = os.path.join("logs", filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            log_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found!")

    if log_data is not None:
        steps = []  # 待調整
        tag_data = []
        for data in log_data:
            steps.append(data.get("step"))
            tag_data.append(data.get(tag))

    return steps, tag_data

def training_curve(filename, tag):
    epoch, tag_data = read_file_contents(filename, tag)

    sns.set(style="darkgrid")  # Set seaborn style
    plt.figure(figsize=(10, 6))

    # Line plot
    plt.plot(epoch, tag_data, label=tag)
    # Mark the best epoch
    best_epoch_index = tag_data.index(max(tag_data))
    best_epoch_value = max(tag_data)
    plt.scatter(epoch[best_epoch_index], best_epoch_value, color='red', marker='*', label=f'Best Epoch')

    plt.xlabel('Epoch')
    plt.ylabel(tag)
    plt.title(f'{tag} Curve')
    plt.legend()
    plt.savefig(os.path.join("logs", "static", f"{tag}.png"))


if __name__ == "__main__":
    # example
    data_dict = {
        'loss': 0.1,
        'accuracy': 0.9
    }
    # save_to_json("YOLO", "training", data_dict, 1)

    app.run(debug=True)     # 架上網頁


