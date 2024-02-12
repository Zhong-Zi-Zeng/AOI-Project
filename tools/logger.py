import tensorflow as tf
import datetime
import random
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

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

def save_to_excel(filename, data_dict, step):
    '''
        紀錄於Excel:
            每一列就是step
            每一行就是傳入的key
    '''
    excel_file = filename + '.xlsx' if not filename.endswith('.xlsx') else filename
    excel_path = os.path.join("logs", "excel", excel_file)

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_file, index_col=0)
        df.loc[step] = data_dict
    else:
        df = pd.DataFrame(data_dict, index=[step])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=True)

def excel_to_json(filename):
    """
    從 Excel 讀取數據後轉換為 JSON 格式並寫入文件
    """
    excel_file = filename + '.xlsx' if not filename.endswith('.xlsx') else filename
    excel_path = os.path.join("logs", "excel", excel_file)

    if not os.path.exists(excel_path):
        print(f"Error: Excel file {excel_path} does not exist.")
        return

    # 讀取 Excel 檔案並轉換為 JSON 格式
    df = pd.read_excel(excel_path, index_col=0)
    json_data = []
    for index, row in df.iterrows():
        step_data = {'step': index}
        for key, value in row.items():
            if not pd.isnull(value):
                step_data[key] = value
        json_data.append(step_data)

    # 寫入 JSON 文件
    json_file = os.path.join("logs", "json", filename + '.json')
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

def load_data_from_json(filename, key):
    '''
        讀取 JSON 文件中指定的數據(key)
    '''
    json_file = filename + '.json' if not filename.endswith('.json') else filename
    json_path = os.path.join("logs", "json", json_file)
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found!")

    steps = []
    key_data = []
    for data in json_data:
        steps.append(data.get("step"))
        key_data.append(data.get(key))

    return steps, key_data

def plot_curve(filename, key):
    '''
        1.讀取 JSON 文件中指定的數據(key)
        2.依指定的key, 繪製曲線
    '''
    steps, key_data = load_data_from_json(filename, key)

    sns.set(style="darkgrid")  # Set seaborn style
    plt.figure(figsize=(10, 6))

    # Line plot
    plt.plot(steps, key_data, label=key.capitalize())
    # Mark the best epoch
    best_epoch_index = key_data.index(max(key_data))
    best_epoch_value = max(key_data)
    plt.scatter(steps[best_epoch_index], best_epoch_value, color='red', marker='*', label=f'Best Epoch')

    plt.xlabel('Epoch')
    plt.ylabel(key.capitalize())
    plt.title(f'{key.capitalize()} Curve')
    plt.legend()
    plt.savefig(os.path.join("logs", "static", f"{filename}_{key}.png"))


if __name__ == "__main__":
    # example
    data_dict = {
        'loss': 0.1,
        'accuracy': 0.8
    }
    # save_to_excel('YOLO', data_dict, step=0)
    # excel_to_json('YOLO')
    # plot_curve('YOLO_500', 'loss')

    # curve架上網頁
    # subprocess.run(["python", "./logs/build_web_app.py"])

