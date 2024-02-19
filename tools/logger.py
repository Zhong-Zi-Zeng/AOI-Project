import os
import pandas as pd
import random
import json
import seaborn as sns
import matplotlib.pyplot as plt
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
        紀錄於Excel
        每一列就是step,每一行就是傳入的key

        Args:
            filename(str): excel檔命名，建議用model名稱
            data_dict(dict): 此epoch要存的數據
            step(int): 第幾個epoch
    '''

    # 檢查路徑
    if not os.path.exists(os.path.join("logs", "excel")):
        os.makedirs(os.path.join("logs", "excel"))

    excel_path = os.path.join("logs", "excel", filename + '.xlsx')

    # 檢查 Excel 檔案是否存在
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path, index_col=0)
    else:
        df = pd.DataFrame(columns=data_dict.keys())

    # 在 DataFrame 中加入新數據
    df.loc[step] = data_dict

    # 寫入excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=True)


def excel_to_json(filename):
    """
    從 Excel 讀取數據後轉換為 JSON 格式並寫入文件
    """
    excel_path = os.path.join("logs", "excel", filename + '.xlsx')

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


def load_data_from_json(filename):
    '''
        讀取 JSON 文件中的所有數據
    '''
    json_path = os.path.join("logs", "json", filename + '.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found!")
        return None

    return json_data


def plot_curve(filename):
    '''
        依據 JSON 文件中的所有key，自動繪製曲線
    '''
    json_data = load_data_from_json(filename)
    if json_data is None:
        return

    sns.set(style="darkgrid")  # Set seaborn style

    # 一種key存一張曲線圖
    for key in json_data[0].keys():  # 取得所有key(第一列)
        if key == "step":
            continue

        steps = []
        values = []

        plt.figure(figsize=(10, 6))
        plt.xlabel('Epoch')
        plt.ylabel(key.capitalize())
        plt.title(f'Curve for {key.capitalize()} in {filename}')

        for data in json_data:
            steps.append(data["step"])
            values.append(data.get(key))

        # Line plot
        plt.plot(steps, values, label=f'{filename} {key.capitalize()}')
        # Mark the best epoch
        best_epoch_index = values.index(max(values))
        best_epoch_value = max(values)
        plt.scatter(steps[best_epoch_index], best_epoch_value, color='red', marker='*',
                    label=f'Best Epoch {key.capitalize()}')

        plt.legend()
        plt.savefig(os.path.join("logs", "static", f"{filename}_{key}.png"))
        plt.close()


if __name__ == "__main__":
    for i in range(100):
        data = {
            'LOSS': random.random(),
            'Accuracy': random.random()
        }
        save_to_excel('Test', data, step=i)

    excel_to_json('Test')

    plot_curve('Test')

    subprocess.run(["python", "./logs/build_web_app.py"])     # curve架上網頁
