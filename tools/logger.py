import json
import os
import random
import time
import multiprocessing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


class Logger:
    '''
        使用者只需呼叫 add_scalar()
        ---------------------------------
        1. 資料存入 Excel --> add_scalar()
        2. Excel 轉 Json --> _excel_to_json()
        3. 讀取 Json，畫曲線圖 --> _plot_curve()
        4. 看網頁時使用 --> subprocess.run(["python", "./logs/build_web_app.py"])
    '''

    def __init__(self):
        if not os.path.exists(os.path.join("tools", "logs", "excel")):
            os.makedirs(os.path.join("tools", "logs", "excel"))

    def _load_json(self, filename: str):  # tools
        '''
            讀取 JSON 文件中的所有數據
        '''
        json_path = os.path.join("tools", "logs", "json", filename + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
        except FileNotFoundError:
            print(f"File not found!")
            return None

        return json_data


    def add_scalar(self, filename: str, data: dict, step: int):  # 使用者只需呼叫add_scalar
        '''
            記錄於Excel
            每一列就是step,每一行就是傳入的key

            Args:
                filename(str): excel檔命名，建議用model名稱
                data(dict): 此epoch要存的數據
                step(int): 第幾個epoch
        '''

        excel_path = os.path.join("tools", "logs", "excel", filename + '.xlsx')
        # 檢查 Excel 檔案是否存在
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path, index_col=0)
        else:
            df = pd.DataFrame(columns=data.keys())

        # 檢查 step 是否存在
        if step in df.index:  # 將新的 data 加入現有的行中
            for key, value in data.items():
                df.loc[step, key] = value
        else:  # 創建一個新的行
            df.loc[step] = data

        # 寫入excel
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=True)

        # ========== Next step ===========
        # self._excel_to_json(filename)


    def _excel_to_json(self, filename: str):
        """
        從 Excel 讀取數據後轉換為 JSON 格式並寫入文件
        """
        excel_path = os.path.join("tools", "logs", "excel", filename + '.xlsx')
        if not os.path.exists(excel_path):
            raise ValueError(f"Error: Excel file {excel_path} does not exist.")

        df = pd.read_excel(excel_path, index_col=0)
        json_file = os.path.join("tools", "logs", "json", filename + '.json')
        df.to_json(json_file, orient="records", force_ascii=False)

        # ========== Next step ===========
        # self._plot_curve(filename)



    def _plot_curve(self, filename: str):
        '''
            依據 JSON 文件中的所有key，自動繪製曲線
        '''
        json_data = self._load_json(filename)
        if json_data is None:
            return

        sns.set(style="darkgrid")  # Set seaborn style

        # 存所有key的資料，一種key存一張曲線圖
        all_data = []

        for i, data in enumerate(json_data):  # 每個epoch的資料
            for key, value in data.items():
                if isinstance(value, (int, float)):  # 只存有數值的
                    all_data.append({"step": i, "key": key, "value": value})

        all_data_df = pd.DataFrame(all_data)

        # 繪製曲線
        for key, data in all_data_df.groupby("key"):
            plt.figure(figsize=(10, 6))

            # Line plot
            plt.plot(data["step"], data["value"], label=f'{key}')

            # Mark the best epoch
            best_epoch_index = data["value"].idxmax()
            best_epoch_step = data.loc[best_epoch_index, "step"]
            best_epoch_value = data.loc[best_epoch_index, "value"]
            plt.scatter(best_epoch_step, best_epoch_value, color='red', marker='*', label=f'best epoch {key}')

            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.title(f'Curve for {key} in {filename}')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join("tools", "logs", "static", f"{filename}_{key}.png"))  # 曲線圖名稱
            plt.close()



# example
# logger = Logger()
#
# for i in range(10):
#     if i < 5:
#         for tag in ['loss', 'acc', 'map', 'ap']:
#             data = {
#                 tag: random.random(),
#             }
#             logger.add_scalar('Test', data, i)
#     else:
#         for tag in ['loss', 'acc']:
#             data = {
#                 tag: random.random(),
#             }
#             logger.add_scalar('Test', data, i)
#
# logger._excel_to_json('Test')
# logger._plot_curve('Test')
