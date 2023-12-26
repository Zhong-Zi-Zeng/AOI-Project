from __future__ import annotations
from .baseModel import BaseModel
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class yolov7_Obj(BaseModel):
    def __init__(self, result_path: str, output_path: str):
        super().__init__(result_path, output_path)
        self.result_path = result_path
        self.output_path = output_path

    def generate_curve(self):
        # Read the result file(txt)
        try:
            file_path_txt = os.path.join(self.result_path, 'results.txt')
            with open(file_path_txt, 'r') as txt_file:
                content = [line.strip().split() for line in txt_file]

            # Add column name
            df = pd.DataFrame(content,
                              columns=['Epoch', 'gpu_mem', 'train/box_loss', 'train/obj_loss', 'train/cls_loss',
                                       'train/total_loss', 'labels', 'img_size', 'precision', 'recall',
                                       'metrics/mAP_0.5(B)', 'metrics/mAP_0.5:0.95(B)', 'val/box_loss', 'val/obj_loss',
                                       'val/cls_loss'])

            # Convert 'Epoch' column to numeric type
            df['Epoch'] = df['Epoch'].str.split('/').str[0]
            df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')

            # print(df.head())
        except FileNotFoundError:
            print(f"File not found!")

        # generate_curve
        curve_configs = [
            ('all_train', 'Training Loss', 'train'),  # All training loss
            ('all_val', 'Validation Loss', 'val'),  # All validation loss
            ('box_mAP0.5', 'Box mAP0.5', 'metrics', '(B)'),  # Box mAP0.5
            # ('mask__mAP0.5', 'Mask mAP0.5', 'metrics', '(M)'),     # Mask mAP0.5
        ]
        for config in curve_configs:
            self.plot_and_save_curve(df, *config)

    def plot_and_save_curve(self, df, output_filename, title_prefix, prefix, suffix=None):
        sns.set(style="darkgrid")  # Set seaborn style
        plt.figure(figsize=(10, 6))

        if prefix == 'metrics':
            y_label = df[f'{prefix}/mAP_0.5{suffix}'].astype(float)
            label = f'{title_prefix}'

            # Mark the best epoch
            best_epoch = y_label.idxmax()
            best_value = y_label.max()

        else:  # train and val
            box_loss = df[f'{prefix}/box_loss'].astype(float)
            obj_loss = df[f'{prefix}/obj_loss'].astype(float)
            cls_loss = df[f'{prefix}/cls_loss'].astype(float)
            y_label = box_loss + obj_loss + cls_loss
            label = f'{title_prefix}'

            # Mark the best epoch
            best_epoch = y_label.idxmin()
            best_value = y_label.min()

        # Line plot
        sns.lineplot(x='Epoch', y=y_label, data=df.iloc[:-1] if prefix != 'val' else df, label=label)

        # Mark the best epoch
        plt.scatter(best_epoch, best_value, color='red', marker='*', label=f'Best Epoch ({int(best_epoch)})')

        plt.xlabel('Epoch')
        plt.ylabel(f'{title_prefix}')
        plt.title(f'{title_prefix} Curve')
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.output_path), f'{output_filename}.png'))
        plt.clf()
