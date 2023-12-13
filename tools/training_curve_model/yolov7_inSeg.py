from __future__ import annotations
from .baseModel import BaseModel
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class yolov7_inSeg(BaseModel):
    def __init__(self, result_path: str, output_path: str):
        super().__init__(result_path, output_path)
        self.result_path = result_path
        self.output_path = output_path

        # Generate a folder to store the results of training curve
        os.makedirs(os.path.join(self.output_path, 'yolov7_inSeg_train_curve'), exist_ok=True)


    def generate_curve(self):
        # Read the result file
        try:
            df = pd.read_csv(os.path.join(self.result_path, 'results.csv'), skipinitialspace=True)
            # print(df.head())
        except FileNotFoundError:
            print(f"File not found!")

        # generate_curve
        curve_configs = [
            ('all_train', 'Training Loss', 'train'),            # All training loss
            ('all_val', 'Validation Loss', 'val'),              # All validation loss
            ('box_mAP0.5', 'Box mAP0.5', 'metrics', '(B)'),        # Box mAP0.5
            ('mask__mAP0.5', 'Mask mAP0.5', 'metrics', '(M)'),     # Mask mAP0.5
        ]

        for config in curve_configs:
            self.plot_and_save_curve(df, *config)


    def plot_and_save_curve(self, df, output_filename, title_prefix, prefix, suffix=None):
        sns.set(style="darkgrid")  # Set seaborn style
        plt.figure(figsize=(10, 6))

        if prefix == 'metrics':
            y_label = f'{prefix}/mAP_0.5{suffix}'
            label = f'{title_prefix}'

            # Mark the best epoch
            best_epoch = df.loc[df[y_label].idxmax()]['epoch']
            best_value = df.loc[df[y_label].idxmax()][y_label]

        else:  # train and val
            combined_loss = df[f'{prefix}/box_loss'] + df[f'{prefix}/seg_loss'] + df[f'{prefix}/obj_loss'] + df[f'{prefix}/cls_loss']
            y_label = combined_loss
            label = f'{title_prefix}'

            # Mark the best epoch
            best_epoch = df.loc[y_label.idxmin()]['epoch']
            best_value = y_label.min()

        # Line plot
        sns.lineplot(x='epoch', y=y_label, data=df.iloc[:-1] if prefix != 'val' else df, label=label)

        # Mark the best epoch
        plt.scatter(best_epoch, best_value, color='red', marker='*', label=f'Best Epoch ({int(best_epoch)})')

        plt.xlabel('Epoch')
        plt.ylabel(f'{title_prefix}')
        plt.title(f'{title_prefix} Curve')
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.output_path, 'yolov7_inSeg_train_curve'), f'{output_filename}.png'))
        plt.clf()