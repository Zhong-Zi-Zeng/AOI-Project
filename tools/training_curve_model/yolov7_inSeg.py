from __future__ import annotations
from .baseModel import BaseModel
import os
import pandas as pd
import matplotlib.pyplot as plt


class yolov7_inSeg(BaseModel):
    def __init__(self, result_dir: str, output_dir: str):
        super().__init__(result_dir, output_dir)
        self.result_dir = result_dir
        self.output_dir = output_dir

        # Generate a folder to store the results of training curve
        os.makedirs(os.path.join(self.output_dir, 'yolov7_inSeg_train_curve'), exist_ok=True)

    def generate_curve(self):
        # Read the result file
        try:
            df = pd.read_csv(os.path.join(self.result_dir, 'results.csv'), skipinitialspace=True)
            # print(df.head())
        except FileNotFoundError:
            print(f"File not found!")

        # ===== generate_curve =====
        # All training loss curve
        self.plot_and_save_curve(df, 'all_train', 'Training Loss', 'train')

        # All validation loss curve
        self.plot_and_save_curve(df, 'all_val', 'Validation Loss', 'val')

        # Box precision, recall, and mAP_0.5 curves
        self.plot_and_save_curve(df, 'box_precision_recall_mAP0.5', 'Box Metrics Value', 'metrics', '(B)')

        # Mask precision, recall, and mAP_0.5 curves
        self.plot_and_save_curve(df, 'mask_precision_recall_mAP0.5', 'Mask Metrics Value', 'metrics', '(M)')

    def plot_and_save_curve(self, df, output_filename, title_prefix, prefix, suffix=None):

        if prefix == 'metrics':
            plt.plot(df['epoch'], df[f'{prefix}/precision{suffix}'], label='Precision')
            plt.plot(df['epoch'], df[f'{prefix}/recall{suffix}'], label='Recall')
            plt.plot(df['epoch'], df[f'{prefix}/mAP_0.5{suffix}'], label='mAP_0.5')
        else:   # train and val
            plt.plot(df['epoch'], df[f'{prefix}/box_loss'], label='Box Loss')
            plt.plot(df['epoch'], df[f'{prefix}/seg_loss'], label='Segmentation Loss')
            plt.plot(df['epoch'], df[f'{prefix}/obj_loss'], label='Object Loss')
            plt.plot(df['epoch'], df[f'{prefix}/cls_loss'], label='Class Loss')

        plt.xlabel('Epoch')
        plt.ylabel(f'{title_prefix}')
        plt.title(f'{title_prefix} Curve')
        plt.legend()
        plt.savefig(os.path.join(os.path.join(self.output_dir, 'yolov7_inSeg_train_curve'), f'{output_filename}.png'))
        plt.clf()