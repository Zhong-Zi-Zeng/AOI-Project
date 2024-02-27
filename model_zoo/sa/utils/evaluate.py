from __future__ import annotations
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import yaml


class Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.threshold_list = [0.3, 0.5, 0.7]

        self.iou_history = []
        self.precision_history = []
        self.recall_history = []
        self.ap_history = []

    @staticmethod
    def _convert_to_numpy(input):
        if isinstance(input, torch.Tensor):
            return input.detach().cpu().numpy()
        return input

    @staticmethod
    def _get_ap(ground_truth, pred_label):
        """
            計算AP
        """
        nd = len(ground_truth)
        tp_list = [0] * nd
        tn_list = [0] * nd
        fn_list = [0] * nd
        fp_list = [0] * nd
        for i in range(len(ground_truth)):
            gt = ground_truth[i]
            pred = pred_label[i]

            if gt == 1 and pred == 1:
                tp_list[i] = 1

            elif gt == 1 and pred == 0:
                fn_list[i] = 1

            elif gt == 0 and pred == 1:
                fp_list[i] = 1

            else:
                tn_list[i] = 1

        tp_fn_len = np.sum(fn_list) + np.sum(tp_list)

        cumsum = 0
        for idx, val in enumerate(fp_list):
            fp_list[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp_list):
            tp_list[idx] += cumsum
            cumsum += val

        recall = tp_list[:]
        for idx, val in enumerate(tp_list):
            recall[idx] = float(tp_list[idx]) / (tp_fn_len + 1e-5)

        precision = tp_list[:]
        for idx, val in enumerate(tp_list):
            precision[idx] = float(tp_list[idx]) / (fp_list[idx] + tp_list[idx] + 1e-5)

        ap = np.trapz(precision, recall)
        return ap

    def reset(self):
        self.iou_history = []
        self.precision_history = []
        self.recall_history = []

    def add_batch(self,
                  target: torch.Tensor,
                  predict: torch.Tensor,
                  need_sigmoid=True):
        """
            將每次傳入的batch進行評估後，並將數值存在history中

            Args:
                target: (B, C, H, W)
                predict: (B, C, H, W) 還未經過sigmoid
                need_sigmoid: 是否需要將predict進行sigmoid
        """
        assert target.shape == predict.shape

        if need_sigmoid:
            if isinstance(predict, (np.ndarray, torch.Tensor)):
                predict = torch.sigmoid(torch.as_tensor(predict))

        if isinstance(target, torch.Tensor):
            target = self._convert_to_numpy(target)

        if isinstance(predict, torch.Tensor):
            predict = self._convert_to_numpy(predict)

        target = np.reshape(target, (-1)).astype(np.int32)
        predict = np.reshape(predict, (-1))

        # Iou、Recall、Precision
        tn, fp, fn, tp = confusion_matrix(target, (predict > 0.5).astype(np.int32)).ravel()
        iou = tp / (tp + fp + fn + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        precision = tp / (tp + fp + 1e-5)

        # AP
        zip_list = zip(target, predict)
        sort_zip_list = sorted(zip_list, key=lambda x: x[1], reverse=True)
        zip_list = zip(*sort_zip_list)
        target, pred_score = [np.array(list(x)) for x in zip_list]
        pred_label = np.where(pred_score >= 0.5, 1, 0)
        ap = self._get_ap(ground_truth=target, pred_label=pred_label)

        # store
        self.iou_history.append(iou)
        self.recall_history.append(recall)
        self.precision_history.append(precision)
        self.ap_history.append(ap)

    def evaluate(self, model: nn.Module, val_dataloader: DataLoader, training=True) -> dict:
        pbar = tqdm(range(len(val_dataloader)), ncols=120)

        for idx, batch in zip(pbar, val_dataloader):
            outputs = model(pixel_values=batch["pixel_values"].to(model.device),
                            multimask_output=False)

            ground_truth_masks = batch["ground_truth_mask"].float().to(model.device).unsqueeze(1)
            predicted_masks = outputs.pred_masks.squeeze(1)
            predicted_masks = nn.functional.interpolate(predicted_masks,
                                                        size=ground_truth_masks.shape[-2:],
                                                        mode='bilinear',
                                                        align_corners=False)

            if not training:
                image = batch['original_image'][0]
                self.visualize(target=ground_truth_masks.squeeze(0, 1), predict=predicted_masks.squeeze(0, 1),
                               image=image, name=str(idx),
                               store_figure=self.config['store_figure'], show=self.config['show'])

            self.add_batch(ground_truth_masks, predicted_masks)

        iou_mean = np.mean(self.iou_history)
        recall_mean = np.mean(self.recall_history)
        precision_mean = np.mean(self.precision_history)
        ap_mean = np.mean(self.ap_history)
        self.reset()

        return {'iou': iou_mean,
                'recall': recall_mean,
                'precision': precision_mean,
                'ap': ap_mean}

    def visualize(self,
                  target: torch.Tensor,
                  predict: torch.Tensor,
                  image: [torch.Tensor | np.ndarray],
                  name: str,
                  store_figure=False,
                  threshold=0.3,
                  show=False,
                  need_sigmoid=True):
        """
            將預測結果可視化

            Args:
                target: (C, H, W)
                predict: (H, W) 還未經過sigmoid
                image: (B, C, H, W) 原始的輸入影像
                name: 影像名稱
                store_figure: 是否儲存結果
                threshold: 閥值
                show: 是否顯示結果
                need_sigmoid: 是否需要將predict進行sigmoid
        """
        if need_sigmoid:
            predict = torch.sigmoid(predict)

        if isinstance(target, torch.Tensor):
            target = self._convert_to_numpy(target)

        if isinstance(predict, torch.Tensor):
            predict = self._convert_to_numpy(predict)

        if isinstance(image, torch.Tensor):
            image = self._convert_to_numpy(image)

        predict = (predict > threshold).astype(np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Input")

        axes[1].imshow(predict)
        axes[1].set_title("Mask")

        axes[2].imshow(target)
        axes[2].set_title("GT")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if store_figure:
            plt.savefig(os.path.join(self.config['figure_path'], str(name) + '.jpg'))

        if show:
            plt.show()


if __name__ == '__main__':
    # load config file
    with open('../configs/config.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    evaluator = Evaluator(config=config)

    import cv2

    gt_mask = cv2.imread(r"D:\AOI\ControllerDataset\white\patch-train-gt\0.jpg", cv2.IMREAD_GRAYSCALE) / 255
    predict = cv2.imread(r"D:\AOI\ControllerDataset\white\patch-train-gt\0.jpg", cv2.IMREAD_GRAYSCALE) / 255

    # predict = np.array([0.8, 0.4, 0.1, 0.7, 0.6, 0.2, 0.9, 0.8, 0.6])
    # ground_truth = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0])

    evaluator.add_batch(target=gt_mask[None, None, ...], predict=predict[None, None, ...], need_sigmoid=False)
