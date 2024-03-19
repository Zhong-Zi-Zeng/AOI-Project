import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
import time
import cv2
import torch


def test_one_epoch(model: torch.nn.Module,
                   test_dataloader: DataLoader,
                   epoch,
                   loss_function,
                   tb_writer):
    model.eval()
    pbar = tqdm(test_dataloader)
    sum_of_loss = 0
    sum_of_precision = 0
    sum_of_recall = 0

    for b, batch in enumerate(pbar):
        tr_image = batch.get('tr_image').permute(0, 3, 1, 2).to(model.device)

        # forward
        outputs = model(pixel_values=tr_image,
                        input_points=None,
                        input_boxes=None,
                        multimask_output=False)

        # compute loss
        ground_truth_masks = batch["gt_mask"].to(model.device).unsqueeze(1)  # (B, 1, H, W)
        predicted_masks = outputs.pred_masks.squeeze(1)
        predicted_masks = nn.functional.interpolate(predicted_masks,
                                                    size=ground_truth_masks.shape[-2:],
                                                    mode='bilinear',
                                                    align_corners=False)
        with torch.no_grad():
            sum_of_loss += loss_function(predicted_masks, ground_truth_masks).item()
            ground_truth_masks = ground_truth_masks.reshape((-1,)).cpu().numpy().astype(int)
            predicted_masks = torch.sigmoid(predicted_masks.reshape((-1,))).cpu().numpy()
            predicted_masks = np.where(predicted_masks > 0.1, 0, 1).astype(int)
            precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth_masks, predicted_masks,
                                                                             zero_division=0, average='macro')
            sum_of_precision += precision
            sum_of_recall += recall

    sum_of_precision /= len(test_dataloader)
    sum_of_recall /= len(test_dataloader)
    sum_of_loss /= len(test_dataloader)
    f1_score = (2 * sum_of_recall * sum_of_precision) / (sum_of_recall + sum_of_precision + 1e-5)
    tb_writer.add_scalar('metrics/precision', sum_of_precision, epoch)
    tb_writer.add_scalar('metrics/recall', sum_of_recall, epoch)
    tb_writer.add_scalar('metrics/f1_score', f1_score, epoch)
    tb_writer.add_scalar('metrics/loss', sum_of_loss, epoch)

    print(
        f"Precision:{sum_of_precision:.5f} | Recall:{sum_of_recall:.5f} | F1 score:{f1_score:.5f} | Loss:{sum_of_loss:.5f}")
