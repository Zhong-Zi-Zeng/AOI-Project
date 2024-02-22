import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import time
import cv2
import torch


def train_one_epoch(model: torch.nn.Module,
                    train_dataloader: DataLoader,
                    epoch: int,
                    optimizer: torch.optim.Optimizer,
                    loss_function):
    for b, batch in enumerate(train_dataloader):
        points = batch.get('points')
        boxes = batch.get('boxes')

        # TODO: Permute
        tr_image = batch.get('tr_image').to(model.device)

        if points is not None:
            points = torch.from_numpy(np.array(points)).to(model.device)

        if boxes is not None:
            boxes = torch.from_numpy(np.array(boxes)).to(model.device)

        # forward
        outputs = model(pixel_values=tr_image,
                        input_points=points,
                        input_boxes=boxes,
                        multimask_output=False)

        # compute loss
        ground_truth_masks = batch["gt_mask"].to(model.device).unsqueeze(1)  # (B, 1, H, W)
        predicted_masks = outputs.pred_masks.squeeze(1)
        predicted_masks = nn.functional.interpolate(predicted_masks,
                                                    size=ground_truth_masks.shape[-2:],
                                                    mode='bilinear',
                                                    align_corners=False)
        loss = loss_function(predicted_masks, ground_truth_masks)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: 新增tensorboard
        print("Epoch:{} | Batch:{}/{} | Loss:{:.3f} ".format(epoch,
                                                             b,
                                                             len(train_dataloader),
                                                             loss.item()))
