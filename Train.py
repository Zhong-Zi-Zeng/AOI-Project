import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import time
import torch


def train_one_epoch(model: torch.nn.Module,
                    train_dataloader: DataLoader,
                    epoch: int,
                    optimizer: torch.optim.Optimizer,
                    loss_function,
                    use_points: bool = False):
    for b, batch in enumerate(train_dataloader):
        # forward pass
        if use_points:
            outputs = model(pixel_values=batch["pixel_values"].to(model.device),
                            input_points=batch["input_points"].to(model.device),
                            multimask_output=False)
        else:
            outputs = model(pixel_values=batch["pixel_values"].to(model.device),
                            multimask_output=False)

        # compute loss
        ground_truth_masks = batch["ground_truth_mask"].float().to(model.device).unsqueeze(1)

        predicted_masks = outputs.pred_masks.squeeze(1)
        predicted_masks = nn.functional.interpolate(predicted_masks,
                                                    size=ground_truth_masks.shape[-2:],
                                                    mode='bilinear',
                                                    align_corners=False)

        loss = loss_function(predicted_masks, ground_truth_masks)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch:{} | Batch:{}/{} | Loss:{:.3f} ".format(epoch,
                                                             b,
                                                             len(train_dataloader),
                                                             loss.item()))
