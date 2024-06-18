import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import cv2


def train_one_epoch(model: torch.nn.Module,
                    train_dataloader: DataLoader,
                    epoch: int,
                    end_epoch: int,
                    optimizer: torch.optim.Optimizer,
                    loss_function,
                    tb_writer):
    model.train()
    pbar = tqdm(train_dataloader)
    for b, batch in enumerate(pbar):
        points = batch.get('points')
        boxes = batch.get('boxes')
        tr_image = batch.get('tr_image').permute(0, 3, 1, 2).to(model.device)  # [B, H, W, C] to [B, C, H, W]

        if points is not None:
            points = torch.from_numpy(np.array(points)).to(model.device)  # [B, number of point, 2]
            points = points[:, None, ...]  # [B, 1, number of point, 2]

        if boxes is not None:
            boxes = torch.from_numpy(np.array(boxes)).to(model.device)

        # forward
        outputs = model(pixel_values=tr_image,
                        input_points=points,
                        input_boxes=boxes,
                        multimask_output=False)

        # If you use bbox be the prompt, the predicted masks will be [B, number of bbox, 1, H, W]
        predicted_masks = torch.sum(outputs.pred_masks, dim=1)  # [B, 1, H, W]

        # compute loss
        ground_truth_masks = batch["gt_mask"].to(model.device).unsqueeze(1)  # [B, 1, H, W]

        if len(predicted_masks.dim_order()) != len(ground_truth_masks.dim_order()):
            predicted_masks = outputs.pred_masks.squeeze(1)  # [B, 1, H, W]

        predicted_masks = nn.functional.interpolate(predicted_masks,
                                                    size=ground_truth_masks.shape[-2:],
                                                    mode='bilinear',
                                                    align_corners=False)
        loss = loss_function(predicted_masks, ground_truth_masks)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learning_rates = [param_group['lr'] for param_group in optimizer.param_groups][0]

        step = b + len(train_dataloader) * epoch
        tb_writer.add_scalar('training loss', loss.item() * tr_image.size(0), step)

        pbar.set_description(
            f"Epoch:{epoch}/{end_epoch - 1} | Loss:{loss.item() * tr_image.size(0):.7f} | Learning rate:{learning_rates:.7f}")

        # if epoch > 50:
        #     tr_image = tr_image.permute(0, 2, 3, 1).cpu().numpy()
        #     ground_truth_masks = ground_truth_masks.cpu().numpy()
        #     predicted_masks = predicted_masks.detach().cpu().numpy()
        #     cv2.imshow('gt', ground_truth_masks[0][0])
        #     cv2.imshow('predicted_masks', predicted_masks[0][0])
        #     cv2.imshow('input', tr_image[0])
        #     cv2.waitKey(0)
