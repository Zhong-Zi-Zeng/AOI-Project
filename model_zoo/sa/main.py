import numpy as np

from Dataset import *
from models import build_model
from transformers import SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from Train import train_one_epoch
import numpy as np
from tqdm import tqdm
from torch import nn
from utils.augmentation import *
from utils.evaluate import Evaluator
import torch
import os
import logging
import yaml
import monai

logging.basicConfig(level=logging.INFO)

torch.manual_seed(10)
np.random.seed(10)


def main(config):
    # evaluator
    evaluator = Evaluator(config=config)

    # augmentation
    if config['use_boxes'] and config['use_points']:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
            keypoint_params=A.KeypointParams(format='xy'),
            additional_targets={'mask': 'image'})
    elif config['use_boxes']:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
            additional_targets={'mask': 'image'})
    elif config['use_points']:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], keypoint_params=A.KeypointParams(format='xy'),
            additional_targets={'mask': 'image'})
    else:
        trans = A.Compose([
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255),
        ], additional_targets={'mask': 'image'})

    # dataset
    train_dataset = CustomDataset(root=Path(config['coco_root']) / "train2017",
                                  ann_file=Path(config['coco_root']) / "annotations/instances_train2017.json",
                                  use_points=config['use_points'],
                                  use_boxes=config['use_boxes'],
                                  trans=trans)

    # val_dataset = CustomDataset(root=Path(config['coco_root']) / "val2017",
    #                             ann_file=Path(config['coco_root']) / "annotations/instances_val2017.json",
    #                             use_points=False,
    #                             use_boxes=False,
    #                             trans=trans)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=config['shuffle'],
                                  num_workers=0,
                                  collate_fn=train_dataset.collate_fn)

    # val_dataloader = DataLoader(val_dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=0,
    #                             collate_fn=lambda batch: collate_fn(batch, processor, False, False))

    # model
    device = config['device']
    model = build_model(config)
    model.to(device)

    # load pretrained weight
    if os.path.isfile(config['pretrained_weight']):
        logging.info('Loading pretrained weight {}'.format(config['pretrained_weight']))
        model.load_state_dict(torch.load(config['pretrained_weight']))
    else:
        logging.warning('Can\'t find the pretrained weight {}'.format(config['pretrained_weight']))

    # optimizer
    if config['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(),
                        lr=config['lr'],
                        momentum=config['momentum'],
                        weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(),
                         lr=config['lr'],
                         betas=(config['betas_0'], config['betas_1']),
                         weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(),
                          lr=config['lr'],
                          betas=(config['betas_0'], config['betas_1']),
                          weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Can not find the optimizer of {config['optimizer']}")

    # scheduler
    linear_scheduler = LinearLR(optimizer=optimizer,
                                start_factor=config['start_factor'],
                                last_epoch=-1,
                                verbose=True)

    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=299,
                                         eta_min=config['minimum_lr'],
                                         last_epoch=-1,
                                         verbose=True)

    # loss function
    loss_function = monai.losses.DiceCELoss(sigmoid=True,
                                            reduction='mean',
                                            squared_pred=True)

    print('Start training')
    model.train()
    for epoch in range(config['start_epoch'], config['end_epoch']):
        # Training
        train_one_epoch(model=model,
                        train_dataloader=train_dataloader,
                        epoch=epoch,
                        optimizer=optimizer,
                        loss_function=loss_function)

        # Update scheduler
        if epoch < config['warmup_epoch']:
            linear_scheduler.step()
        else:
            cosine_scheduler.step()

        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            # TODO: Evaluate
            pass

        # Save
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(model.state_dict(), config['output_path'])


if __name__ == "__main__":
    # load config file
    with open('./configs/config_1.yaml', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    main(config)
