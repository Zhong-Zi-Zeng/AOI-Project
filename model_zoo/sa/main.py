from dataset import *
from models import build_model
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from train import train_one_epoch
from test import test_one_epoch
from utils.augmentation import create_augmentation
from utils.logger import colorstr
import time
import numpy as np
import torch
import os
import logging
import yaml
import monai
import argparse

from hooks import RecordTrainingLossHook, RemainingTimeHook, CheckStopTrainingHook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")

torch.manual_seed(10)
np.random.seed(10)


def get_args_parser():
    parser = argparse.ArgumentParser('Model evaluation scripten.', add_help=False)

    parser.add_argument('--cfg', type=str, required=True, help='The path of config.')
    parser.add_argument('--work_dir', type=str, help='save to project/name')

    return parser


def main(args, config: dict):
    # work_dir
    work_dir_path = args.work_dir
    os.makedirs(work_dir_path, exist_ok=True)

    # tensorboard
    tb_writer = SummaryWriter(log_dir=work_dir_path + '/log')  # Tensorboard

    # Now not support use boxes and use points be the prompt at the same time
    # if config['use_points'] and config['use_boxes']:
    #     raise ValueError("Now not support use boxes and use points be the prompt at the same time")

    # dataset
    train_dataset = CustomDataset(root=Path(config['coco_root']) / "train",
                                  ann_file=Path(config['coco_root']) / "annotations/instances_train.json",
                                  use_points=config['use_points'],
                                  use_boxes=config['use_boxes'],
                                  **create_augmentation(hyp=config, mode='training'))

    test_dataset = CustomDataset(root=Path(config['coco_root']) / "val",
                                 ann_file=Path(config['coco_root']) / "annotations/instances_val.json",
                                 use_points=False,
                                 use_boxes=False,
                                 **create_augmentation(hyp=config, mode='testing'))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=config['shuffle'],
                                  num_workers=config['num_workers'],
                                  collate_fn=train_dataset.collate_fn)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 collate_fn=train_dataset.collate_fn)

    # model
    device = config['device']
    model = build_model(config)
    model.to(int(device) if device.isdigit() and device != 'cpu' else device)

    # load pretrained weight
    start_epoch = 0
    if config['weight'] is not None:
        logger.info('Loading pretrained weight from {}'.format(config['weight']))
        ckpt = torch.load(config['weight'])
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model_state_dict'])
    logger.info(colorstr('parameter: ') + ', '.join([f"{key}={value}" for key, value in config.items()]))

    if start_epoch > config['end_epoch']:
        raise ValueError(
            f"The start epoch of the given weight file is '{start_epoch}', "
            f"but you specify the end epoch as {config['end_epoch']}.")

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
    logger.info(colorstr('optimizer: ') + config['optimizer'] + f" lr:{config['lr']}")

    # scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer,
                                         T_max=299,
                                         eta_min=config['minimum_lr'],
                                         verbose=False)

    # loss function
    loss_function = monai.losses.DiceCELoss(sigmoid=True,
                                            reduction='mean',
                                            squared_pred=True)

    # Init hooks
    max_iter = (config['end_epoch'] - start_epoch) * len(train_dataloader)
    remaining_time_hook = RemainingTimeHook(max_iter)
    check_stop_training_hook = CheckStopTrainingHook(str(work_dir_path))
    record_training_loss_hook = RecordTrainingLossHook()

    logger.info(colorstr('Start training'))
    for epoch in range(start_epoch, config['end_epoch']):
        # Training
        train_one_epoch(model=model,
                        train_dataloader=train_dataloader,
                        epoch=epoch,
                        end_epoch=config['end_epoch'],
                        optimizer=optimizer,
                        loss_function=loss_function,
                        tb_writer=tb_writer,
                        remaining_time_hook=remaining_time_hook,
                        check_stop_training_hook=check_stop_training_hook,
                        record_training_loss_hook=record_training_loss_hook)
        # Update scheduler
        cosine_scheduler.step()

        # Save
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict()
        }
        torch.save(ckpt, os.path.join(work_dir_path, "last.pt"))
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(ckpt, os.path.join(work_dir_path, f"weight_{epoch}" + '.pt'))

        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0 and config['eval_interval'] > 0:
            logger.info("\n" + colorstr('Evaluate...'))
            test_one_epoch(model=model,
                           test_dataloader=test_dataloader,
                           epoch=epoch,
                           loss_function=loss_function,
                           tb_writer=tb_writer,
                           work_dir_path=work_dir_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # load config file
    with open(args.cfg, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    main(args, config)
