from Dataset import *
from models import build_model
from transformers import SamProcessor
from torch.utils.data import DataLoader
from torch.optim import Adam
from Train import train_one_epoch
from tqdm import tqdm
from torch import nn
from utils.augmentation import *
from utils.evaluate import Evaluator
import torch
import logging
import yaml
import monai

logging.basicConfig(level=logging.INFO)


def main(config):
    # evaluator
    evaluator = Evaluator(config=config)

    # augmentation
    augmentation = [
        GaussianNoise(variance=config['variance']),
        RandomFlip(prob=config['random_flip_prob']),
        ColorEnhance(brightness_gain=config['brightness_gain'],
                     contrast_gain=config['contrast_gain'],
                     saturation_gain=config['saturation_gain'],
                     hue_gain=config['hue_gain']),
        RandomPerspective(degrees=config['degrees'],
                          translate=config['translate'],
                          scale=config['scale'],
                          shear=config['shear'])
    ]

    # dataset
    processor = SamProcessor.from_pretrained(config['pretrained_model'])
    train_dataset = CustomDataset(image_path=config['train_image_path'],
                                  mask_path=config['train_mask_path'],
                                  use_points=config['use_points'],
                                  use_boxes=config['use_boxes'],
                                  processor=processor,
                                  trans=augmentation)

    val_dataset = CustomDataset(image_path=config['val_image_path'],
                                mask_path=config['val_mask_path'],
                                use_points=config['use_points'],
                                use_boxes=config['use_boxes'],
                                processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                                  num_workers=config['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=config['num_workers'])

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config)
    model.to(device)

    # load pretrained weight
    if os.path.isfile(config['pretrained_weight']):
        logging.info('Loading pretrained weight {}'.format(config['pretrained_weight']))
        model.load_state_dict(torch.load(config['pretrained_weight']))
    else:
        logging.warning('Can\'t find the pretrained weight {}'.format(config['pretrained_weight']))

    # optimizer
    optimizer = Adam(model.mask_decoder.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # loss function
    loss_function = monai.losses.DiceCELoss(sigmoid=True,
                                            reduction='mean',
                                            lambda_dice=config['lambda_dice'],
                                            lambda_ce=config['lambda_ce'])

    print('Start training')
    model.train()
    for epoch in range(config['start_epoch'], config['epoch']):
        # Training
        train_one_epoch(model=model,
                        train_dataloader=train_dataloader,
                        epoch=epoch,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        use_points=config['use_points'])

        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            print('Evaluate......')
            metric = evaluator.evaluate(model=model, val_dataloader=val_dataloader, training=True)
            for k, v in metric.items():
                print('{}:{:.3f}'.format(k, v), end='  | ')
            print('\n')
            torch.save(model.state_dict(), config['output_path'])


if __name__ == "__main__":
    # load config file
    with open('./configs/config_1.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    main(config)
