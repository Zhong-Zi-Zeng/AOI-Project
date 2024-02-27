from __future__ import annotations
from dataset import *
from models import build_model
from transformers import SamProcessor
from torch.utils.data import DataLoader
from utils.evaluate import Evaluator
import torch
import logging
import yaml

logging.basicConfig(level=logging.INFO)


def main(config):
    # evaluator
    evaluator = Evaluator(config=config)

    # dataset
    processor = SamProcessor.from_pretrained(config['pretrained_model'])

    val_dataset = CustomDataset(image_path=config['val_image_path'],
                                mask_path=config['val_mask_path'],
                                use_points=False,
                                use_boxes=False,
                                processor=processor)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

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

    # Testing
    logging.info('Start testing')
    model.eval()
    metric = evaluator.evaluate(model=model, val_dataloader=val_dataloader, training=False)

    print('Evaluate......')
    for k, v in metric.items():
        print('{}:{:.3f}'.format(k, v), end='  | ')
    print('\n')



if __name__ == "__main__":
    # load config file
    with open('configs/config.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    main(config)
