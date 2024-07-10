from __future__ import annotations
import sys
import os
sys.path.append(os.path.join(os.getcwd()))

import time
import argparse

import numpy as np
import torch

from engine.builder import Builder

torch.manual_seed(10)
np.random.seed(10)

def get_args_parser():
    parser = argparse.ArgumentParser('Model training script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    parser.add_argument('--dir_name', type=str,
                        help='The name of work dir.')

    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # Create builder
    builder = Builder(config_path=args.config, task='train', work_dir_name=args.dir_name)

    # Build config
    cfg = builder.build_config()

    # Build model
    model = builder.build_model(cfg)

    # Training
    start_time = time.time()
    model.train()
    print(f'\nAll training processes took {(time.time() - start_time) / 3600:.2f} hours.')

