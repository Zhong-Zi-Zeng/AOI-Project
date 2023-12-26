from __future__ import annotations
import sys
import os

sys.path.append(os.path.join(os.getcwd()))
from engine.builder import Builder
from engine.general import get_work_dir_path
import numpy as np
import argparse
import torch
import subprocess

torch.manual_seed(10)
np.random.seed(10)

def get_args_parser():
    parser = argparse.ArgumentParser('Model training script.', add_help=False)

    parser.add_argument('--config', '-c', type=str, required=True,
                        help='The path of config.')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model evaluation script.',
                                     parents=[get_args_parser()])
    args = parser.parse_args()

    # Create builder
    builder = Builder(config_path=args.config, task='train')

    # Build config
    cfg = builder.build_config()

    # Build model
    model = builder.build_model(cfg)

    # Training
    model.train()

    # Generate training curve
    subprocess.run(['python', "./tools/training_curve.py",
                    '--model_type', cfg['model_name'],
                    '--result_path', get_work_dir_path(cfg),
                    '--output_path', get_work_dir_path(cfg),
                    ])