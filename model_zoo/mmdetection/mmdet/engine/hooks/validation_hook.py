import os
import sys

sys.path.append(os.path.join(os.getcwd()))

from typing import Optional
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from torch.utils.tensorboard import SummaryWriter
from tools.evaluation import Evaluator
from tqdm import tqdm
from engine.general import load_yaml
import torch


@HOOKS.register_module()
class ValidationHook(Hook):
    def __init__(self):
        self.step = 0

    def after_train_epoch(self, runner) -> None:
        if not hasattr(self, "tb_writer"):
            setattr(self, "tb_writer", SummaryWriter(log_dir=os.path.join(runner.work_dir, 'log')))

        if not hasattr(self, "final_config"):
            setattr(self, "final_config",
                    load_yaml(os.path.join(runner.work_dir.replace(" ", ""), 'final_config.yaml')))

        model = runner.model
        model.eval()

        # Validation loss
        bar = tqdm(runner.val_dataloader)
        for data in bar:
            outputs = runner.model.train_step(data, runner.optim_wrapper)
            for key, value in outputs.items():
                self.tb_writer.add_scalar('Val/' + key, value.item(), self.step)
            self.step += 1

        # Evaluate recall and FPR
        if (runner.epoch + 1) % self.final_config['eval_period'] == 0:
            print("Evaluate:")

            # Save last epoch
            last_weight_path = os.path.join(runner.work_dir, "last.pt")
            torch.save(runner.model.state_dict(), last_weight_path)

            # Evaluate
            self.final_config.update({'weight': last_weight_path})
            evaluator = Evaluator.build_by_config(cfg=self.final_config)
            recall_and_fpr_for_all = evaluator.eval()
            tags = ["metrics/Recall(image)", "metrics/FPR(image)", "metrics/Recall(defect)", "metrics/FPR(defect)"]
            for x, tag in zip(recall_and_fpr_for_all, tags):
                self.tb_writer.add_scalar(tag, x, runner.epoch)
