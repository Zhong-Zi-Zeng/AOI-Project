from typing import Optional
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner.runner import Runner
from mmengine.config import Config, ConfigDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@HOOKS.register_module()
class ValidationHook(Hook):
    def __init__(self):
        self.step = 0

    def after_train_epoch(self, runner) -> None:
        if not hasattr(self, "tb_writer"):
            self.tb_writer = SummaryWriter(log_dir=runner.work_dir + './log')

        model = runner.model
        model.eval()

        print("Evaluate:")
        bar = tqdm(runner.val_dataloader)
        for data in bar:
            outputs = runner.model.train_step(data, runner.optim_wrapper)
            for key, value in outputs.items():
                self.tb_writer.add_scalar('Val/' + key, value.item(), self.step)
            self.step += 1
        model.train()
