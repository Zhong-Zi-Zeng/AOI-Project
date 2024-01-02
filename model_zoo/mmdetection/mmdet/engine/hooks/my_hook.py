from typing import Optional
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from mmengine.runner.runner import Runner
from mmengine.config import Config, ConfigDict
import torch



@HOOKS.register_module()
class MyHook(Hook):
    def __init__(self):
        pass

    def before_run(self, runner) -> None:
        model = runner.model
        model.eval()

        for i, data in enumerate(runner.val_dataloader):
            outputs = runner.model.train_step(data, runner.optim_wrapper)
            print(outputs)
