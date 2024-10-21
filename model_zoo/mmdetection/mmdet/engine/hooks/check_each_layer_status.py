from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class CheckEachLayerStatus(Hook):
    priority = 'VERY_LOW'

    def before_run(self, runner) -> None:
        model = runner.model

        if hasattr(runner.model, 'module'):
            model = runner.model.module

        print("Checking if layers are frozen (requires_grad=False):")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: trainable")
            else:
                print(f"{name}: frozen")
