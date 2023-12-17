import torch.nn as nn
from mmdet.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class SAMBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvModule(3, 128, 3)

    def forward(self, inputs):
        print(inputs.shape)

        return self.conv1(inputs)

    def loss(self):
        pass