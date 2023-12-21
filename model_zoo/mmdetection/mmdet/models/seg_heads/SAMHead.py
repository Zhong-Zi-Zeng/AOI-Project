import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.registry import MODELS
from mmengine.model import ModuleList
from mmcv.cnn import ConvModule, Linear
from ..layers import ConvUpsample
from mmdet.models.seg_heads.base_semantic_head import BaseSemanticHead
from mmdet.structures import SampleList
from typing import Dict, List, Tuple, Union
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType


@MODELS.register_module()
class SAMHead(BaseSemanticHead):
    def __init__(self,
                 num_classes: int,
                 in_channel: int,
                 img_size: list,
                 inner_channels: int = 128,
                 seg_rescale_factor: float = 1 / 4.,
                 loss_seg: ConfigType = dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=1.0),
                 init_cfg: OptMultiConfig = None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True)) -> None:
        super().__init__(num_classes, seg_rescale_factor, loss_seg, init_cfg)

        self.decoder = ModuleList()
        self.img_size = img_size

        for i in range(4):
            self.decoder.append(
                ConvUpsample(
                    in_channel,
                    inner_channels,
                    num_layers=i if i > 0 else 1,
                    num_upsample=i if i > 0 else 0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )

        self.conv_logits = nn.Conv2d(inner_channels, self.num_classes, 1)


    def forward(self, x: Union[Tensor, Tuple[Tensor]]) -> Dict[str, Tensor]:
        """Placeholder of forward function.

        Args:
            x (Tensor): Feature maps.

        Returns:
            Dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'seg_preds'
                and 'feats'.
        """

        feats = []
        for i, layer in enumerate(self.conv_upsample_layers):
            f = layer(x[self.start_level + i])
            feats.append(f)

        seg_feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        seg_preds = self.conv_logits(seg_feats)
        seg_preds = F.interpolate(seg_preds, self.img_size, mode='bilinear')
        seg_preds = F.sigmoid(seg_preds)

        output = dict(seg_preds=seg_preds, seg_feats=seg_feats)

        return output

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Args:
            x (Tensor): Feature maps.

        Returns:
            Dict[str, Tensor]: The loss of semantic head.
        """
        pass
