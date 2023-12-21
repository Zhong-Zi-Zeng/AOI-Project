from mmdet.registry import MODELS
from .base import BaseDetector
from torch import Tensor
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import Dict, List, Tuple, Union
import torch.nn.functional as F


@MODELS.register_module()
class SAM(BaseDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 seg_head: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None
                 ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.seg_head = MODELS.build(seg_head)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.

        """

        feature = self.backbone(batch_inputs)
        output = self.seg_head.predict(feature, batch_data_samples)
        output = F.sigmoid(output['seg_preds'])

        return output

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process. Usually includes backbone, neck and head forward without any post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``seg_head`` forward
        """

        feature = self.backbone(batch_inputs)
        output = self.seg_head.forward(feature)

        return output

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: (N, C, H ,W)
        """
        x = self.backbone(batch_inputs)
        x = self.neck(x)

        return x
