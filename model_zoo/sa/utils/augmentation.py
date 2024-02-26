from __future__ import annotations
from typing import Optional, Dict
import albumentations as A


def create_augmentation(hyp: Optional[dict] = None,
                        mode='training') -> Dict[str, A.Compose]:
    """
        Args:
            hyp: 存放每個aug參數的字典
            mode: training or testing, 如果選擇testing則只有使用resize和ToFloat功能
        Returns:
            回傳有使用box、point和沒有使用box、point所對應的augment
            {
                "trans": 不使用point、box的augment,
                "trans_with_bboxes": 不使用point，使用box的augment,
                "trans_with_points": 使用point，不使用box的augment,
                "trans_with_bboxes_points": 使用point、box的augment
            }
    """
    if hyp is None:
        hyp = {}

    if mode == 'training':
        aug = [
            A.Resize(width=1024, height=1024),
            # A.ColorJitter(hue=hyp.get('hsv_h', 0.2),
            #               saturation=hyp.get('hsv_s', 0.2),
            #               brightness=hyp.get('hsv_v', 0.2)),
            A.Affine(scale=hyp.get('scale', 1.0),
                     translate_px=hyp.get('translate', 0),
                     shear=hyp.get('shear', 0),
                     rotate=hyp.get('degrees', 20)),
            A.Perspective(scale=hyp.get('perspective', 0.2)),
            A.HorizontalFlip(p=hyp.get('fliplr', 0.5)),
            A.VerticalFlip(p=hyp.get('flipud', 0.5)),
            A.ToFloat(max_value=255),
        ]
    elif mode == 'testing':
        aug = [
            A.Resize(width=1024, height=1024),
            A.ToFloat(max_value=255)
        ]
    else:
        raise ValueError("The argument 'mode' must in {'train', 'test'}")

    trans = A.Compose(aug, additional_targets={'mask': 'image'})
    trans_with_bboxes = A.Compose(aug,
                                  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
                                  additional_targets={'mask': 'image'})
    trans_with_points = A.Compose(aug, keypoint_params=A.KeypointParams(format='xy'),
                                  additional_targets={'mask': 'image'})
    trans_with_bboxes_points = A.Compose(aug,
                                         bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
                                         keypoint_params=A.KeypointParams(format='xy'),
                                         additional_targets={'mask': 'image'})

    return {"trans": trans,
            "trans_with_bboxes": trans_with_bboxes,
            "trans_with_points": trans_with_points,
            "trans_with_bboxes_points": trans_with_bboxes_points}
