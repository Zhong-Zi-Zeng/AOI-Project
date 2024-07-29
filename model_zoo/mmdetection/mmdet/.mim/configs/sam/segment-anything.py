_base_ = ['../_base_/datasets/coco_instance.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']

default_scope = 'mmdet'

# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='SAM',
    data_preprocessor=data_preprocessor,

    # Backbone
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    # Neck
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=1),

    # Head
    seg_head=dict(
        type='SAMHead',
        in_channel=256,
        num_classes=16,
        img_size=[2048, 3072],
        loss_seg=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=False,
            loss_weight=1.0,
        )
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

dataset_type = 'CocoDataset'
data_root = "D:/AOI/AOI-Project/utils/coco/"

classes = ('Border',
           'black_Marks',
           'Bright_Marks',
           'Pass',
           'Dirty',
           'Friction',
           'Black_point',
           'Buttom_scratches',
           'Small_scratches',
           'AnotherColor',
           'Severe_scratches',
           'Glossy',
           'ReturnPoint',
           'Assembly',
           'Scractches_Blackbackground',
           'Buttom_Friction')

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
    ),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
)

test_evaluator = val_evaluator
