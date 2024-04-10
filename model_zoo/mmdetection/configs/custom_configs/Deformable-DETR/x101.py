_base_ = [
    'deformable_detr/deformable-detr.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_2x.py',
    '_base_/default_runtime.py'
]

# ==========Dataset setting==========
data_root = " "
classes = ()

# ==========Training setting==========
batch_size = 12
epochs = 50
width = 1024
height = 600
num_classes = 32
lr = 0.001
start_factor = 0.3
minimum_lr = 0
warmup_begin = 0
warmup_end = 3
nms_threshold = 0.7
check_interval = 1
optimizer = 'SGD'
backend_args = None

# ==========optimizer==========
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type=optimizer, lr=lr, betas=(0.937, 0.999), weight_decay=0.0005))

# ==========scheduler==========
param_scheduler = [
    dict(type='LinearLR',
         start_factor=start_factor,
         by_epoch=True,
         begin=warmup_begin,
         end=warmup_end),
    dict(type='CosineAnnealingLR',
         by_epoch=True,
         begin=warmup_end,
         end=epochs,
         eta_min=minimum_lr
         )
]

# ==========train_pipeline==========
albu_train_transforms = [
    dict(
        type='ColorJitter',
        hue=0.2,
        saturation=0.2,
        brightness=0.2),
    dict(
        type='Affine',
        scale=1.0,
        translate_px=0,
        shear=0,
        rotate=20
    ),
    dict(type='Perspective', scale=0.2),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ==========train_cfg==========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=check_interval)

# ==========model==========
model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),

    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=num_classes,
    )
)

# ==========dataloader==========
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        pipeline=test_pipeline,
        data_root=data_root,
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/instances_val2017.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# checkpoint
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=check_interval),
)













