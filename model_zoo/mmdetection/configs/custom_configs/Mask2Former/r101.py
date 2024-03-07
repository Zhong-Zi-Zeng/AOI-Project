_base_ = [
    'custom_configs/Mask2Former/r50_coco_panoptic.py',
    '_base_/datasets/coco_panoptic.py',
    '_base_/default_runtime.py'
]

# ==========Dataset setting==========
data_root = ' '
classes = ()

# ==========Training setting==========
batch_size = 12
epochs = 50
width = 1024
height = 600
num_classes = 80
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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# ==========train_cfg==========
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=10)


# ==========model==========
batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=(1024, 1024),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    batch_augments=batch_augments)
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    data_preprocessor=data_preprocessor,
    panoptic_head=dict(
        num_things_classes=num_classes,
        num_stuff_classes=0,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_classes,
        num_stuff_classes=0),
    test_cfg=dict(panoptic_on=False, iou_thr=nms_threshold),
)

# ==========dataloader==========
dataset_type = 'CocoDataset'
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + '/annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
)
test_evaluator = val_evaluator

# ==========Hook==========
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=check_interval),
)