_base_ = [
    '_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_2x.py',
    '_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

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
eval_interval = 1
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=eval_interval)

# ==========model==========
model = dict(
    # Backbone
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth',  # noqa,
            prefix='backbone.')),
    # Neck
    neck=dict(in_channels=[96, 192, 384, 768]),
    # Head
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=num_classes,
            )
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=num_classes
        )
    ),

    # ==========test_cfg==========
    test_cfg=dict(
        rcnn=dict(nms=dict(type='nms', iou_threshold=nms_threshold))
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
    ann_file=data_root + '/annotations/instances_val.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator

# checkpoint
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=check_interval),
)
