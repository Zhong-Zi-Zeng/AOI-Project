_base_ = [
    '_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
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
optimizer = 'SGD'
backend_args = None

# optimizer
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
         eta_min = minimum_lr
         )
]

# ==========train_pipeline==========
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
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
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=1)

# ==========model==========
model = dict(
    # Backbone
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
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
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator

# checkpoint
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
)
