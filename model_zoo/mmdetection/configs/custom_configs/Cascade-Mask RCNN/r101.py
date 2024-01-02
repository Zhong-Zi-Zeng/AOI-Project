_base_ = [
    '_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_2x.py',
    '_base_/default_runtime.py'
]

# ==========Dataset setting==========
coco_root = " "
classes = ()

# ==========Training setting==========
batch_size = 12
epochs = 50
width = 1024
height = 683
num_classes = 32
backend_args = None

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.937, 0.999), weight_decay=0.0005))

# scheduler
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.3,
         by_epoch=True,
         begin=0,
         end=3),
    dict(type='CosineAnnealingLR',
         by_epoch=True,
         begin=3,
         end=epochs,
         eta_min = 0
         )
]

# checkpoint
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
    dict(type='RandomFlip', prob=0.),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=1)

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

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        pipeline=train_pipeline,
        data_root=coco_root,
        metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        pipeline=test_pipeline,
        data_root=coco_root,
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=coco_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator
