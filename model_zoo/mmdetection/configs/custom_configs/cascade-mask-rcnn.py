_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# ==========Dataset setting==========
data_root = "/home/miislab-server/MiiSLab_NAS/Student_Work/YA/coco"
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

# ==========Training setting==========
batch_size = 8
epochs = 50
width = 1024
height = 683
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
         end=epochs)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=epochs),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(width, height), keep_ratio=False),
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
    type='CascadeRCNN',

    # Backbone
    backbone=dict(
        type='ResNet',
        depth=50
    ),

    # Neck
    neck=dict(
        type='FPN'
    ),

    # Head
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=16,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=16,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=16,
            )
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=16
        )
    )
)

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
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm']
)
test_evaluator = val_evaluator
