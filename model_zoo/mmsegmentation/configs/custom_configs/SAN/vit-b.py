_base_ = [
    '_base_/models/san_vit-b16.py',
    '_base_/datasets/coco-stuff164k.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_160k.py'
]

# ==========Dataset setting==========
data_root = ' '
# metainfo = dict(
#     classes=("Scratch", "Friction", "Dirty", "Assembly"),
#     palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192]],
# )

# ==========Training setting==========
batch_size = 12
epochs = 50
width = 1024
height = 600
num_classes = 171
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
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(width, height), keep_ratio=True),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=(width, height), max_size=2560),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# ==========train_cfg==========
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=check_interval)

# ==========model==========
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'  # noqa
data_preprocessor = dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32))
model = dict(
    pretrained=pretrained,
    text_encoder=dict(dataset_name='coco-stuff164k'),
    decode_head=dict(num_classes=num_classes))

# ==========dataloader==========
train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=2,
    dataset=dict(
        type='ExampleDataset',
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(
            img_path='images/train2017'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='ExampleDataset',
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(
            img_path='images/val2017'),
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

# ==========Hook==========
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=check_interval),
)
