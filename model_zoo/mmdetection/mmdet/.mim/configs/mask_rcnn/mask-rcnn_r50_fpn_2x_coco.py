_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
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

train_pipeline = [
    dict(type='Resize', scale=(2048, 1365), keep_ratio=True),
    dict(type='RandomFlip', prob=0.),
]


model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=16),
        mask_head=dict(
            num_classes=16
        )
    )
)

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
    ann_file=data_root + 'annotations/instances_val2017.json',
)

test_evaluator = val_evaluator