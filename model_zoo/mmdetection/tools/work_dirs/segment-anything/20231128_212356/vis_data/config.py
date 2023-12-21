launcher = 'none'
model = dict(
    backbone=dict(type='SAMBackbone'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    head=dict(type='SAMHead'),
    neck=None,
    type='SAM')
work_dir = './work_dirs\\segment-anything'
