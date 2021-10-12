_base_ = ['./faster_rcnn_r50_fpn_1x.py']
model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='ROHEMSampler'
            )
        )
    )
)

dataset_type = 'SSDD'
data_root = '/data1/public_dataset/SAR/SSDD/'
train_ann_folder = 'train.txt'
train_img_folder = ''
val_ann_folder = 'test.txt'
val_img_folder = ''
test_ann_folder = 'test.txt'
test_img_folder = ''

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RLoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(608, 608)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore','hor_gt_bboxes', 'hor_gt_bboxes_ignore',
                               'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='RRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + train_ann_folder,
        img_prefix=data_root + train_img_folder,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + val_ann_folder,
        img_prefix=data_root + val_img_folder,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_ann_folder,
        img_prefix=data_root + test_img_folder,
        pipeline=test_pipeline,
        test_mode=True))
work_dir = '/home/lzy/xyh/Netmodel/rotate_detection/checkpoints/rf/'
