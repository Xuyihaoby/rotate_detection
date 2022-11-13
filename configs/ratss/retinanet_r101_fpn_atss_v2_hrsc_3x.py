angle_version = 'v2'
model = dict(
    type='RRetinaNet',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RRetinaHeadATSS',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='RAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1,
            # scales_per_octave=3,
            ratios=[1.0],
            # ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
            angles=[0.],
            version=angle_version),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            target_means=[.0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
            angle_range=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0, version=angle_version)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(type='RATSSAssigner', topk=9, version=angle_version,
                      iou_calculator=dict(type='RBboxOverlaps2D', version=angle_version)),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='rnms', iou_threshold=0.1, version=angle_version),
        max_per_img=2000))

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33])
total_epochs = 36

dataset_type = 'HRSC'
data_root = '/data/Aerial/HRSC2016/images/HRSC2016'
train_ann_folder = '/Train/trainval.txt'
train_img_folder = '/Train/'
val_ann_folder = '/FullDataSet/test.txt'
val_img_folder = '/FullDataSet'
test_ann_folder = '/FullDataSet/test.txt'
test_img_folder = '/FullDataSet'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RLoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Randomrotate', border_value=0, rotate_mode='value', rotate_ratio=0.5,
         rotate_values=[30, 60, 90, 120, 150],
         auto_bound=False,
         version=angle_version),
    dict(type='RResize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RRandomFlip', flip_ratio=0.5, version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='RResize', keep_ratio=True),
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
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        ann_file=data_root + val_ann_folder,
        img_prefix=data_root + val_img_folder,
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_ann_folder,
        img_prefix=data_root + test_img_folder,
        pipeline=test_pipeline,
        test_mode=True,
        version=angle_version))
evaluation = dict(interval=24, metric='bbox')

checkpoint_config = dict(interval=12)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/data/Aerial/checkpoints/rretinanet/hrsc_atss_v2_3x_rr_keepratio'
# voc12
# ---------------iou_thr: 0.5---------------
#
# +-------+------+------+--------+-------+
# | class | gts  | dets | recall | ap    |
# +-------+------+------+--------+-------+
# | ship  | 1228 | 2424 | 0.965  | 0.950 |
# +-------+------+------+--------+-------+
# | mAP   |      |      |        | 0.950 |
# +-------+------+------+--------+-------+
# {'mAP': 0.9495044350624084}

# voc 07
# ---------------iou_thr: 0.5---------------
#
# +-------+------+------+--------+-------+
# | class | gts  | dets | recall | ap    |
# +-------+------+------+--------+-------+
# | ship  | 1228 | 2424 | 0.965  | 0.899 |
# +-------+------+------+--------+-------+
# | mAP   |      |      |        | 0.899 |
# +-------+------+------+--------+-------+
# {'mAP': 0.8985073566436768}