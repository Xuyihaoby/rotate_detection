angle_version = 'v2'
model = dict(
    type='RGFL',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
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
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='RGFLHeadVanilla',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_max=8,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='RAnchorGenerator',
            ratios=[1.0],
            octave_base_scale=4,
            scales_per_octave=1,
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
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='RATSSAssigner', topk=9, iou_calculator=dict(type='RBboxOverlaps2D')),
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
    step=[8, 11])
total_epochs = 12

dataset_type = 'DOTADatasetV1'
data_root = '/data/Aerial/DOTA1_0/simple/'
trainsplit_ann_folder = 'trainval200/labelTxt'
trainsplit_img_folder = 'trainval200/images'
valsplit_ann_folder = 'trainval200/labelTxt'
valsplit_img_folder = 'trainval200/images'
val_ann_folder = 'trainval200/labelTxt'
val_img_folder = 'trainval200/images'
test_img_folder = 'test200/images'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RLoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RResize', img_scale=(1024, 1024)),
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
        img_scale=(1024, 1024),
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
        ann_file=data_root + trainsplit_ann_folder,
        img_prefix=data_root + trainsplit_img_folder,
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        ann_file=data_root + valsplit_ann_folder,
        img_prefix=data_root + valsplit_img_folder,
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_img_folder,
        img_prefix=data_root + test_img_folder,
        pipeline=test_pipeline,
        test_mode=True,
        version=angle_version))

checkpoint_config = dict(interval=4)

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
work_dir = '/data/Aerial/checkpoints/rgfl/vanilla'

# train dataset no val
# mAP: 0.7202526755309105
# ap of each class: plane:0.8838651453675991, baseball-diamond:0.7296169920058765, bridge:0.4793641132180634,
# ground-track-field:0.6721617789147231, small-vehicle:0.7882338589790234, large-vehicle:0.7874030194547998,
# ship:0.8682390736177887, tennis-court:0.9090226077453727, basketball-court:0.7614667772498331,
# storage-tank:0.821164062778348, soccer-ball-field:0.6034334825472147, roundabout:0.6261959944723979,
# harbor:0.6462752342728243, swimming-pool:0.6700135510858202, helicopter:0.5573344412539748
# COCO style result:
# AP50: 0.7202526755309105
# AP75: 0.4269751226629144
# mAP: 0.4132831038814989
