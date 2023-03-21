angle_version = 'v2'
model = dict(
    type='RGFL',
    pretrained='/home/oneco/xyh/rotate_detection/checkpoints/lsk_t_backbone.pth.tar',
    backbone=dict(
        type='LSKNet',
        embed_dims=[32, 64, 160, 256],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2]),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 256],
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
optimizer = dict(
    type='AdamW',
    lr=0.0002/8, #/8*gpu_number,
    betas=(0.9, 0.999),
    weight_decay=0.05)
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
trainsplit_ann_folder = 'train/labelTxt'
trainsplit_img_folder = 'train/images'
valsplit_ann_folder = 'train/labelTxt'
valsplit_img_folder = 'train/images'
val_ann_folder = 'train/labelTxt'
val_img_folder = 'train/images'
# test_img_folder = 'vis/images'
test_img_folder = 'test/images'

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
work_dir = '/data/Aerial/checkpoints/lsknet/vanilla'
# mAP: 0.7178690717262265
# ap of each class: plane:0.8938191614432829, baseball-diamond:0.7131519904158613, bridge:0.5252875236036454, ground-track-field:0.6928146996257646, small-vehicle:0.7883459642522623, large-vehicle:0.7715583200828587, ship:0.8598633691702388, tennis-court:0.9088217481999867, basketball-court:0.7774958247965591, storage-tank:0.844648702498684, soccer-ball-field:0.4983786528511852, roundabout:0.6067074038122129, harbor:0.6482006434659426, swimming-pool:0.717501550635126, helicopter:0.5214405210397879
# COCO style result:
#
# AP50: 0.7178690717262265
# AP75: 0.41310116878785563
# mAP: 0.41445712109256894