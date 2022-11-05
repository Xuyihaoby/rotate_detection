model = dict(
    type='RRetinaNet',
    pretrained='/home/oneco/xyh/rotate_detection/checkpoints/focalnet_tiny_srf.pth',
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[11,9,9,7],
        focal_levels=[2,2,2,2],
        use_conv_embed=False,
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RRetinaHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='RAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
            angles=[0.]),
        bbox_coder=dict(
            type='DeltaRXYWHThetaBBoxCoder',
            target_means=[.0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='rnms', iou_threshold=0.1),
        max_per_img=2000))

# optimizer
optimizer = dict(type='AdamW', lr=0.0001/4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
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
test_img_folder = 'test/images'
example_ann_folder = 'train/labelTxt'
example_img_folder = 'train/images'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RLoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
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
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + valsplit_ann_folder,
        img_prefix=data_root + valsplit_img_folder,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_img_folder,
        img_prefix=data_root + test_img_folder,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=24, metric='bbox')

checkpoint_config = dict(interval=2)

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
work_dir = '/data/Aerial/checkpoints/focal/rotate_iou_tiny'
# mAP: 0.7247888939877017
# ap of each class: plane:0.8890799728796611, baseball-diamond:0.7597342236479144, bridge:0.463426363048833,
# ground-track-field:0.7397327815351659, small-vehicle:0.7883954444849601, large-vehicle:0.7056218022287561,
# ship:0.8552730310102287, tennis-court:0.9083972315206953, basketball-court:0.8111157508950382,
# storage-tank:0.809432126376383, soccer-ball-field:0.6383107104962339, roundabout:0.6546842551108486,
# harbor:0.6345839638952897, swimming-pool:0.7228760835128437, helicopter:0.4911696691726716
# COCO style result:
# AP50: 0.7247888939877017
# AP75: 0.44788596768523264
# mAP: 0.429792551043821

