angle_version = 'v2'
model = dict(
    type='RGFL',
    pretrained='/home/oneco/xyh/rotate_detection/checkpoints/convnextv2_tiny_1k_224_ema.pt',
    backbone=dict(
        type='ConvNeXtV2',
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        out_indices=[0, 1, 2, 3],
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
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
        assigner=dict(type='RATSSAssigner', topk=9,
                      iou_calculator=dict(type='RBboxOverlaps2D')),
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
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
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
test_img_folder = 'test/images'
example_ann_folder = 'train/labelTxt'
example_img_folder = 'train/images'

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
work_dir = '/data/Aerial/checkpoints/convnextv2/sup_in1k_mixversion'

# consistent version
# mAP: 0.7219308884839666
# ap of each class: plane:0.8892314509785048, baseball-diamond:0.7470812648662907, bridge:0.5081310588429688, ground-track-field:0.6865744082866941, small-vehicle:0.7733794539160468, large-vehicle:0.8015322442794031, ship:0.8714372256420894, tennis-court:0.9079197979363719, basketball-court:0.7680481037581673, storage-tank:0.8347169165024132, soccer-ball-field:0.5576524965419764, roundabout:0.6249184367190639, harbor:0.6748911441236604, swimming-pool:0.7187556659324483, helicopter:0.46469365893340076
# COCO style result:
#
# AP50: 0.7219308884839666
# AP75: 0.45201948198235203
# mAP: 0.4324892222445328


# mix version
# mAP: 0.7315988408902057
# ap of each class: plane:0.8914831274147982, baseball-diamond:0.7537478120289557, bridge:0.5082378241022397, ground-track-field:0.6921112462539067, small-vehicle:0.7650547509860582, large-vehicle:0.7718859010871623, ship:0.8739342770786094, tennis-court:0.9082566261396324, basketball-court:0.7928292854420429, storage-tank:0.8292468044033257, soccer-ball-field:0.5928842722889106, roundabout:0.643765733872101, harbor:0.6818582212587954, swimming-pool:0.7204615290232207, helicopter:0.5482252019733279
# COCO style result:
#
# AP50: 0.7315988408902057
# AP75: 0.45852467461257257
# mAP: 0.4393599881816117