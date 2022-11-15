model = dict(
    type='RReppointsDetector',
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
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='CFAHead',
        num_classes=15,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.5),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        transform_method='direct',
        use_cfa=True,
        topk=6,
        anti_factor=0.75),
    # model training and testing settings
    train_cfg=dict(
        init=dict(
            assigner=dict(type='RPointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='ConvexOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='rnms', iou_threshold=0.1),
        max_per_img=1000))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
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
    samples_per_gpu=4,
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
work_dir = '/data/Aerial/checkpoints/cfa/vanila'
# with out anti-Aliasing positive weight
# mAP: 0.6029221399282415
# ap of each class: plane:0.8819266680661432, baseball-diamond:0.7238464666311806, bridge:0.32893691657804663,
# ground-track-field:0.365532134609223, small-vehicle:0.6909498204551833, large-vehicle:0.515259493409674,
# ship:0.7247332020513517, tennis-court:0.8681391019906041, basketball-court:0.5720441878933643,
# storage-tank:0.8258359480364655, soccer-ball-field:0.5281822729397356, roundabout:0.6103338533441663,
# harbor:0.5102272201033338, swimming-pool:0.5697717987309132, helicopter:0.3281130140842371

# with anti-aliasing positive weight
# mAP: 0.6809795968486421
# ap of each class: plane:0.8830916800050922, baseball-diamond:0.6848790087285794, bridge:0.4405399879888629,
# ground-track-field:0.6778293587441953, small-vehicle:0.7420282249151648, large-vehicle:0.7390311669860727,
# ship:0.8601089545688875, tennis-court:0.908743878843001, basketball-court:0.7636224134494405,
# storage-tank:0.8268111020796128, soccer-ball-field:0.5208674676261925, roundabout:0.5854761342340916,
# harbor:0.6422842155008283, swimming-pool:0.6142458757185784, helicopter:0.3251344833410289
# COCO style result:
# AP50: 0.6809795968486421
# AP75: 0.4163386552639328
# mAP: 0.40707325917107384