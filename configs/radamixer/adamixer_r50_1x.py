def __get_debug():
    import os
    return 'C_DEBUG' in os.environ


debug = __get_debug()

log_interval = 100

IMAGE_SCALE = (1024, 1024)

# dataset settings
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
    dict(type='RLoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=IMAGE_SCALE),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMAGE_SCALE,
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
        test_mode=True),
)
evaluation = dict(interval=1, metric='bbox')

num_stages = 6
num_query = 300
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048

# P_in for spatial mixing in the paper.
in_points_list = [32, ] * num_stages

# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns_list = [128, ] * num_stages

# G for the mixer grouping in the paper. Please distinguishe it from num_heads in MHSA in this codebase.
n_group_list = [4, ] * num_stages

model = dict(
    type='QueryBased',
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
        type='ChannelMapping',
        in_channels=[256, 512, 1024, 2048],
        out_channels=FEAT_DIM,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    rpn_head=dict(
        type='RInitialQueryGenerator',
        num_query=num_query,
        content_dim=QUERY_DIM),
    roi_head=dict(
        type='RVAdaMixerDecoder',
        featmap_strides=[4, 8, 16, 32],
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=QUERY_DIM,
        bbox_head=[
            dict(
                type='RVAdaMixerDecoderStage',
                num_classes=15,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=FF_DIM,
                content_dim=QUERY_DIM,
                feat_channels=FEAT_DIM,
                dropout=0.0,
                in_points=in_points_list[stage_idx],
                out_points=out_patterns_list[stage_idx],
                n_groups=n_group_list[stage_idx],
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='RotatedIoULoss', loss_weight=2, linear=True),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or
                # updating bounding boxes.
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for stage_idx in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywhtheta'),
                    iou_cost=dict(type='IoUCost', iou_mode='iou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_query)))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.000025/4,
    weight_decay=0.0001,
)

optimizer_config = dict(
    grad_clip=dict(max_norm=35.0, norm_type=2),
)

# learning policy
lr_config = dict(
    policy='step',
    step=[16, 22],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)
total_epochs = 24

log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

# postfix = '_' + __date()

find_unused_parameters = True

resume_from = None
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
work_dir = '/data/Aerial/checkpoints/rgfl/vanilla'
workflow = [('train', 1)]

# epoch_24
# 2*2 bs 2 lr
# mAP: 0.6558763288596184
# ap of each class: plane:0.7961611104230616, baseball-diamond:0.6085435034466244, bridge:0.4239515224564489,
# ground-track-field:0.603381356936641, small-vehicle:0.7160045371175391, large-vehicle:0.7304007567490252,
# ship:0.7817344679013655, tennis-court:0.8950621406156152, basketball-court:0.6891526986618293,
# storage-tank:0.7938219671895063, soccer-ball-field:0.4811753788741474, roundabout:0.5515250753194495,
# harbor:0.6180008404671178, swimming-poo0l:0.6668546458500035, helicopter:0.48237493088590266

# COCO style result:
# AP50: 0.6558763288596184
# AP75: 0.3893807978921721
# mAP: 0.3761706725334581

# 2*3
# mAP: 0.624576745275124
# ap of each class: plane:0.8423903518210242, baseball-diamond:0.573917309865648, bridge:0.3456663267018484,
# ground-track-field:0.44340752697206176, small-vehicle:0.6884250109125609, large-vehicle:0.7185376179350393,
# ship:0.7661357658932424, tennis-court:0.8870847865174986, basketball-court:0.6580122468660018,
# storage-tank:0.7271023754546506, soccer-ball-field:0.4598386257155087, roundabout:0.5878757977020009,
# harbor:0.5068206337751449, swimming-pool:0.663442886517719, helicopter:0.49999391647690844
# COCO style result:
# AP50: 0.624576745275124
# AP75: 0.3288853628106347
# mAP: 0.33796009619484824

# 2*2 add iof
# mAP: 0.6784332742865774
# ap of each class: plane:0.8532370142853193, baseball-diamond:0.6943663472245005, bridge:0.4302411481590787,
# ground-track-field:0.6182520789551005, small-vehicle:0.7207912309109321, large-vehicle:0.752356918795921,
# ship:0.7901608642935807, tennis-court:0.9059303417125092, basketball-court:0.7170918617402545,
# storage-tank:0.778005960720313, soccer-ball-field:0.4969478248918821, roundabout:0.5909025902329897,
# harbor:0.6322379682377117, swimming-pool:0.6612061807408649, helicopter:0.5347707833977008
# COCO style result:
# AP50: 0.6784332742865774
# AP75: 0.40871970107628136
# mAP: 0.39962992129983993
