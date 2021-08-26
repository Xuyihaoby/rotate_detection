merge_nms_iou_thr_dict = {
    'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.1,
    'soccer-ball-field': 0.3, 'small-vehicle': 0.05, 'ship': 0.05, 'plane': 0.3,
    'large-vehicle': 0.05, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
    'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3
}
diff_r_max_num = {'roundabout': 100, 'tennis-court': 100, 'swimming-pool': 200, 'storage-tank': 200,
                  'soccer-ball-field': 100, 'small-vehicle': 500, 'ship': 500, 'plane': 300,
                  'large-vehicle': 500, 'helicopter': 100, 'harbor': 200, 'ground-track-field': 100,
                  'bridge': 100, 'basketball-court': 100, 'baseball-diamond': 100, 'container-crane': 100}
merge_nms_iou_thr_dict_h = {'roundabout': 0.35, 'tennis-court': 0.35, 'swimming-pool': 0.4, 'storage-tank': 0.3,
                            'soccer-ball-field': 0.3, 'small-vehicle': 0.4, 'ship': 0.35, 'plane': 0.35,
                            'large-vehicle': 0.4, 'helicopter': 0.4, 'harbor': 0.3, 'ground-track-field': 0.4,
                            'bridge': 0.3, 'basketball-court': 0.4, 'baseball-diamond': 0.3, 'container-crane': 0.3}

diff_h_max_num = {'roundabout': 100, 'tennis-court': 100, 'swimming-pool': 200, 'storage-tank': 200,
                  'soccer-ball-field': 100, 'small-vehicle': 500, 'ship': 500, 'plane': 300,
                  'large-vehicle': 500, 'helicopter': 100, 'harbor': 200, 'ground-track-field': 100,
                  'bridge': 100, 'basketball-court': 100, 'baseball-diamond': 100, 'container-crane': 100}

model = dict(
    type='FeatureAttenNetAllLvlSingleMaskDOTA',
    obb=False,
    submission=False,
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
        num_outs=5),
    rpn_head=dict(
        type='RPNHeadSingleMaskDotDOTA',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        loss_mask=dict(type='FocalLoss')),
    roi_head=dict(
        type='HSPRoIHead',
        bbox_roi_extractor=dict(
            type='MultiRoIWithOriginalImageSingleMaskExtractorDOTA',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='MultiLvlsWithOriginalImageSingleMaskShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            bbox_coder_r=dict(
                type='DeltaXYWHBThetaBoxCoder',
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                gpu_assign_thr=20),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=20),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms_h=dict(type='nms', iou_threshold=merge_nms_iou_thr_dict_h),
            max_per_img_h=diff_h_max_num,
            # nms_r=dict(type='nms_rotated', iou_threshold=0.5),
            nms_r=dict(type='rnms', iou_threshold=merge_nms_iou_thr_dict),
            max_per_img=diff_r_max_num)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

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
data_root = '/data/xuyihao/mmdetection/dataset/DOTA/'
trainsplit_ann_folder = 'trainall/labelTxt'
trainsplit_img_folder = 'trainall/images'
valsplit_ann_folder = 'valsplit/labelTxt'
valsplit_img_folder = 'valsplit/images'
val_ann_folder = 'val/labelTxt'
val_img_folder = 'val/images'
test_img_folder = 'testmulsplit/images'
example_ann_folder = 'examplesplit/labelTxt'
example_img_folder = 'examplesplit/images'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RLoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'hor_gt_bboxes', 'hor_gt_bboxes_ignore',
                               'gt_labels', 'gt_masks']),
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
    workers_per_gpu=0,
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
# yapf:disable
log_config = dict(
    interval=50,
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
work_dir = '/data/xuyihao/mmdetection/configs/HSP/wor_dir/' + 'feat_atten_net_resnet50_1x_dotav1_ms_new'
