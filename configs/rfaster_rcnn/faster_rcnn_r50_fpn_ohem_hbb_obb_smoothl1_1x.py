_base_ = ['./faster_rcnn_r50_fpn_hbb_obb_smoothl1_1x.py']
model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='ROHEMSampler'
            )
        )
    )
)

dataset_type = 'DOTADatasetV1'
data_root = '/data1/public_dataset/DOTA/DOTA1_0/simple/'
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
    dict(type='RLoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
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
        ann_file=data_root + val_ann_folder,
        img_prefix=data_root + val_img_folder,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test_img_folder,
        img_prefix=data_root + test_img_folder,
        pipeline=test_pipeline,
        test_mode=True))
work_dir = '/home/lzy/xyh/Netmodel/rotate_detection/checkpoints/simDOTA1_0/faster_rcnn_r50_fpn_ohem_1x'
# mAP: 0.6842872197679469
# ap of each class: plane:0.8120735439519112, baseball-diamond:0.7674378371475775, bridge:0.46665686742017304, ground-track-field:0.6985183825170451, small-vehicle:0.7169216139681184, large-vehicle:0.7241423113454417, ship:0.7572016751403186, tennis-court:0.9086682222798376, basketball-court:0.7986998756749295, storage-tank:0.7863210920163202, soccer-ball-field:0.5359777766539439, roundabout:0.5864470000299897, harbor:0.6669786032793019, swimming-pool:0.6101581757354116, helicopter:0.4281053193588842
# COCO style result:
#
# AP50: 0.6842872197679469
# AP75: 0.3745879432675384
# mAP: 0.3820845355995245
