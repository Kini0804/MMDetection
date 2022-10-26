_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

img_scale = (960, 960)  # height, width

model = dict(
    type='YOLOF',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True),
        # style='caffe',
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8]),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=6,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=100))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
dataset_type = 'VOCDataset'
data_root = 'data/SeaShips/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))


# optimizer

# optimizer
# optimizer = dict(
#     type='SGD',
#     lr=0.12,
#     momentum=0.9,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         norm_decay_mult=0., custom_keys={'backbone': dict(lr_mult=1. / 3)}))
# lr_config = dict(warmup_iters=1500, warmup_ratio=0.00066667)
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

max_epochs = 50
num_last_epochs = 5
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(interval=100)

evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP')

checkpoint_config = dict(interval=interval)

# resume_from = 'work_dirs/my_yolov5/latest.pth'
custom_hooks = [
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
# optimizer = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)
#
# max_epochs = 50
# num_last_epochs = 5
# resume_from = None
# interval = 10
#
# # learning policy
# lr_config = dict(
#     _delete_=True,
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=5,  # 5 epoch
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.05)
#
# runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
#
# custom_hooks = [
#     dict(
#         type='YOLOXModeSwitchHook',
#         num_last_epochs=num_last_epochs,
#         priority=48),
#     dict(
#         type='SyncNormHook',
#         num_last_epochs=num_last_epochs,
#         interval=interval,
#         priority=48),
#     dict(
#         type='ExpMomentumEMAHook',
#         resume_from=resume_from,
#         momentum=0.0001,
#         priority=49)
# ]
# checkpoint_config = dict(interval=interval)
# evaluation = dict(
#     save_best='auto',
#     # The evaluation interval is 'interval' when running epoch is
#     # less than ‘max_epochs - num_last_epochs’.
#     # The evaluation interval is 1 when running epoch is greater than
#     # or equal to ‘max_epochs - num_last_epochs’.
#     interval=interval,
#     dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
#     metric='mAP')
# log_config = dict(interval=100)
#
# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (8 GPUs) x (8 samples per GPU)
# auto_scale_lr = dict(base_batch_size=64)