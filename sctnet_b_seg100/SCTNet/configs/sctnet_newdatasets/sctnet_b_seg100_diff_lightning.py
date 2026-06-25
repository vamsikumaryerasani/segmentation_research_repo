_base_ = ['../sctnet_railsem/sctnet_b_seg100_rs19_aug_85_10_5.py']

dataset_type = 'CustomDataset'
data_root = '/data/pool/qmc-41b/dataset_diff_lightning_conditions'
log_level = 'INFO'
freeze_bn = True

classes = ('sky', 'terrain', 'nature', 'car', 'building', 'railway')
palette = [
    [0, 0, 0],
    [128, 64, 128],
    [107, 142, 35],
    [0, 0, 142],
    [70, 70, 70],
    [153, 153, 153],
]

val_loss_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1920, 1080), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='splits/train.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='splits/val.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        split='splits/test.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
    ),
)

runner = dict(type='EpochBasedRunner', max_epochs=150)

checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=200,
)

evaluation = dict(
    interval=1,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU',
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
    ]
)

custom_hooks = [
    dict(
        type='ValLossEarlyStopHook',
        dataset_cfg=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images',
            ann_dir='masks',
            split='splits/val.txt',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=classes,
            palette=palette,
            pipeline=val_loss_pipeline,
        ),
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='val_loss',
        rule='less',
        patience=10,
        min_delta=0.0,
    ),
]
