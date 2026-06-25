# configs/sctnet_railsem/sctnet_b_seg100_railsem90_10_400e.py

# -------------------------
# Dataset settings
# -------------------------
dataset_type = 'CustomDataset'
data_root = '/data/pool/qmc-41b/rs19_val_aug_bri_0p5_0p6_0p7_con_0p6_0p7'

# RailSem masks: labels 0..18, ignore=255 (your masks_new were like that; this split preserves them)
classes = [str(i) for i in range(19)]
palette = [[i * 10, i * 5, i * 12] for i in range(19)]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='Resize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

# NOTE: single GPU -> keep batch size reasonable.
# If you hit CUDA OOM, drop samples_per_gpu to 4 or 2.
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,

    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images_new/train',
        ann_dir='masks_new/train',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
    ),

    # IMPORTANT: val uses test_pipeline (MultiScaleFlipAug) so img becomes a LIST
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images_new/val',
        ann_dir='masks_new/val',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
    ),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images_new/test',
        ann_dir='masks_new/test',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
    ),
)
# -------------------------
# Model settings
# -------------------------
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SCTNet',
        base_channels=64,
        init_cfg=None
    ),
    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=128,
        num_classes=19,

        # CRITICAL FIX: pick the tensor feature with 256 channels (outs[0])
        in_index=0,
        input_transform=None
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# -------------------------
# Training settings
# -------------------------
log_level = 'INFO'
workflow = [('train', 1)]

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(type='OptimizerHook', grad_clip=None)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)

# REQUIRED: 400 epochs
runner = dict(type='EpochBasedRunner', max_epochs=150)

# REQUIRED: validate every epoch
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

# Checkpointing (adjust if you want every epoch)
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)

# Required by MMCV runtime
resume_from = None
auto_resume = False
load_from = None

# -------------------------
# Early stopping on VAL LOSS (patience=3, interval=1)
# Uses your custom hook: tools/val_loss_early_stop_hook.py
# -------------------------
custom_imports = dict(
    imports=[
        'tools.val_loss_early_stop_hook',
        'tools.early_stopping_hook',
    ],
    allow_failed_imports=False)


# Build a GT-enabled val-loss dataset cfg (MUST include LoadAnnotations).
# We use the VAL split, but a deterministic pipeline (flip prob=0).
val_loss_data = dict(**data['val'])
val_loss_data['test_mode'] = False
val_loss_data['pipeline'] = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

custom_hooks = [
    dict(
        type='ValLossEarlyStopHook',
        dataset_cfg=val_loss_data,
        samples_per_gpu=32,
        workers_per_gpu=data.get('workers_per_gpu', 4),
        interval=1,
        num_batches=32,  # set 200 if you want faster
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='val.loss',
        rule='less',
        patience=3,
        min_delta=0.0,
        start_epoch=1,
        interval=1,
    ),
]

