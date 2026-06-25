# =========================================================
# SCTNet-B Seg75 - RailSem19 Augmented - 85/10/5 split
# No pretrained, No teacher/guidance, Epoch-based, EarlyStop=5
# Compatible with mmcv==1.6.0, mmseg==0.26.0 (NO _delete_)
# =========================================================

# -------------------------
# Dataset
# -------------------------
dataset_type = 'CustomDataset'
data_root = 'data/railsem19_mmseg/'

classes = [str(i) for i in range(19)]
palette = [
    [0, 0, 0], [10, 5, 12], [20, 10, 24], [30, 15, 36], [40, 20, 48],
    [50, 25, 60], [60, 30, 72], [70, 35, 84], [80, 40, 96],
    [90, 45, 108], [100, 50, 120], [110, 55, 132], [120, 60, 144],
    [130, 65, 156], [140, 70, 168], [150, 75, 180], [160, 80, 192],
    [170, 85, 204], [180, 90, 216]
]

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
        ],
    )
]

data = dict(
    samples_per_gpu=32,       # adjust if GPU OOM
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
# Model (NO pretrained / NO teacher)
# -------------------------
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    # Keep distill wrapper because SCT repo sometimes returns tuple from head;
    # but we disable teacher/guidance via auxiliary_head=None.
    type='EncoderDecoder_Distill',

    backbone=dict(
        type='SCTNet',
        base_channels=64,
        # IMPORTANT: no checkpoint key here
        init_cfg=dict(type='Kaiming', layer='Conv2d'),
        pretrained=None,
    ),

    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=128,
        num_classes=19,
    ),

    # Disable teacher/guidance head completely
    auxiliary_head=None,

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# -------------------------
# Optimizer / LR
# -------------------------
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)

# Epoch-based training (IMPORTANT: no max_iters anywhere)
runner = dict(type='EpochBasedRunner', max_epochs=400)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)

# -------------------------
# Logging / Checkpoints / Eval
# -------------------------
log_level = 'INFO'

checkpoint_config = dict(by_epoch=True, interval=10, max_keep_ckpts=5)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU', by_epoch=True)

log_config = dict(
    interval=50,  # IMPORTANT: when by_epoch=True, interval is in *epochs*
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook', by_epoch=True),
    ])

workflow = [('train', 1)]
resume_from = None
auto_resume = False
load_from = None
dist_params = dict(backend='nccl')
gpu_ids = [0]

# -------------------------
# Custom hooks: val-loss + early stopping
# (Only works if these python files exist in your repo)
# -------------------------
custom_imports = dict(
    imports=[
        'projects.railsem_hooks.early_stopping_hook',
        'projects.railsem_hooks.val_loss_hook',
        'projects.railsem_hooks.jsonl_logger_hook',
    ],
    allow_failed_imports=False,
)

# A separate dataset cfg for ValLossHook (needs annotations, not MultiScaleFlipAug)
val_loss_data = dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='images_new/val',
    ann_dir='masks_new/val',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1280, 720)),
        dict(type='RandomFlip', prob=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ],
    classes=classes,
    palette=palette,
    test_mode=False,
)

custom_hooks = [
    dict(
        type='ValLossHook',
        dataset_cfg=val_loss_data,
        samples_per_gpu=1,
        workers_per_gpu=2,
        interval=1,
        num_batches=20,
        priority='LOW',
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='val.loss',
        rule='less',
        patience=5,         # <- you asked for 5
        min_delta=0.0,
        start_epoch=1,
        priority='LOW',
    ),
]

# -------------------------
# Work dir (store under /data/pool/qmc-41b/)
# -------------------------
work_dir = '/data/pool/qmc-41b/work_dirs/sctnet_b_seg75_rs19_aug_85_10_5_epoch400_es_valloss'
