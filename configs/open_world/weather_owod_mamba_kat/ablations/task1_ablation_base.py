_base_ = '../our/train_task1_mamba_kat.py'

# Experiment 7 uses the full Task 1 training set. The combined synthetic
# fog/rain/snow test set is used as the common validation set for every run.
max_epochs = 6
randomness = dict(seed=3407, diff_rank_seed=False, deterministic=False)

model = dict(
    pipline=[],
    alignment_loss=dict(
        type='ChannelWiseDivergenceLoss',
        temperature=1.0,
        loss_weight=10.0),
    det_loss_weights=dict(
        loss_cls=0.0,
        loss_bbox=0.1,
        loss_dfl=0.1),
    degradation_adapter=dict(
        gamma_init=0.1,
        use_mamba_ssm=False,
        kat_hidden_ratio=0.5,
        use_mamba_branch=True,
        use_kat_branch=True,
        use_conv_branch=True))

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=max_epochs)

custom_hooks = []

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=max_epochs,
        save_best='Known AP50',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=500))

load_from = (
    'pretrained_models/'
    'yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth')
resume = False
find_unused_parameters = True
