_base_ = './local_task1_open_test_thr030_fog.py'

# Default local training entry for the proposed OW-OVD + Mamba-KAT adapter.
# This config uses synthetic fog as degraded training input and evaluates on
# the Task1 fog split with score_thr=0.30 and unknown_id=20.

max_epochs = 12
train_batch_size_per_gpu = 1
base_lr = 5e-5
save_epoch_intervals = 1
persistent_workers = False
work_dir = './work_dirs/weather_owod_mamba_kat_task1_fog'

model = dict(
    alignment_loss=None,
    degradation_adapter=dict(
        gamma_init=0.1,
        use_mamba_ssm=False))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=0,
    persistent_workers=persistent_workers,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/coco/task1/train_fog.json',
            data_root='/data/weather/WeatherOWOD',
            data_prefix=dict(img=''))))

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOWv5OptimizerConstructor')

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals,
        max_keep_ckpts=2,
        save_best='Unknown Recall50',
        rule='greater',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'))

load_from = 'pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth'
resume = False
find_unused_parameters = True
