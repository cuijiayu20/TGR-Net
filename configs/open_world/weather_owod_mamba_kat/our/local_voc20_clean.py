_base_ = '../../weather_owod/our/local_voc20_clean.py'

adapter_in_channels = [256, 512, 512]

train_pipeline = [
    *_base_.train_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction', 'texts'))
]

model = dict(
    type='OurMambaKATDetector',
    alignment_loss=dict(
        type='ChannelWiseDivergenceLoss',
        temperature=1.0,
        loss_weight=0.1),
    freeze_base_detector=True,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    degradation_adapter=dict(
        type='MambaKATDegradationAdapter',
        in_channels=adapter_in_channels,
        gamma_init=0.1,
        use_mamba_ssm=True,
        kat_hidden_ratio=0.5))

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2e-4,
        weight_decay=0.05,
        batch_size_per_gpu=2),
    constructor='YOLOWv5OptimizerConstructor')

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
