_base_ = './local_task1_open_test_thr030_fog_rain_snow.py'

# Shared WeatherOWOD Task 1 training setup.
# Only the five known classes are provided as class prompts. The attribute
# bank remains class-agnostic and is configured by the experiment entrypoint.

data_root = '/data/weather/WeatherOWOD'
known_class_names = ['person', 'car', 'bus', 'bicycle', 'motorbike']
embedding_path = f'{data_root}/texts/weather_t1_embeddings.npy'
class_text_path = f'{data_root}/texts/task1_class_texts.json'
num_classes = len(known_class_names)

max_epochs = 12
train_batch_size_per_gpu = 16
base_lr = 5e-5
weight_decay = 0.05
save_epoch_intervals = 1
persistent_workers = True
num_workers = 8

paired_train_pipeline = [
    dict(
        type='RandomWeatherVariant',
        weather_names=('fog', 'rain', 'snow'),
        strict=True),
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts')),
    dict(type='LoadPairedCleanImage')
]

model = dict(
    num_train_classes=num_classes,
    num_test_classes=num_classes,
    embedding_path=embedding_path,
    alignment_loss=dict(
        type='ChannelWiseDivergenceLoss',
        temperature=1.0,
        loss_weight=0.1),
    det_loss_weights=dict(
        loss_cls=0.0,
        loss_bbox=1.0,
        loss_dfl=1.0),
    degradation_adapter=dict(
        gamma_init=0.1,
        use_mamba_ssm=False,
        kat_hidden_ratio=0.5),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        class_text_path=class_text_path,
        pipeline=paired_train_pipeline,
        dataset=dict(
            metainfo=dict(classes=known_class_names),
            data_root=data_root,
            ann_file='annotations/coco/task1/train_fog.json',
            data_prefix=dict(img=''))))

test_dataloader = dict(
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/coco/task1/test_fog_rain_snow_ow.json',
        data_prefix=dict(img='')))

test_evaluator = dict(
    cfg=dict(
        dataset_root=f'{data_root}/eval_voc/task1_test_fog_rain_snow',
        file_name='test_fog_rain_snow.txt',
        unknown_id=20))

val_dataloader = test_dataloader
val_evaluator = test_evaluator

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOWv5OptimizerConstructor')

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals,
        max_keep_ckpts=3,
        save_best='Unknown Recall50',
        rule='greater',
        type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'))

load_from = 'pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth'
resume = False
find_unused_parameters = True
