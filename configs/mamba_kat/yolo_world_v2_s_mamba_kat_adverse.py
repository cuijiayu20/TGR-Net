_base_ = '../finetune_coco/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py'

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# The user-provided degradation pipeline should emit paired tensors named
# clean_inputs and degraded_inputs. This config only consumes those tensors.
cwd_loss_weight = 0.1
adapter_in_channels = [128, 256, 512]

model = dict(
    type='YOLOWorldMambaKATDetector',
    alignment_loss=dict(
        type='ChannelWiseDivergenceLoss',
        temperature=1.0,
        loss_weight=cwd_loss_weight),
    freeze_yolo_world=True,
    degradation_adapter=dict(
        type='MambaKATDegradationAdapter',
        in_channels=adapter_in_channels,
        gamma_init=0.1,
        use_mamba_ssm=True,
        kat_hidden_ratio=0.5))

train_dataloader = dict(
    collate_fn=dict(type='yolow_collate', use_ms_training=False))

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2e-4,
        weight_decay=0.05,
        batch_size_per_gpu=16),
    constructor='YOLOWv5OptimizerConstructor')
