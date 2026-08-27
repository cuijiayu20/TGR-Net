_base_ = './task1_mamba_kat_base.py'

# Formal WeatherOWOD Task 1 experiment, retaining the validated Exp F recipe.
# Detection sees five known class prompts. Unknown discovery uses the complete
# 500-attribute bank from OW-OVD, independent of num_train/test_classes.

prev_intro_cls = 0
cur_intro_cls = 5
num_classes = 5

work_dir = './work_dirs/task1_mamba_kat'

model = dict(
    num_train_classes=num_classes,
    num_test_classes=num_classes,
    alignment_loss=dict(
        type='ChannelWiseDivergenceLoss',
        temperature=1.0,
        loss_weight=10.0),
    det_loss_weights=dict(
        loss_cls=0.0,
        loss_bbox=0.1,
        loss_dfl=0.1),
    degradation_adapter=dict(gamma_init=0.1),
    bbox_head=dict(
        type='OurHead',
        att_embeddings='data/VOC2007/MOWOD/task_att_1_embeddings.pth',
        thr=0.55,
        alpha=0.2,
        use_sigmoid=True,
        prev_intro_cls=prev_intro_cls,
        cur_intro_cls=cur_intro_cls,
        head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

test_evaluator = dict(
    cfg=dict(
        prev_intro_cls=prev_intro_cls,
        cur_intro_cls=cur_intro_cls,
        unknown_id=20))

val_evaluator = test_evaluator
