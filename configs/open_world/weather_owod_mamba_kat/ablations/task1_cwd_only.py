_base_ = './task1_ablation_base.py'

work_dir = './work_dirs/experiment7/task1_cwd_only'

model = dict(
    det_loss_weights=dict(
        loss_cls=0.0,
        loss_bbox=0.0,
        loss_dfl=0.0))
