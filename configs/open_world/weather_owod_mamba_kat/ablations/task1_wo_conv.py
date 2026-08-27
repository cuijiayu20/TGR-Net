_base_ = './task1_ablation_base.py'

work_dir = './work_dirs/experiment7/task1_wo_conv'

model = dict(
    degradation_adapter=dict(use_conv_branch=False))
