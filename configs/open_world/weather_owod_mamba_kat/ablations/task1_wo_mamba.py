_base_ = './task1_ablation_base.py'

work_dir = './work_dirs/experiment7/task1_wo_mamba'

model = dict(
    degradation_adapter=dict(use_mamba_branch=False))
