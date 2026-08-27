_base_ = './task1_ablation_base.py'

work_dir = './work_dirs/experiment7/task1_wo_kat'

model = dict(
    degradation_adapter=dict(use_kat_branch=False))
