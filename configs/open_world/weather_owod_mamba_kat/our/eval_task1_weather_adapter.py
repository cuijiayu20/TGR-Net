_base_ = './train_task1_mamba_kat.py'

# Branch-enabled counterpart to eval_task1_no_adapter.py. Use this entry with
# a trained Mamba-KAT checkpoint to measure the adverse-weather branch gain.
model = dict(degradation_adapter=dict(bypass=False))
