_base_ = './train_task1_mamba_kat.py'

# Formal no-adapter baseline. This is the configuration that produced
# Unknown Recall50 = 40.5966 on the combined fog/rain/snow Task 1 test set.
model = dict(degradation_adapter=dict(bypass=True))
