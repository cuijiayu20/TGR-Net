# Mamba-KAT Degradation Adapter

This project extension trains a feature adapter for adverse-weather
open-vocabulary object detection while keeping the original YOLO-World model
frozen.

## Environment

Use the isolated conda environment only:

```powershell
conda activate mamba_kat
python -c "import sys; print(sys.executable)"
```

The printed path must be under:

```text
D:\appinstall\anaconda2\envs\mamba_kat
```

Do not install dependencies or run training from `base`.

## Paired Input Contract

The degradation code is expected to run before collation and add these fields to
each sample:

```python
sample['clean_inputs'] = clean_tensor
sample['degraded_inputs'] = degraded_tensor
sample['inputs'] = degraded_tensor
```

The detection annotation remains shared through `sample['data_samples']`.
`yolow_collate` will stack `clean_inputs` and `degraded_inputs` into:

```python
batch['inputs'] = {
    'inputs': degraded_batch,
    'clean_inputs': clean_batch,
    'degraded_inputs': degraded_batch,
}
```

At test time, a plain tensor input is still accepted and is treated as degraded
input.

KADet-derived offline weather synthesis and VOC paired-manifest tools are in:

```text
tools/adverse_weather/
```

Use these tools to prepare rain, snow, and fog paired records before writing a
dataset/transform that emits the fields above.

## Training Objective

`YOLOWorldMambaKATDetector` computes:

```text
L_total = L_det + lambda_align * L_align
L_align = SmoothL1(normalize(Fe), stopgrad(normalize(Fc)))
```

Only `degradation_adapter` parameters are trainable. The YOLO-World backbone,
neck, text encoder, and detection head are frozen.
