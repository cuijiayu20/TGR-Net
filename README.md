# TGR-Net: Source Code and Reproduction Package

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21888531.svg)](https://doi.org/10.5281/zenodo.21888531)
[![DriverWeather DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22121743.svg)](https://doi.org/10.5281/zenodo.22121743)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Overview

This repository contains the TGR-Net source-code overlay, configurations, environment records, lightweight assets, tests, and machine-readable result summaries used for the adverse-weather open-vocabulary open-world object detection experiments reported in the associated manuscript.

TGR-Net was developed internally under the name `Mamba-KAT`. Legacy class names, configuration paths, test names, and result labels retain that identifier so the released files remain traceable to the evaluated implementation. This is a **reproduction overlay**, not a standalone package; it must be copied into the compatible YOLO-World base repository described below.

## Public Records

| Resource | Persistent identifier |
| --- | --- |
| Source repository | <https://github.com/cuijiayu20/TGR-Net> |
| Software archive, v1.0.0 | <https://doi.org/10.5281/zenodo.21888532> |
| Software concept DOI (all versions) | <https://doi.org/10.5281/zenodo.21888531> |
| DriverWeather test dataset, v1.0.0 | <https://doi.org/10.5281/zenodo.22121743> |
| DriverWeather concept DOI (all versions) | <https://doi.org/10.5281/zenodo.22121742> |
| VOC-FOG, VOC-Rain, VOC-Snow, and synthesis materials | <https://doi.org/10.21227/8g5n-b326> |

For an exact software citation, cite the version DOI. Use the concept DOI when referring to the project across versions.

## Reproduction Scope

The release provides implementation and experiment entry points for inspecting, training, evaluating, and analyzing TGR-Net, plus aggregate result files for checking the manuscript tables.

It does not redistribute the full third-party YOLO-World, VOC, or RTTS resources; the large YOLO-World pretrained checkpoint; a trained TGR-Net checkpoint; or full per-sample prediction logs. Evaluation therefore requires a checkpoint produced by the training procedure below or another compatible checkpoint supplied by the user.

## Package Layout

```text
TGR-Net_Source_Code_and_Reproduction_Package_v1.0.0/
├── code_overlay/                 # TGR-Net/OW-OVD code and tools copied into YOLO-World
├── configs/                      # Training, evaluation, and ablation configurations
├── checkpoints/                  # Target location for user-produced checkpoints
├── data_assets/                  # Lightweight attribute and text assets
├── docs/                         # Supporting documentation
├── env/                          # Conda and pip environment records
├── results/                      # Manuscript table data and semantic summaries
├── tests/                        # Adapter, loss, detector, and config smoke tests
├── CITATION.cff                  # Machine-readable citation metadata
├── LICENSE                       # GPL-3.0-only license
└── MANIFEST.files.txt            # Complete archived file list
```

## Compatible Base Code

- YOLO-World repository: <https://github.com/AILab-CVC/YOLO-World>
- reference commit: `b449b98202e931590513c16e4830318be2dde946`
- MMYOLO submodule revision recorded by that commit: `4d97b3a06609dba94b8ec584be2f2029cfdb7519`

```bash
git clone https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World
git checkout b449b98202e931590513c16e4830318be2dde946
git submodule update --init --recursive
```

The overlay supplies the TGR-Net and open-world additions referenced by the released configurations, including the adapter, detector/head extensions, channel-wise divergence (CWD) loss, evaluator, and WeatherOWOD utilities.

## Installation and Requirements

From the cloned YOLO-World root, replace `/path/to/TGR-Net_release` with this repository:

```bash
cp -R /path/to/TGR-Net_release/code_overlay/. .
cp -R /path/to/TGR-Net_release/configs/. configs/
mkdir -p data
cp -R /path/to/TGR-Net_release/data_assets/data/. data/
```

The recorded environment used Python 3.10, PyTorch 2.0.1 with CUDA 11.8, MMCV 2.0.0, MMDetection 3.0.0, MMEngine 0.10.3, and MMYOLO 0.6.0.

```bash
conda env create -f /path/to/TGR-Net_release/env/ow_ovd_environment.yml
conda activate ow_ovd
pip install -v -e .
```

`env/ow_ovd_requirements_freeze.txt` records the full original environment. If it conflicts with local CUDA or hardware, use the Conda file and the versions above as the compatibility reference.

Download the public [YOLO-World V2-L pretrained checkpoint](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth) and place it at:

```text
pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth
```

## Dataset Information

### WeatherOWOD training and benchmark data

The protocol uses VOC-derived clean/adverse-weather pairs and the five Task 1 known classes `person`, `car`, `bus`, `bicycle`, and `motorbike`. Public VOC-FOG, VOC-Rain, VOC-Snow, and synthesis materials are available at <https://doi.org/10.21227/8g5n-b326>. VOC and RTTS source data are not redistributed; users must obtain them from their providers and comply with their terms.

The default configurations expect:

```text
/data/weather/WeatherOWOD/
├── annotations/coco/task1/
│   ├── train_fog.json
│   ├── train_rain.json
│   ├── train_snow.json
│   └── test_fog_rain_snow_ow.json
├── eval_voc/
│   ├── task1_test_clean/
│   ├── task1_test_fog/
│   ├── task1_test_rain/
│   ├── task1_test_snow/
│   ├── task1_test_fog_rain_snow/
│   └── task1_test_rtts/
└── texts/
    ├── weather_t1_embeddings.npy
    ├── task1_class_texts.json
    ├── weather_voc20_embeddings.npy
    └── voc20_class_texts.json
```

If the root differs, edit `data_root` in `configs/open_world/weather_owod_mamba_kat/our/task1_mamba_kat_base.py` and the relevant evaluation `dataset_root` values. Dataset-building help:

```bash
python tools/weather_owod/build_weather_owod.py --help
python tools/adverse_weather/build_voc_weather_pairs.py --help
```

### DriverWeather independent test set

[DriverWeather v1.0.0](https://doi.org/10.5281/zenodo.22121743) is publicly available under CC BY 4.0. It contains 473 images at 3840 × 2160 pixels and 473 paired LabelMe-compatible JSON annotations: 150 clean, 30 fog, and 293 rain images. The five annotated classes are `person`, `car`, `bus`, `bicycle`, and `motorbike`.

DriverWeather is a **pure test dataset**. It must not be used for training, validation, model selection, threshold selection, or hyperparameter tuning.

## Methodology and Reproduction Workflow

1. Clone and pin the compatible YOLO-World revision.
2. Apply the code/configuration overlay and lightweight assets.
3. Recreate the recorded software environment.
4. Obtain the third-party data and public synthesis resources, then build WeatherOWOD.
5. Download the YOLO-World V2-L pretrained checkpoint.
6. Train the formal Task 1 configuration or a specified ablation.
7. Evaluate the trained checkpoint on the predefined conditions without tuning on DriverWeather.
8. Run the analysis scripts and compare outputs with `results/`.

In the formal recipe, the base detector is frozen and only the adverse-weather feature adapter is trained. The Task 1 classification-loss weight is zero. During training, CWD uses the paired clean image as a detached clean-response regularization reference; inference uses only the degraded/test image.

## Training

```bash
python tools/train.py \
  configs/open_world/weather_owod_mamba_kat/our/train_task1_mamba_kat.py
```

Single-weather variants:

```bash
python tools/train.py configs/open_world/weather_owod_mamba_kat/our/train_task1_fog.py
python tools/train.py configs/open_world/weather_owod_mamba_kat/our/train_task1_rain.py
python tools/train.py configs/open_world/weather_owod_mamba_kat/our/train_task1_snow.py
```

Ablation configurations are under `configs/open_world/weather_owod_mamba_kat/ablations/`.

## Evaluation

Place a trained checkpoint at `checkpoints/mambakat_task1_full.pth` or replace the path:

```bash
python tools/test.py \
  configs/open_world/weather_owod_mamba_kat/our/eval_task1_weather_adapter.py \
  checkpoints/mambakat_task1_full.pth

python tools/test.py \
  configs/open_world/weather_owod_mamba_kat/our/eval_task1_no_adapter.py \
  checkpoints/mambakat_task1_full.pth
```

Condition-specific configurations for clean, fog, rain, snow, mixed weather, Foggy Driving, and RTTS are under `configs/open_world/weather_owod_mamba_kat/our/`.

## Analysis and Result Files

The `results/` directory contains aggregate machine-readable data for manuscript-table checking:

| Purpose | File |
| --- | --- |
| Main results | `results/paper_tables/data/main_mambakat_full_results.csv` |
| Strict baseline | `results/paper_tables/data/strict_baseline_results.csv` |
| Module/loss ablations | `module_ablation_results.csv`, `loss_ablation_results.csv` |
| Adapter/branch controls | `adapter_bypass_results.csv`, `branch_contribution_results.csv` |
| Causal-intervention summaries | `causal_intervention_results.csv` |
| Semantic summaries | `results/semantic_match_stats/full_all/` |

```bash
python tools/analysis/run_semantic_match_full.py --help
python tools/analysis/open_vocab_add_class_probe.py --help
```

These CSV/JSON summaries are not substitutes for raw data, trained weights, or full prediction logs.

## Verification

After applying the overlay:

```bash
pytest tests/test_mamba_kat_adapter.py
pytest tests/test_channel_wise_divergence_loss.py
pytest tests/test_mamba_kat_detector_smoke.py
pytest tests/test_task1_ablation_configs.py
```

These are lightweight code/configuration checks; they do not reproduce full training or benchmark metrics.

## Citation

```bibtex
@software{cui_2026_tgr_net,
  author    = {Cui, Jiayu},
  title     = {TGR-Net Source Code and Reproduction Package},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21888532},
  url       = {https://doi.org/10.5281/zenodo.21888532}
}
```

```bibtex
@dataset{cuijiayu_2026_driverweather,
  author    = {cuijiayu},
  title     = {DriverWeather Test Dataset},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22121743},
  url       = {https://doi.org/10.5281/zenodo.22121743}
}
```

Also cite the associated manuscript and the original YOLO-World, MMDetection, MMYOLO, VOC, RTTS, and other resources as applicable.

## License

This release is distributed under GNU GPL v3.0 only (`GPL-3.0-only`); see [LICENSE](LICENSE). Third-party code, models, and datasets remain subject to their original licenses and terms. DriverWeather is released separately under CC BY 4.0.

## Contributions and Questions

Open reproducibility questions or corrections at <https://github.com/cuijiayu20/TGR-Net/issues>. Focused pull requests are welcome; do not include third-party data or files the contributor is not authorized to redistribute.
