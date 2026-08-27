# TGR-Net Source Code and Reproduction Package

This package contains the TGR-Net code, configuration files, environment files, lightweight assets, and result summaries needed to reproduce the adverse-weather open-vocabulary open-world detection experiments reported in the associated manuscript.

It is intended to be overlaid on the full YOLO-World / OW-OVD codebase, not used as a standalone repository.

The implementation was developed under the internal name `Mamba-KAT`. Legacy class names, configuration paths, test names, and result labels retain that identifier so that the released code remains identical to the implementation used for the reported experiments. In the manuscript and this release, the complete method is referred to as TGR-Net.

## Package Layout

```text
TGR-Net_Source_Code_and_Reproduction_Package_v1.0.0/
├── code_overlay/                 # Files to copy into the base codebase
│   ├── yolo_world/               # TGR-Net implementation (legacy Mamba-KAT identifiers retained)
│   └── tools/                    # Training/testing, WeatherOWOD builders, analysis scripts
├── configs/
│   ├── open_world/weather_owod_mamba_kat/
│   └── mamba_kat/
├── checkpoints/                  # Optional trained TGR-Net checkpoint location
├── data_assets/                  # Lightweight OW-OVD attribute/text assets
├── env/                          # conda / pip freeze from ow_ovd environment
├── results/                      # Table CSVs and semantic matching summaries
├── tests/                        # Adapter/config smoke tests
└── MANIFEST.files.txt
```

## Base Code Required

Use the original YOLO-World / OW-OVD project as the base repository. The reproduction needs these base components:

1. YOLO-World V2 detector implementation and MMYOLO/MMDetection training stack.
2. OW-OVD custom open-world detection pieces, especially:
   - `OurDetector` / `OurHead` style detector and head;
   - attribute embedding based unknown branch;
   - OWOD evaluator for known / unknown / A-OSE metrics;
   - Task 1 protocol using five known classes: `person`, `car`, `bus`, `bicycle`, `motorbike`.
3. Pretrained base detector checkpoint:
   - expected config path: `pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth`
   - this large file is not included in this package.
4. Dataset root expected by configs:
   - default: `/data/weather/WeatherOWOD`
   - edit `data_root` in `configs/open_world/weather_owod_mamba_kat/our/task1_mamba_kat_base.py` if your path differs.

## Overlay Installation

From the base repository root:

```bash
cp -R /path/to/TGR-Net_Source_Code_and_Reproduction_Package_v1.0.0/code_overlay/* .
cp -R /path/to/TGR-Net_Source_Code_and_Reproduction_Package_v1.0.0/configs/* configs/
cp -R /path/to/TGR-Net_Source_Code_and_Reproduction_Package_v1.0.0/data_assets/data/* data/
```

Then check that these files exist in the base repo:

```text
yolo_world/models/adapters/mamba_kat_adapter.py
yolo_world/models/detectors/yolo_world_mamba_kat.py
yolo_world/models/detectors/our_mamba_kat.py
yolo_world/models/losses/channel_wise_divergence_loss.py
yolo_world/datasets/transformers/mm_transforms.py
yolo_world/evaluator/OW_evaluator.py
configs/open_world/weather_owod_mamba_kat/our/train_task1_mamba_kat.py
```

## Environment

The original training environment was named `ow_ovd`.

```bash
conda env create -f env/ow_ovd_environment.yml
conda activate ow_ovd
pip install -r env/ow_ovd_requirements_freeze.txt
pip install -v -e .
```

If the frozen requirements conflict with your CUDA/PyTorch version, use `ow_ovd_environment.yml` as the primary reference and install the matching PyTorch/MMCV/MMEngine/MMDetection/MMYOLO stack manually.

## Data Needed

The configs assume the following WeatherOWOD layout:

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
├── texts/
│   ├── weather_t1_embeddings.npy
│   ├── task1_class_texts.json
│   ├── weather_voc20_embeddings.npy
│   └── voc20_class_texts.json
└── images / VOC-style split files
```

The package includes lightweight OW-OVD assets under `data_assets/data/VOC2007/MOWOD/`, including `task_att_1_embeddings.pth`. It does not redistribute the full VOC, RTTS, or generated weather image data. The public adverse-weather datasets and synthesis materials used by this study are available from IEEE DataPort at <https://doi.org/10.21227/8g5n-b326>. Users remain responsible for complying with the licenses and terms of the original third-party datasets.

DriverWeather contains road-scene images with potentially identifiable information and is not included in this public code release. Its access restriction and the available de-identified research materials are described in the associated manuscript.

Use the dataset tools if you need to rebuild the protocol:

```bash
python code_overlay/tools/weather_owod/build_weather_owod.py --help
python code_overlay/tools/adverse_weather/build_voc_weather_pairs.py --help
```

## Checkpoints

Trained model checkpoints are not included in this source-code release. The provided configurations, scripts, and machine-readable result summaries correspond to the reported experiments.

Place the trained branch/full MMEngine checkpoint here:

```text
checkpoints/mambakat_task1_full.pth
```

For evaluation, either pass the real checkpoint path directly or copy it into the base repository, for example:

```bash
mkdir -p checkpoints
cp /path/to/mambakat_task1_full.pth checkpoints/
```

The base YOLO-World pretrained checkpoint is separate and should be placed at:

```text
pretrained_models/yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth
```

## Training

Main Task 1 training:

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

Ablation configs are under:

```text
configs/open_world/weather_owod_mamba_kat/ablations/
```

## Evaluation

TGR-Net adapter enabled (legacy configuration path retained):

```bash
python tools/test.py \
  configs/open_world/weather_owod_mamba_kat/our/eval_task1_weather_adapter.py \
  checkpoints/mambakat_task1_full.pth
```

Adapter bypass / OW-OVD baseline under the same checkpoint:

```bash
python tools/test.py \
  configs/open_world/weather_owod_mamba_kat/our/eval_task1_no_adapter.py \
  checkpoints/mambakat_task1_full.pth
```

Weather-specific test configs:

```text
configs/open_world/weather_owod_mamba_kat/our/local_task1_open_test_thr030_clean.py
configs/open_world/weather_owod_mamba_kat/our/local_task1_open_test_thr030_fog.py
configs/open_world/weather_owod_mamba_kat/our/local_task1_open_test_thr030_rain.py
configs/open_world/weather_owod_mamba_kat/our/local_task1_open_test_thr030_snow.py
configs/open_world/weather_owod_mamba_kat/our/local_task1_open_test_thr030_rtts.py
```

## Analysis Scripts

Semantic matching statistics:

```bash
python tools/analysis/run_semantic_match_full.py --help
```

Open-vocabulary added-class probe:

```bash
python tools/analysis/open_vocab_add_class_probe.py --help
```

The package includes result summaries under `results/` for paper table checking, but not full per-sample logs except selected aggregate files.

## Smoke Tests

After overlaying the package:

```bash
pytest tests/test_mamba_kat_adapter.py
pytest tests/test_channel_wise_divergence_loss.py
pytest tests/test_task1_ablation_configs.py
```

## Notes

- TGR-Net freezes the base OV-OW detector and trains only the adverse-weather feature adapter.
- Classification loss is set to zero in the formal Task 1 recipe to avoid reshaping open-vocabulary semantic boundaries with five known classes only.
- CWD uses the paired clean image only as a detached clean-response regularization reference during training; inference uses only the degraded/test image.
- The current package is designed for reproducibility and review. It is not a minimal pip package.
