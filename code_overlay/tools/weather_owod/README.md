# WeatherOWOD Dataset Protocol

This dataset protocol adapts VOC2007-derived adverse-weather data to a small
open-world object detection setting. VOC2007 provides the category protocol and
synthetic weather domains. RTTS and other real adverse-weather datasets are used
for cross-domain evaluation.

## Source Data

Expected source root:

```text
E:\data
├── VOC2007
│   ├── Annotations
│   ├── JPEGImages
│   ├── VOC2007-FOG
│   ├── RainyImages
│   ├── SnowyImages
│   └── dataest
│       └── voc-fog(9578+2129)
│           ├── train
│           │   ├── Annotations
│           │   ├── JPEGImages
│           │   └── VOC2007-FOG
│           └── test
│               ├── Annotations
│               ├── JPEGImages
│               └── VOCtest-FOG
├── RTTS
│   ├── Annotations
│   └── JPEGImages
└── RTTStest
    ├── Annotations
    └── Images
```

The VOC split is treated as the synthetic benchmark. RTTS is not used to define
the open-world training protocol because it contains a narrower traffic-oriented
label space; it is used for real-fog generalization.

## VOC20 Open-World Tasks

The protocol uses four incremental tasks over VOC20. Task 1 is aligned with the
RTTS traffic classes so that real-fog testing can be reported directly.

| Task | Newly introduced known classes | Future unknown classes |
| --- | --- | --- |
| T1 | person, car, bus, bicycle, motorbike | remaining 15 VOC classes |
| T2 | aeroplane, boat, train, bottle, chair | remaining 10 VOC classes |
| T3 | bird, cat, cow, dog, horse | remaining 5 VOC classes |
| T4 | diningtable, pottedplant, sheep, sofa, tvmonitor | none |

For task `t`, all classes introduced up to `t` are known classes. Classes from
future tasks are mapped to `unknown` during open-world evaluation.

## Organized Dataset Layout

Recommended output root:

```text
E:\data\WeatherOWOD
├── dataset_meta.json
├── images
│   ├── voc
│   │   ├── train
│   │   │   ├── clean
│   │   │   ├── fog
│   │   │   ├── rain
│   │   │   └── snow
│   │   └── test
│   │       ├── clean
│   │       └── fog
│   └── real
│       ├── rtts
│       │   └── fog
│       └── foggy_driving
│           └── fog
├── annotations
│   ├── xml
│   │   ├── voc
│   │   │   ├── train
│   │   │   └── test
│   │   └── real
│   │       ├── rtts
│   │       └── foggy_driving
│   └── coco
│       ├── task1
│       ├── task2
│       ├── task3
│       └── task4
├── pairs
│   ├── task1_fog_train.jsonl
│   ├── task1_rain_train.jsonl
│   ├── task1_snow_train.jsonl
│   └── task1_mixed_train.jsonl
└── texts
    ├── voc20_class_texts.json
    ├── task1_class_texts.json
    ├── task2_class_texts.json
    ├── task3_class_texts.json
    └── task4_class_texts.json
```

The organization script materializes a clean dataset tree. Use `--mode move` to
cut files from the original scattered directories into `E:\data\WeatherOWOD`.
This is the recommended cleanup mode when the old directory layout is no longer
needed. Use `--mode copy` for a non-destructive backup-style organization.

```powershell
python D:\workingspace\mamba\yolo-world-ow-ovd\tools\weather_owod\organize_weather_owod.py `
  --src-root E:\data `
  --out-root E:\data\WeatherOWOD `
  --mode move
```

## Annotation Semantics

Generated annotation files use COCO-style JSON.

- Training JSON keeps only currently known classes.
- Open-world test JSON keeps current and previous known classes as normal
  categories and maps future task classes to `unknown`.
- The unknown category uses category id `80`, matching common OWOD evaluation
  conventions.

The paired JSONL files are used by the Mamba-KAT feature alignment loader:

```json
{
  "image_id": "2007_000032",
  "task_id": 1,
  "weather": "fog",
  "clean_path": "E:/data/.../JPEGImages/2007_000032.jpg",
  "degraded_path": "E:/data/.../VOC2007-FOG/2007_000032.jpg",
  "annotation_path": "E:/data/.../Annotations/2007_000032.xml"
}
```

## Recommended Experiments

Use VOC synthetic weather as the main quantitative benchmark:

- VOC-FOG-test
- VOC-Rain-test, if a matched test set is available
- VOC-Snow-test, if a matched test set is available

Use RTTS as real-fog cross-domain evaluation:

- report known-class AP for traffic classes in Task 1
- optionally report unknown metrics only when annotations contain future VOC
  classes or when an unknown mapping is explicitly defined
