# KADet Adverse-Weather Data Tools

这些脚本从 `D:\workingspace\mamba\KADet` 中整理而来，用于当前
YOLO-World + Mamba-KAT Adapter 项目的 clean/degraded 配对数据准备。

## KADet 中的数据处理方式

KADet 采用离线退化数据组织方式：

- clean 图像、雨图、雪图、雾图分别放在不同目录。
- 同一图像 ID 的不同天气版本共享同一份 VOC XML 标注。
- 雨图命名通常为 `<image_id>_rain.jpg`。
- 雪图命名通常为 `<image_id>_snow.jpg`。
- 雾图通常来自已有 `VOC2007-FOG`、`VOCtest-FOG`、`Foggy_Driving_voc` 或 `RTTStest` 目录，不是 KADet 脚本在线生成。
- KADet 训练 txt 格式为：

```text
image_path xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id ...
```

在当前 YOLO-World 方法中，更重要的是 paired 关系：

```text
clean image + degraded image + shared VOC XML annotation
```

因此这里额外提供 JSONL manifest，每一行包含：

```json
{
  "clean_img_path": "...",
  "degraded_img_path": "...",
  "ann_path": "...",
  "weather": "rain",
  "width": 500,
  "height": 375,
  "instances": [
    {
      "bbox": [xmin, ymin, xmax, ymax],
      "bbox_label": 0,
      "category_name": "person"
    }
  ]
}
```

## 1. 雨天合成

来源：`KADet/Rain_Synthesizing.py`

迁移后脚本：

```powershell
conda run -n mamba_kat python tools/adverse_weather/synthesize_rain_kadet.py `
  --clean-dir data/voc-fog/train/JPEGImages `
  --output-dir data/voc-fog/train/RainyImages `
  --suffix _rain `
  --seed 0
```

默认行为：

- 生成随机稀疏噪声。
- 对噪声施加带角度的运动模糊，形成雨纹。
- 将雨纹以 alpha blending 方式叠加到 clean 图像上。
- 输出文件名为 `<stem>_rain<ext>`。

## 2. 雪天合成

来源：`KADet/Snow_Synthesizing.py`

迁移后脚本：

```powershell
conda run -n mamba_kat python tools/adverse_weather/synthesize_snow_kadet.py `
  --clean-dir data/voc-fog/train/JPEGImages `
  --snow-mask-dir data/snow_masks `
  --output-dir data/voc-fog/train/SnowyImages `
  --suffix _snow `
  --seed 0
```

默认行为：

- 从 `snow-mask-dir` 随机选取雪幕 mask。
- 将 snow mask resize 到 clean 图像大小。
- 使用加权融合生成雪天图像。
- 输出文件名为 `<stem>_snow<ext>`。

## 3. 雾天数据处理

KADet 中没有独立的雾合成脚本。雾天数据主要按已有目录读取：

- `VOC2007-FOG`
- `VOCtest-FOG`
- `Foggy_Driving_voc/JPEGImages`
- `RTTStest/Images`

如果是 VOC-FOG paired 数据，通常 clean 图像在 `JPEGImages`，雾图在
`VOC2007-FOG` 或 `VOCtest-FOG`，两者共享 `Annotations`。

## 4. 生成 paired manifest

### 雨天

```powershell
conda run -n mamba_kat python tools/adverse_weather/build_voc_weather_pairs.py `
  --clean-dir data/voc-fog/train/JPEGImages `
  --degraded-dir data/voc-fog/train/RainyImages `
  --ann-dir data/voc-fog/train/Annotations `
  --image-set-file data/voc-fog/train/ImageSets/Main/train.txt `
  --classes-file data/texts/voc_classes.txt `
  --weather rain `
  --degraded-suffix _rain `
  --output-jsonl data/adverse_weather/voc_train_rain_pairs.jsonl `
  --output-kadet-txt data/adverse_weather/voc_train_rain_kadet.txt `
  --output-pair-tsv data/adverse_weather/voc_train_rain_pairs.tsv
```

### 雪天

```powershell
conda run -n mamba_kat python tools/adverse_weather/build_voc_weather_pairs.py `
  --clean-dir data/voc-fog/train/JPEGImages `
  --degraded-dir data/voc-fog/train/SnowyImages `
  --ann-dir data/voc-fog/train/Annotations `
  --image-set-file data/voc-fog/train/ImageSets/Main/train.txt `
  --classes-file data/texts/voc_classes.txt `
  --weather snow `
  --degraded-suffix _snow `
  --output-jsonl data/adverse_weather/voc_train_snow_pairs.jsonl
```

### 雾天

```powershell
conda run -n mamba_kat python tools/adverse_weather/build_voc_weather_pairs.py `
  --clean-dir data/voc-fog/train/JPEGImages `
  --degraded-dir data/voc-fog/train/VOC2007-FOG `
  --ann-dir data/voc-fog/train/Annotations `
  --image-set-file data/voc-fog/train/ImageSets/Main/train.txt `
  --classes-file data/texts/voc_classes.txt `
  --weather fog `
  --degraded-suffix "" `
  --output-jsonl data/adverse_weather/voc_train_fog_pairs.jsonl
```

## 5. 接入 Mamba-KAT Adapter

`YOLOWorldMambaKATDetector` 训练阶段需要：

```text
clean_inputs
degraded_inputs
shared detection annotations
texts
```

这些工具只负责生成退化图像和 paired manifest，不在脚本内改写
YOLO-World 数据集类。后续 dataset/transform 可以读取 JSONL，并在
sample 中填入：

```python
sample['inputs'] = degraded_tensor
sample['clean_inputs'] = clean_tensor
sample['degraded_inputs'] = degraded_tensor
sample['data_samples'] = shared_detection_sample
```

已有的 `yolo_world/datasets/utils.py::yolow_collate` 会将这些字段打包成
detector 可用的 paired batch。
