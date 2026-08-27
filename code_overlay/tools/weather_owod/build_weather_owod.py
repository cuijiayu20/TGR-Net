import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


VOC20_TASKS = [
    ['person', 'car', 'bus', 'bicycle', 'motorbike'],
    ['aeroplane', 'boat', 'train', 'bottle', 'chair'],
    ['bird', 'cat', 'cow', 'dog', 'horse'],
    ['diningtable', 'pottedplant', 'sheep', 'sofa', 'tvmonitor'],
]

UNKNOWN_ID = 80


def all_classes_until(task_id):
    classes = []
    for task_classes in VOC20_TASKS[:task_id]:
        classes.extend(task_classes)
    return classes


def future_classes_after(task_id):
    classes = []
    for task_classes in VOC20_TASKS[task_id:]:
        classes.extend(task_classes)
    return classes


def read_ids_from_dir(image_dir, suffix='.jpg', strip_suffix=None):
    ids = []
    for path in sorted(Path(image_dir).glob(f'*{suffix}')):
        stem = path.stem
        if strip_suffix and stem.endswith(strip_suffix):
            stem = stem[:-len(strip_suffix)]
        ids.append(stem)
    return ids


def parse_voc_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext('filename') or (xml_path.stem + '.jpg')
    size = root.find('size')
    width = int(float(size.findtext('width'))) if size is not None else None
    height = int(float(size.findtext('height'))) if size is not None else None
    objects = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        difficult = int(obj.findtext('difficult') or 0)
        box = obj.find('bndbox')
        if box is None or not name:
            continue
        xmin = float(box.findtext('xmin'))
        ymin = float(box.findtext('ymin'))
        xmax = float(box.findtext('xmax'))
        ymax = float(box.findtext('ymax'))
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'difficult': difficult,
        })
    return filename, width, height, objects


def image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def make_categories(known_classes, include_unknown):
    categories = [
        {'id': idx, 'name': name}
        for idx, name in enumerate(known_classes)
    ]
    if include_unknown:
        categories.append({'id': UNKNOWN_ID, 'name': 'unknown'})
    return categories


def build_coco_json(ids, image_dir, ann_dir, out_path, task_id, mode):
    known_classes = all_classes_until(task_id)
    future_classes = set(future_classes_after(task_id))
    class_to_id = {name: idx for idx, name in enumerate(known_classes)}
    include_unknown = mode == 'test'
    images = []
    annotations = []
    ann_id = 1

    image_dir = Path(image_dir)
    ann_dir = Path(ann_dir)
    for img_idx, image_id in enumerate(ids, start=1):
        xml_path = ann_dir / f'{image_id}.xml'
        if not xml_path.exists():
            continue
        image_path = image_dir / f'{image_id}.jpg'
        filename, width, height, objects = parse_voc_xml(xml_path)
        if width is None or height is None:
            width, height = image_size(image_path)

        images.append({
            'id': img_idx,
            'file_name': str(image_path.as_posix()),
            'width': width,
            'height': height,
            'source_id': image_id,
        })

        for obj in objects:
            name = obj['name']
            if name in class_to_id:
                category_id = class_to_id[name]
            elif include_unknown and name in future_classes:
                category_id = UNKNOWN_ID
            else:
                continue
            xmin, ymin, xmax, ymax = obj['bbox']
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                'id': ann_id,
                'image_id': img_idx,
                'category_id': category_id,
                'bbox': [xmin, ymin, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': obj['difficult'],
            })
            ann_id += 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({
            'images': images,
            'annotations': annotations,
            'categories': make_categories(known_classes, include_unknown),
        }, f, ensure_ascii=False)


def write_pair_jsonl(ids, clean_dir, degraded_dir, ann_dir, out_path, task_id,
                     weather, degraded_suffix=''):
    clean_dir = Path(clean_dir)
    degraded_dir = Path(degraded_dir)
    ann_dir = Path(ann_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in ids:
        clean_path = clean_dir / f'{image_id}.jpg'
        degraded_name = f'{image_id}{degraded_suffix}.jpg'
        degraded_path = degraded_dir / degraded_name
        ann_path = ann_dir / f'{image_id}.xml'
        if not clean_path.exists() or not degraded_path.exists() or not ann_path.exists():
            continue
        rows.append({
            'image_id': image_id,
            'task_id': task_id,
            'weather': weather,
            'clean_path': str(clean_path.as_posix()),
            'degraded_path': str(degraded_path.as_posix()),
            'annotation_path': str(ann_path.as_posix()),
        })
    with out_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(rows)


def write_text_files(out_root):
    text_root = Path(out_root) / 'texts'
    text_root.mkdir(parents=True, exist_ok=True)
    voc20 = [name for task in VOC20_TASKS for name in task]
    payload = [[name] for name in voc20]
    (text_root / 'voc20_class_texts.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    for task_id in range(1, len(VOC20_TASKS) + 1):
        known = [[name] for name in all_classes_until(task_id)]
        (text_root / f'task{task_id}_class_texts.json').write_text(
            json.dumps(known, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', default=r'E:\data')
    parser.add_argument('--out-root',
                        default='data/WeatherOWOD',
                        help='Output root inside the current repository.')
    args = parser.parse_args()

    src = Path(args.src_root)
    out = Path(args.out_root)

    voc_train = src / 'VOC2007' / 'dataest' / 'voc-fog(9578+2129)' / 'train'
    voc_test = src / 'VOC2007' / 'dataest' / 'voc-fog(9578+2129)' / 'test'
    train_clean = voc_train / 'JPEGImages'
    train_ann = voc_train / 'Annotations'
    train_fog = voc_train / 'VOC2007-FOG'
    test_clean = voc_test / 'JPEGImages'
    test_ann = voc_test / 'Annotations'
    test_fog = voc_test / 'VOCtest-FOG'
    train_rain = src / 'VOC2007' / 'RainyImages'
    train_snow = src / 'VOC2007' / 'SnowyImages'
    rtts_img = src / 'RTTS' / 'JPEGImages'
    rtts_ann = src / 'RTTS' / 'Annotations'

    train_ids = read_ids_from_dir(train_clean)
    test_ids = read_ids_from_dir(test_clean)
    rtts_ids = read_ids_from_dir(rtts_img)

    write_text_files(out)

    for task_id in range(1, len(VOC20_TASKS) + 1):
        ann_root = out / 'annotations' / f'task{task_id}'
        build_coco_json(train_ids, train_clean, train_ann,
                        ann_root / 'train_clean.json', task_id, 'train')
        build_coco_json(train_ids, train_fog, train_ann,
                        ann_root / 'train_fog.json', task_id, 'train')
        build_coco_json(train_ids, train_rain, train_ann,
                        ann_root / 'train_rain.json', task_id, 'train')
        build_coco_json(train_ids, train_snow, train_ann,
                        ann_root / 'train_snow.json', task_id, 'train')
        build_coco_json(test_ids, test_clean, test_ann,
                        ann_root / 'test_clean_ow.json', task_id, 'test')
        build_coco_json(test_ids, test_fog, test_ann,
                        ann_root / 'test_fog_ow.json', task_id, 'test')
        build_coco_json(rtts_ids, rtts_img, rtts_ann,
                        ann_root / 'test_rtts_ow.json', task_id, 'test')

        pair_root = out / 'pairs'
        fog_n = write_pair_jsonl(train_ids, train_clean, train_fog, train_ann,
                                 pair_root / f'task{task_id}_fog_train.jsonl',
                                 task_id, 'fog')
        rain_n = write_pair_jsonl(train_ids, train_clean, train_rain, train_ann,
                                  pair_root / f'task{task_id}_rain_train.jsonl',
                                  task_id, 'rain', degraded_suffix='_rain')
        snow_n = write_pair_jsonl(train_ids, train_clean, train_snow, train_ann,
                                  pair_root / f'task{task_id}_snow_train.jsonl',
                                  task_id, 'snow', degraded_suffix='_snow')
        mixed_path = pair_root / f'task{task_id}_mixed_train.jsonl'
        with mixed_path.open('w', encoding='utf-8') as mixed:
            for name in ['fog', 'rain', 'snow']:
                p = pair_root / f'task{task_id}_{name}_train.jsonl'
                if p.exists():
                    mixed.write(p.read_text(encoding='utf-8'))
        print(f'task{task_id}: train={len(train_ids)} test={len(test_ids)} '
              f'rtts={len(rtts_ids)} pairs fog/rain/snow={fog_n}/{rain_n}/{snow_n}')


if __name__ == '__main__':
    main()
