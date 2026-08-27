import argparse
import json
import os
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


def known_classes(task_id):
    return [name for task in VOC20_TASKS[:task_id] for name in task]


def future_classes(task_id):
    return [name for task in VOC20_TASKS[task_id:] for name in task]


def voc20_classes():
    return [name for task in VOC20_TASKS for name in task]


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def materialize(src, dst, mode='hardlink', overwrite=False):
    src = Path(src)
    dst = Path(dst)
    if not src.exists() and dst.exists():
        return True
    if not src.exists():
        return False
    ensure_parent(dst)
    if dst.exists():
        if not overwrite:
            return True
        dst.unlink()
    if mode == 'move':
        shutil.move(str(src), str(dst))
    elif mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return True


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    width = int(float(size.findtext('width'))) if size is not None else None
    height = int(float(size.findtext('height'))) if size is not None else None
    objects = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        bbox = obj.find('bndbox')
        if not name or bbox is None:
            continue
        xmin = float(bbox.findtext('xmin'))
        ymin = float(bbox.findtext('ymin'))
        xmax = float(bbox.findtext('xmax'))
        ymax = float(bbox.findtext('ymax'))
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax],
            'difficult': int(obj.findtext('difficult') or 0),
        })
    return width, height, objects


def get_image_size(path):
    with Image.open(path) as img:
        return img.size


def list_ids(image_dir, strip_suffix=''):
    ids = []
    for p in sorted(Path(image_dir).glob('*')):
        if not p.is_file():
            continue
        stem = p.stem
        if strip_suffix and stem.endswith(strip_suffix):
            stem = stem[:-len(strip_suffix)]
        ids.append(stem)
    return ids


def find_image_by_stem(image_dir, image_id):
    image_dir = Path(image_dir)
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        p = image_dir / f'{image_id}{ext}'
        if p.exists():
            return p
    matches = list(image_dir.glob(f'{image_id}.*'))
    return matches[0] if matches else None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8')


def write_text_prompts(root):
    text_root = Path(root) / 'texts'
    write_json(text_root / 'voc20_class_texts.json',
               [[name] for name in voc20_classes()])
    for task_id in range(1, 5):
        write_json(text_root / f'task{task_id}_class_texts.json',
                   [[name] for name in known_classes(task_id)])
    write_json(text_root / 'unknown_prompts.json',
               [['unknown object'], ['unlabeled object'], ['novel object']])


def coco_categories(task_id, include_unknown):
    cats = [{'id': i, 'name': name} for i, name in enumerate(known_classes(task_id))]
    if include_unknown:
        cats.append({'id': UNKNOWN_ID, 'name': 'unknown'})
    return cats


def build_coco_from_organized(root, ids, image_subdir, xml_subdir, out_json,
                              task_id, split_mode):
    root = Path(root)
    known = {name: i for i, name in enumerate(known_classes(task_id))}
    future = set(future_classes(task_id))
    include_unknown = split_mode == 'test'
    images = []
    annotations = []
    ann_id = 1

    for img_idx, image_id in enumerate(ids, start=1):
        image_path = find_image_by_stem(root / image_subdir, image_id)
        image_rel = image_path.relative_to(root) if image_path else Path(image_subdir) / f'{image_id}.jpg'
        xml_path = root / xml_subdir / f'{image_id}.xml'
        if image_path is None or not xml_path.exists():
            continue
        width, height, objects = parse_xml(xml_path)
        if width is None or height is None:
            width, height = get_image_size(image_path)
        images.append({
            'id': img_idx,
            'file_name': image_rel.as_posix(),
            'width': width,
            'height': height,
            'source_id': image_id,
        })
        for obj in objects:
            name = obj['name']
            if name in known:
                cat_id = known[name]
            elif include_unknown and name in future:
                cat_id = UNKNOWN_ID
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
                'category_id': cat_id,
                'bbox': [xmin, ymin, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': obj['difficult'],
            })
            ann_id += 1

    write_json(out_json, {
        'images': images,
        'annotations': annotations,
        'categories': coco_categories(task_id, include_unknown),
    })
    return len(images), len(annotations)


def write_pairs(root, task_id, train_ids, weather):
    root = Path(root)
    pair_path = root / 'pairs' / f'task{task_id}_{weather}_train.jsonl'
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id in train_ids:
        clean = root / 'images' / 'voc' / 'train' / 'clean' / f'{image_id}.jpg'
        degraded = root / 'images' / 'voc' / 'train' / weather / f'{image_id}.jpg'
        xml = root / 'annotations' / 'xml' / 'voc' / 'train' / f'{image_id}.xml'
        if clean.exists() and degraded.exists() and xml.exists():
            rows.append({
                'image_id': image_id,
                'task_id': task_id,
                'weather': weather,
                'clean_path': clean.relative_to(root).as_posix(),
                'degraded_path': degraded.relative_to(root).as_posix(),
                'annotation_path': xml.relative_to(root).as_posix(),
            })
    with pair_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(rows)


def materialize_voc(src_root, out_root, mode, overwrite):
    src_root = Path(src_root)
    out_root = Path(out_root)
    fog_base = src_root / 'VOC2007' / 'dataest' / 'voc-fog(9578+2129)'
    train_clean = fog_base / 'train' / 'JPEGImages'
    train_xml = fog_base / 'train' / 'Annotations'
    train_fog = fog_base / 'train' / 'VOC2007-FOG'
    test_clean = fog_base / 'test' / 'JPEGImages'
    test_xml = fog_base / 'test' / 'Annotations'
    test_fog = fog_base / 'test' / 'VOCtest-FOG'
    rain = src_root / 'VOC2007' / 'RainyImages'
    snow = src_root / 'VOC2007' / 'SnowyImages'

    train_ids = list_ids(train_clean)
    if not train_ids:
        train_ids = list_ids(out_root / 'images' / 'voc' / 'train' / 'clean')
    test_ids = list_ids(test_clean)
    if not test_ids:
        test_ids = list_ids(out_root / 'images' / 'voc' / 'test' / 'clean')

    for image_id in train_ids:
        materialize(train_clean / f'{image_id}.jpg',
                    out_root / 'images' / 'voc' / 'train' / 'clean' / f'{image_id}.jpg',
                    mode, overwrite)
        materialize(train_fog / f'{image_id}.jpg',
                    out_root / 'images' / 'voc' / 'train' / 'fog' / f'{image_id}.jpg',
                    mode, overwrite)
        materialize(rain / f'{image_id}_rain.jpg',
                    out_root / 'images' / 'voc' / 'train' / 'rain' / f'{image_id}.jpg',
                    mode, overwrite)
        materialize(snow / f'{image_id}_snow.jpg',
                    out_root / 'images' / 'voc' / 'train' / 'snow' / f'{image_id}.jpg',
                    mode, overwrite)
        materialize(train_xml / f'{image_id}.xml',
                    out_root / 'annotations' / 'xml' / 'voc' / 'train' / f'{image_id}.xml',
                    mode, overwrite)

    for image_id in test_ids:
        materialize(test_clean / f'{image_id}.jpg',
                    out_root / 'images' / 'voc' / 'test' / 'clean' / f'{image_id}.jpg',
                    mode, overwrite)
        materialize(test_fog / f'{image_id}.jpg',
                    out_root / 'images' / 'voc' / 'test' / 'fog' / f'{image_id}.jpg',
                    mode, overwrite)
        materialize(test_xml / f'{image_id}.xml',
                    out_root / 'annotations' / 'xml' / 'voc' / 'test' / f'{image_id}.xml',
                    mode, overwrite)

    return train_ids, test_ids


def materialize_real(src_root, out_root, mode, overwrite):
    src_root = Path(src_root)
    out_root = Path(out_root)
    rtts_img = src_root / 'RTTS' / 'JPEGImages'
    rtts_xml = src_root / 'RTTS' / 'Annotations'
    rtts_ids = list_ids(rtts_img)
    if not rtts_ids:
        rtts_ids = list_ids(out_root / 'images' / 'real' / 'rtts' / 'fog')
    for image_id in rtts_ids:
        img_path = find_image_by_stem(rtts_img, image_id)
        if img_path is not None:
            materialize(img_path,
                        out_root / 'images' / 'real' / 'rtts' / 'fog' /
                        f'{image_id}{img_path.suffix.lower()}',
                        mode, overwrite)
        xml_path = rtts_xml / f'{image_id}.xml'
        if not xml_path.exists():
            xml_path = out_root / 'annotations' / 'xml' / 'real' / 'rtts' / f'{image_id}.xml'
        materialize(xml_path,
                    out_root / 'annotations' / 'xml' / 'real' / 'rtts' / f'{image_id}.xml',
                    mode, overwrite)

    fd_root = src_root / 'VOC2007' / 'dataest' / 'Foggy_Driving_voc'
    fd_img = fd_root / 'JPEGImages'
    fd_xml = fd_root / 'Annotations'
    fd_ids = []
    fd_images = sorted(fd_img.glob('*')) if fd_img.exists() else []
    if not fd_images:
        fd_images = sorted((out_root / 'images' / 'real' / 'foggy_driving' /
                            'fog').glob('*'))
    for img in fd_images:
        if not img.is_file():
            continue
        image_id = img.stem
        fd_ids.append(image_id)
        dst_ext = img.suffix.lower()
        materialize(img,
                    out_root / 'images' / 'real' / 'foggy_driving' / 'fog' / f'{image_id}{dst_ext}',
                    mode, overwrite)
        materialize(fd_xml / f'{image_id}.xml',
                    out_root / 'annotations' / 'xml' / 'real' / 'foggy_driving' / f'{image_id}.xml',
                    mode, overwrite)
    return rtts_ids, fd_ids


def build_annotations(out_root, train_ids, test_ids, rtts_ids, fd_ids):
    out_root = Path(out_root)
    write_text_prompts(out_root)
    for task_id in range(1, 5):
        task_dir = out_root / 'annotations' / 'coco' / f'task{task_id}'
        specs = [
            ('train_clean.json', train_ids, 'images/voc/train/clean',
             out_root / 'annotations/xml/voc/train', 'train'),
            ('train_fog.json', train_ids, 'images/voc/train/fog',
             out_root / 'annotations/xml/voc/train', 'train'),
            ('train_rain.json', train_ids, 'images/voc/train/rain',
             out_root / 'annotations/xml/voc/train', 'train'),
            ('train_snow.json', train_ids, 'images/voc/train/snow',
             out_root / 'annotations/xml/voc/train', 'train'),
            ('test_clean_ow.json', test_ids, 'images/voc/test/clean',
             out_root / 'annotations/xml/voc/test', 'test'),
            ('test_fog_ow.json', test_ids, 'images/voc/test/fog',
             out_root / 'annotations/xml/voc/test', 'test'),
            ('test_rtts_ow.json', rtts_ids, 'images/real/rtts/fog',
             out_root / 'annotations/xml/real/rtts', 'test'),
            ('test_foggy_driving_ow.json', fd_ids, 'images/real/foggy_driving/fog',
             out_root / 'annotations/xml/real/foggy_driving', 'test'),
        ]
        for filename, ids, image_subdir, xml_dir, mode in specs:
            n_img, n_ann = build_coco_from_organized(
                out_root, ids, image_subdir, xml_dir, task_dir / filename,
                task_id, mode)
            print(f'task{task_id} {filename}: images={n_img} annotations={n_ann}')
        pair_counts = {
            w: write_pairs(out_root, task_id, train_ids, w)
            for w in ['fog', 'rain', 'snow']
        }
        mixed_path = out_root / 'pairs' / f'task{task_id}_mixed_train.jsonl'
        with mixed_path.open('w', encoding='utf-8') as mixed:
            for weather in ['fog', 'rain', 'snow']:
                p = out_root / 'pairs' / f'task{task_id}_{weather}_train.jsonl'
                mixed.write(p.read_text(encoding='utf-8'))
        print(f'task{task_id} pairs: {pair_counts}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', default=r'E:\data')
    parser.add_argument('--out-root', default=r'E:\data\WeatherOWOD')
    parser.add_argument('--mode', choices=['move', 'hardlink', 'copy', 'symlink'],
                        default='hardlink')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    train_ids, test_ids = materialize_voc(
        args.src_root, out_root, args.mode, args.overwrite)
    rtts_ids, fd_ids = materialize_real(
        args.src_root, out_root, args.mode, args.overwrite)
    build_annotations(out_root, train_ids, test_ids, rtts_ids, fd_ids)
    write_json(out_root / 'dataset_meta.json', {
        'name': 'WeatherOWOD',
        'source_root': str(Path(args.src_root)),
        'tasks': {f'task{i + 1}': cls for i, cls in enumerate(VOC20_TASKS)},
        'unknown_category_id': UNKNOWN_ID,
        'materialization': args.mode,
        'splits': {
            'voc_train': len(train_ids),
            'voc_test': len(test_ids),
            'rtts': len(rtts_ids),
            'foggy_driving': len(fd_ids),
        },
    })


if __name__ == '__main__':
    main()
