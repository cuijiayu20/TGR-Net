"""Build paired clean/degraded manifests from KADet-style VOC weather data.

KADet stores clean, rain, snow, and fog images in different directories while
sharing one VOC XML annotation directory. This script converts that layout into
a JSONL manifest for YOLO-World adapter training.

Each output line contains:
    clean_img_path, degraded_img_path, ann_path, weather, width, height,
    instances[{bbox, bbox_label, category_name}]

For fog data, pass an empty suffix and point --degraded-dir to VOC2007-FOG or
VOCtest-FOG. For rain/snow, the default KADet naming is handled by suffixes
such as _rain and _snow.
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def read_classes(classes_file: Optional[Path]) -> Optional[List[str]]:
    if classes_file is None:
        return None
    with classes_file.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def collect_xml_classes(ann_dir: Path) -> List[str]:
    names = set()
    for xml_path in sorted(ann_dir.glob('*.xml')):
        root = ET.parse(xml_path).getroot()
        for obj in root.iter('object'):
            name_node = obj.find('name')
            if name_node is not None and name_node.text:
                names.add(name_node.text.strip())
    return sorted(names)


def parse_voc_xml(xml_path: Path,
                  class_to_id: Dict[str, int],
                  keep_difficult: bool = False) -> Tuple[int, int, List[dict]]:
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    width = int(float(size.findtext('width', default='0'))) if size is not None else 0
    height = int(float(size.findtext('height', default='0'))) if size is not None else 0
    instances = []

    for obj in root.iter('object'):
        difficult = int(float(obj.findtext('difficult', default='0')))
        if difficult == 1 and not keep_difficult:
            continue
        category_name = obj.findtext('name')
        if category_name not in class_to_id:
            continue
        box = obj.find('bndbox')
        if box is None:
            continue
        xmin = int(float(box.findtext('xmin')))
        ymin = int(float(box.findtext('ymin')))
        xmax = int(float(box.findtext('xmax')))
        ymax = int(float(box.findtext('ymax')))
        instances.append({
            'bbox': [xmin, ymin, xmax, ymax],
            'bbox_label': class_to_id[category_name],
            'category_name': category_name,
        })
    return width, height, instances


def image_ids_from_file(image_set_file: Optional[Path]) -> Optional[List[str]]:
    if image_set_file is None:
        return None
    with image_set_file.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def find_image_by_stem(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    lower_stem = stem.lower()
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem.lower() == lower_stem:
                return path
    return None


def iter_degraded_images(degraded_dir: Path) -> Iterable[Path]:
    for path in sorted(degraded_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def make_record(clean_path: Optional[Path],
                degraded_path: Path,
                ann_path: Path,
                weather: str,
                class_to_id: Dict[str, int],
                keep_difficult: bool,
                path_mode: str) -> dict:
    width, height, instances = parse_voc_xml(ann_path, class_to_id,
                                             keep_difficult)

    def fmt(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        if path_mode == 'absolute':
            return str(path.resolve())
        return os.path.normpath(str(path))

    return {
        'clean_img_path': fmt(clean_path),
        'degraded_img_path': fmt(degraded_path),
        'ann_path': fmt(ann_path),
        'weather': weather,
        'width': width,
        'height': height,
        'instances': instances,
    }


def write_kadet_txt(records: List[dict], output_path: Path) -> None:
    with output_path.open('w', encoding='utf-8') as f:
        for rec in records:
            parts = [rec['degraded_img_path']]
            for inst in rec['instances']:
                bbox = inst['bbox']
                parts.append(','.join(
                    [str(v) for v in bbox] + [str(inst['bbox_label'])]))
            f.write(' '.join(parts) + '\n')


def write_pair_tsv(records: List[dict], output_path: Path) -> None:
    with output_path.open('w', encoding='utf-8') as f:
        f.write('degraded_img_path\tclean_img_path\tweather\tann_path\n')
        for rec in records:
            clean = rec['clean_img_path'] or ''
            f.write(f"{rec['degraded_img_path']}\t{clean}\t{rec['weather']}\t{rec['ann_path']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert KADet-style VOC weather data to paired JSONL.')
    parser.add_argument('--clean-dir', type=Path, required=True)
    parser.add_argument('--degraded-dir', type=Path, required=True)
    parser.add_argument('--ann-dir', type=Path, required=True)
    parser.add_argument('--image-set-file', type=Path, default=None)
    parser.add_argument('--classes-file', type=Path, default=None)
    parser.add_argument('--weather', required=True)
    parser.add_argument('--degraded-suffix', default='')
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--output-kadet-txt', type=Path, default=None)
    parser.add_argument('--output-pair-tsv', type=Path, default=None)
    parser.add_argument('--path-mode',
                        choices=('relative', 'absolute'),
                        default='relative')
    parser.add_argument('--keep-difficult', action='store_true')
    parser.add_argument('--allow-missing-clean', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = read_classes(args.classes_file)
    if classes is None:
        classes = collect_xml_classes(args.ann_dir)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    image_ids = image_ids_from_file(args.image_set_file)
    records = []
    skipped = []

    if image_ids is not None:
        degraded_candidates = []
        for image_id in image_ids:
            degraded_stem = f'{image_id}{args.degraded_suffix}'
            degraded_path = find_image_by_stem(args.degraded_dir,
                                               degraded_stem)
            if degraded_path is not None:
                degraded_candidates.append((image_id, degraded_path))
            else:
                skipped.append((image_id, 'missing degraded image'))
    else:
        degraded_candidates = []
        for degraded_path in iter_degraded_images(args.degraded_dir):
            stem = degraded_path.stem
            if args.degraded_suffix and stem.endswith(args.degraded_suffix):
                image_id = stem[:-len(args.degraded_suffix)]
            else:
                image_id = stem
            degraded_candidates.append((image_id, degraded_path))

    for image_id, degraded_path in degraded_candidates:
        clean_path = find_image_by_stem(args.clean_dir, image_id)
        ann_path = args.ann_dir / f'{image_id}.xml'
        if clean_path is None and not args.allow_missing_clean:
            skipped.append((image_id, 'missing clean image'))
            continue
        if not ann_path.exists():
            skipped.append((image_id, 'missing annotation'))
            continue
        records.append(
            make_record(clean_path, degraded_path, ann_path, args.weather,
                        class_to_id, args.keep_difficult, args.path_mode))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    if args.output_kadet_txt is not None:
        args.output_kadet_txt.parent.mkdir(parents=True, exist_ok=True)
        write_kadet_txt(records, args.output_kadet_txt)

    if args.output_pair_tsv is not None:
        args.output_pair_tsv.parent.mkdir(parents=True, exist_ok=True)
        write_pair_tsv(records, args.output_pair_tsv)

    print(f'Wrote {len(records)} records to {args.output_jsonl}')
    if skipped:
        print(f'Skipped {len(skipped)} records')
        for image_id, reason in skipped[:20]:
            print(f'  {image_id}: {reason}')
        if len(skipped) > 20:
            print('  ...')


if __name__ == '__main__':
    main()
