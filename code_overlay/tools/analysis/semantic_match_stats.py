import argparse
import csv
import json
import math
import os
from collections import defaultdict

import torch
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


KNOWN_CLASSES = ('person', 'car', 'bus', 'bicycle', 'motorbike')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure GT-center visual-semantic matching statistics.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--data-root', default='/data/weather/WeatherOWOD')
    parser.add_argument('--work-dir', default='work_dirs/semantic_match_stats')
    parser.add_argument('--out-dir', default='work_dirs/semantic_match_stats')
    parser.add_argument('--num-known', type=int, default=5)
    parser.add_argument('--max-images', type=int, default=-1)
    return parser.parse_args()


def find_dataset_cfg(dataset_cfg):
    cur = dataset_cfg
    while isinstance(cur, (dict, ConfigDict)) and 'dataset' in cur:
        cur = cur['dataset']
    return cur


def override_test_dataset(cfg, ann_file, data_root):
    for key in ('test_dataloader', 'val_dataloader'):
        if key not in cfg:
            continue
        dataset_cfg = find_dataset_cfg(cfg[key]['dataset'])
        dataset_cfg['data_root'] = data_root
        dataset_cfg['ann_file'] = ann_file
        dataset_cfg['data_prefix'] = dict(img='')
        if 'filter_cfg' in dataset_cfg:
            dataset_cfg['filter_cfg'] = dict(filter_empty_gt=False,
                                             min_size=1)
    return cfg


def build_runner(config, checkpoint, ann_file, data_root, work_dir):
    cfg = Config.fromfile(config)
    cfg.launcher = 'none'
    cfg.load_from = checkpoint
    cfg.work_dir = work_dir
    cfg.log_level = 'WARNING'
    cfg = override_test_dataset(cfg, ann_file, data_root)
    is_metainfo_lower(cfg)
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.load_or_resume()
    return runner


def get_test_loader(runner):
    if hasattr(runner, 'test_dataloader'):
        return runner.test_dataloader
    if hasattr(runner, 'test_loop') and hasattr(runner.test_loop, 'dataloader'):
        return runner.test_loop.dataloader
    if hasattr(runner, '_test_loop') and hasattr(runner._test_loop,
                                                'dataloader'):
        return runner._test_loop.dataloader
    raise RuntimeError('Cannot locate test dataloader from runner.')


def as_model(runner):
    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    return model


def choose_scale_vector(cls_scores, batch_idx, center_x, center_y, img_w, img_h,
                        num_known):
    best_vec = None
    best_top_logit = None
    for level_score in cls_scores:
        score = level_score[batch_idx]
        channels, h, w = score.shape
        if channels < num_known:
            continue
        x_idx = int(math.floor(center_x / max(img_w, 1) * w))
        y_idx = int(math.floor(center_y / max(img_h, 1) * h))
        x_idx = min(max(x_idx, 0), w - 1)
        y_idx = min(max(y_idx, 0), h - 1)
        vec = score[:num_known, y_idx, x_idx].detach().float().cpu()
        top_logit = float(vec.max().item())
        if best_top_logit is None or top_logit > best_top_logit:
            best_top_logit = top_logit
            best_vec = vec
    return best_vec


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all'].append(row)
        groups[row['class_name']].append(row)

    summary = {}
    for group, items in groups.items():
        if not items:
            continue
        n = len(items)
        keys = ('correct_logit', 'correct_prob', 'margin_logit',
                'top1_correct', 'entropy')
        entry = {'n': n}
        for key in keys:
            vals = [float(item[key]) for item in items]
            entry[key + '_mean'] = sum(vals) / n
        summary[group] = entry
    return summary


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runner = build_runner(args.config, args.checkpoint, args.ann_file,
                          args.data_root, args.work_dir)
    model = as_model(runner)
    model.eval()
    if hasattr(model, 'bbox_head') and hasattr(model, 'num_test_classes'):
        model.bbox_head.num_classes = model.num_test_classes

    rows = []
    processed_images = 0
    dataloader = get_test_loader(runner)

    with torch.no_grad():
        for data in dataloader:
            data = model.data_preprocessor(data, False)
            inputs = data['inputs']
            samples = data['data_samples']
            outs = model._forward(inputs, samples)
            cls_scores = outs[0]

            for batch_idx, sample in enumerate(samples):
                metainfo = sample.metainfo
                img_h, img_w = metainfo.get('img_shape',
                                            metainfo.get('ori_shape'))[:2]
                gt = sample.gt_instances
                labels = gt.labels.detach().cpu()
                bboxes = gt.bboxes.detach().cpu()
                for gt_index, (label, bbox) in enumerate(zip(labels, bboxes)):
                    label = int(label.item())
                    if label < 0 or label >= args.num_known:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                    center_x = (x1 + x2) * 0.5
                    center_y = (y1 + y2) * 0.5
                    vec = choose_scale_vector(cls_scores, batch_idx,
                                              center_x, center_y, img_w,
                                              img_h, args.num_known)
                    if vec is None:
                        continue
                    correct_logit = float(vec[label].item())
                    probs = torch.sigmoid(vec)
                    correct_prob = float(probs[label].item())
                    other_mask = torch.ones(args.num_known, dtype=torch.bool)
                    other_mask[label] = False
                    best_other = float(vec[other_mask].max().item())
                    margin = correct_logit - best_other
                    top1 = int(torch.argmax(vec).item() == label)
                    prob_dist = torch.softmax(vec, dim=0)
                    entropy = float(
                        -(prob_dist *
                          torch.log(prob_dist.clamp_min(1e-9))).sum().item())
                    rows.append({
                        'model': args.model_name,
                        'split': args.split,
                        'img_id': metainfo.get('img_id', ''),
                        'img_path': metainfo.get('img_path', ''),
                        'gt_index': gt_index,
                        'class_id': label,
                        'class_name': KNOWN_CLASSES[label],
                        'correct_logit': correct_logit,
                        'correct_prob': correct_prob,
                        'best_other_logit': best_other,
                        'margin_logit': margin,
                        'top1_correct': top1,
                        'entropy': entropy,
                    })

                processed_images += 1
                if args.max_images > 0 and processed_images >= args.max_images:
                    break
            if args.max_images > 0 and processed_images >= args.max_images:
                break

    stem = f'{args.model_name}_{args.split}'.replace('/', '_')
    csv_path = os.path.join(args.out_dir, stem + '_gt_center_stats.csv')
    json_path = os.path.join(args.out_dir, stem + '_summary.json')
    fieldnames = [
        'model', 'split', 'img_id', 'img_path', 'gt_index', 'class_id',
        'class_name', 'correct_logit', 'correct_prob', 'best_other_logit',
        'margin_logit', 'top1_correct', 'entropy'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        'model': args.model_name,
        'split': args.split,
        'processed_images': processed_images,
        'gt_rows': len(rows),
        'summary': summary.get('all', {}),
        'csv': csv_path,
        'json': json_path,
    }, indent=2))


if __name__ == '__main__':
    main()
