import argparse
import csv
import json
import math
import os
from collections import defaultdict

import torch
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner, load_checkpoint

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower


VOC20_CLASSES = (
    'person', 'car', 'bus', 'bicycle', 'motorbike',
    'aeroplane', 'boat', 'train', 'bottle', 'chair',
    'bird', 'cat', 'cow', 'dog', 'horse',
    'diningtable', 'pottedplant', 'sheep', 'sofa', 'tvmonitor')


def parse_csv_ints(text):
    return tuple(int(item.strip()) for item in text.split(',')
                 if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe whether adding an open-vocabulary class hurts '
        'visual-semantic matching after Mamba-KAT adaptation.')
    parser.add_argument('config')
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--adapter-checkpoint', default=None)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--data-root', default='/data/weather/WeatherOWOD')
    parser.add_argument('--work-dir',
                        default='work_dirs/open_vocab_add_class_probe')
    parser.add_argument('--out-dir',
                        default='work_dirs/open_vocab_add_class_probe')
    parser.add_argument(
        '--candidate-label-ids',
        default='0,1,2,3,4,5',
        help='Vocabulary labels used as competing prompts. Default simulates '
        'Task 1 plus aeroplane.')
    parser.add_argument(
        '--probe-label-ids',
        default='0,1,2,3,4,5',
        help='GT labels to summarize. Default: five Task1 classes plus '
        'aeroplane.')
    parser.add_argument('--max-images', type=int, default=-1)
    parser.add_argument('--bypass-adapter', action='store_true')
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


def configure_adapter(cfg, bypass_adapter):
    if 'degradation_adapter' not in cfg.model:
        return cfg
    cfg.model.degradation_adapter.update(
        dict(
            gamma_init=0.1,
            use_mamba_ssm=False,
            kat_hidden_ratio=0.5,
            use_mamba_branch=True,
            use_kat_branch=True,
            use_conv_branch=True,
            bypass=bypass_adapter))
    return cfg


def build_runner(args):
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = args.work_dir
    cfg.log_level = 'WARNING'
    cfg = override_test_dataset(cfg, args.ann_file, args.data_root)
    cfg = configure_adapter(cfg, args.bypass_adapter)
    is_metainfo_lower(cfg)
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
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


def load_adapter_weights(model, adapter_checkpoint):
    if adapter_checkpoint is None:
        return {'loaded': 0, 'missing': [], 'unexpected': []}
    checkpoint = torch.load(adapter_checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    adapter_state = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('degradation_adapter.'):
            adapter_key = key[len('degradation_adapter.'):]
            adapter_state[adapter_key] = value
    if not adapter_state:
        raise RuntimeError(
            f'No degradation_adapter weights found in {adapter_checkpoint}')
    incompatible = model.degradation_adapter.load_state_dict(adapter_state,
                                                             strict=False)
    return {
        'loaded': len(adapter_state),
        'missing': list(incompatible.missing_keys),
        'unexpected': list(incompatible.unexpected_keys),
    }


def choose_scale_vector(cls_scores, batch_idx, center_x, center_y, img_w, img_h,
                        candidate_label_ids):
    best_vec = None
    best_top_logit = None
    max_label_id = max(candidate_label_ids)
    candidate_index = torch.tensor(candidate_label_ids, dtype=torch.long)
    for level_score in cls_scores:
        score = level_score[batch_idx]
        channels, h, w = score.shape
        if channels <= max_label_id:
            continue
        x_idx = int(math.floor(center_x / max(img_w, 1) * w))
        y_idx = int(math.floor(center_y / max(img_h, 1) * h))
        x_idx = min(max(x_idx, 0), w - 1)
        y_idx = min(max(y_idx, 0), h - 1)
        vec = score[candidate_index, y_idx, x_idx].detach().float().cpu()
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
        n = len(items)
        if n == 0:
            continue
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
    candidate_label_ids = parse_csv_ints(args.candidate_label_ids)
    probe_label_ids = set(parse_csv_ints(args.probe_label_ids))
    candidate_pos = {label_id: pos
                     for pos, label_id in enumerate(candidate_label_ids)}

    runner = build_runner(args)
    model = as_model(runner)
    load_checkpoint(model,
                    args.base_checkpoint,
                    map_location='cpu',
                    strict=False)
    adapter_info = load_adapter_weights(model, args.adapter_checkpoint)
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
                for label, bbox in zip(labels, bboxes):
                    label = int(label.item())
                    if label not in probe_label_ids:
                        continue
                    if label not in candidate_pos:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
                    center_x = (x1 + x2) * 0.5
                    center_y = (y1 + y2) * 0.5
                    vec = choose_scale_vector(cls_scores, batch_idx,
                                              center_x, center_y, img_w,
                                              img_h, candidate_label_ids)
                    if vec is None:
                        continue
                    correct_pos = candidate_pos[label]
                    correct_logit = float(vec[correct_pos].item())
                    probs = torch.sigmoid(vec)
                    correct_prob = float(probs[correct_pos].item())
                    other_mask = torch.ones(len(candidate_label_ids),
                                            dtype=torch.bool)
                    other_mask[correct_pos] = False
                    best_other = float(vec[other_mask].max().item())
                    margin = correct_logit - best_other
                    top1 = int(torch.argmax(vec).item() == correct_pos)
                    prob_dist = torch.softmax(vec, dim=0)
                    entropy = float(
                        -(prob_dist *
                          torch.log(prob_dist.clamp_min(1e-9))).sum().item())
                    rows.append({
                        'model': args.model_name,
                        'split': args.split,
                        'img_id': metainfo.get('img_id', ''),
                        'img_path': metainfo.get('img_path', ''),
                        'class_id': label,
                        'class_name': VOC20_CLASSES[label],
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
    csv_path = os.path.join(args.out_dir, stem + '_add_class_stats.csv')
    json_path = os.path.join(args.out_dir, stem + '_add_class_summary.json')
    fieldnames = [
        'model', 'split', 'img_id', 'img_path', 'class_id', 'class_name',
        'correct_logit', 'correct_prob', 'best_other_logit', 'margin_logit',
        'top1_correct', 'entropy'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with open(json_path, 'w') as f:
        json.dump({
            'summary': summary,
            'candidate_label_ids': candidate_label_ids,
            'probe_label_ids': sorted(probe_label_ids),
            'adapter_info': adapter_info,
        }, f, indent=2)
    print(json.dumps({
        'model': args.model_name,
        'split': args.split,
        'processed_images': processed_images,
        'gt_rows': len(rows),
        'summary': summary.get('all', {}),
        'aeroplane': summary.get('aeroplane', {}),
        'csv': csv_path,
        'json': json_path,
        'adapter_info': adapter_info,
    }, indent=2))


if __name__ == '__main__':
    main()
