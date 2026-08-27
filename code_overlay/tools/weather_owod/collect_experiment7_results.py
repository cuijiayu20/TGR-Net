import json
import math
import re
from pathlib import Path


EXPERIMENTS = (
    'task1_full',
    'task1_wo_mamba',
    'task1_wo_kat',
    'task1_wo_conv',
    'task1_det_only',
    'task1_cwd_only',
)

DATASETS = (
    'clean',
    'fog',
    'rain',
    'snow',
    'mixed',
    'rtts',
    'foggy_driving',
)

METRICS = {
    'known_ap50': r'Known AP50:\s*([^\s]+)',
    'known_precision50': r'Known Precisions50:\s*([^\s]+)',
    'known_recall50': r'Known Recall50:\s*([^\s]+)',
    'unknown_ap50': r'Unknown AP50:\s*([^\s]+)',
    'unknown_precision50': r'Unknown Precisions50:\s*([^\s]+)',
    'unknown_recall50': r'Unknown Recall50:\s*([^\s]+)',
    'a_ose': r'total_num_unk_det_as_known:\s*\{50:\s*([0-9.]+)',
}


def parse_number(raw_value):
    try:
        value = float(raw_value.rstrip(','))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_log(log_path: Path):
    text = log_path.read_text(errors='replace')
    result = {}
    for metric_name, pattern in METRICS.items():
        matches = re.findall(pattern, text)
        result[metric_name] = (
            parse_number(matches[-1]) if matches else None)
    return result


def format_metric(value):
    if value is None:
        return 'N/A'
    return f'{value:.4f}'


def build_markdown(results):
    lines = [
        '# Experiment 7 Results',
        '',
        '## Branch Ablations',
        '',
        '| Method | Clean | Fog | Rain | Snow | Mixed | RTTS | '
        'Mixed U-Recall | Mixed A-OSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    branch_experiments = (
        'task1_full',
        'task1_wo_mamba',
        'task1_wo_kat',
        'task1_wo_conv',
    )
    for experiment in branch_experiments:
        row = results.get(experiment, {})
        values = [
            format_metric(row.get(dataset, {}).get('known_ap50'))
            for dataset in ('clean', 'fog', 'rain', 'snow', 'mixed', 'rtts')
        ]
        mixed = row.get('mixed', {})
        lines.append(
            f'| {experiment} | {" | ".join(values)} | '
            f'{format_metric(mixed.get("unknown_recall50"))} | '
            f'{format_metric(mixed.get("a_ose"))} |')

    lines.extend([
        '',
        '## Weather-by-Branch Contributions',
        '',
        '| Weather | Mamba | KAT | Conv |',
        '|---|---:|---:|---:|',
    ])
    full = results.get('task1_full', {})
    branch_map = {
        'Mamba': 'task1_wo_mamba',
        'KAT': 'task1_wo_kat',
        'Conv': 'task1_wo_conv',
    }
    for dataset in ('clean', 'fog', 'rain', 'snow', 'mixed', 'rtts'):
        contributions = []
        full_ap = full.get(dataset, {}).get('known_ap50')
        for experiment in branch_map.values():
            ablated_ap = results.get(experiment, {}).get(
                dataset, {}).get('known_ap50')
            contribution = (
                full_ap - ablated_ap
                if full_ap is not None and ablated_ap is not None else None)
            contributions.append(format_metric(contribution))
        lines.append(
            f'| {dataset} | {" | ".join(contributions)} |')

    lines.extend([
        '',
        '## Loss Ablations',
        '',
        '| Method | Clean | Mixed | RTTS | Mixed U-Recall | Mixed A-OSE |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for experiment in (
            'task1_full', 'task1_det_only', 'task1_cwd_only'):
        row = results.get(experiment, {})
        mixed = row.get('mixed', {})
        lines.append(
            f'| {experiment} | '
            f'{format_metric(row.get("clean", {}).get("known_ap50"))} | '
            f'{format_metric(mixed.get("known_ap50"))} | '
            f'{format_metric(row.get("rtts", {}).get("known_ap50"))} | '
            f'{format_metric(mixed.get("unknown_recall50"))} | '
            f'{format_metric(mixed.get("a_ose"))} |')
    return '\n'.join(lines) + '\n'


def main():
    repo_root = Path(__file__).resolve().parents[2]
    experiment_root = repo_root / 'work_dirs' / 'experiment7'
    results = {}
    for experiment in EXPERIMENTS:
        experiment_results = {}
        for dataset in DATASETS:
            log_path = (
                experiment_root / experiment / 'tests' / dataset /
                'test_stdout.log')
            if log_path.exists():
                experiment_results[dataset] = parse_log(log_path)
        if experiment_results:
            results[experiment] = experiment_results

    output_json = experiment_root / 'results.json'
    output_markdown = experiment_root / 'results.md'
    output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=True) + '\n')
    output_markdown.write_text(build_markdown(results))
    print(output_json)
    print(output_markdown)


if __name__ == '__main__':
    main()
