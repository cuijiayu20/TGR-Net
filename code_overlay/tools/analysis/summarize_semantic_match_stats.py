import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    'correct_logit',
    'correct_prob',
    'margin_logit',
    'top1_correct',
    'entropy',
)

TABLE_METRICS = (
    ('correct_logit', r'$\overline{R}$', 4, 1.0),
    ('margin_logit', r'$\overline{M}$', 4, 1.0),
    ('top1_correct', r'$\overline{A}$ (\%)', 2, 100.0),
    ('entropy', r'$\overline{H}$', 4, 1.0),
)

SPLIT_LABELS = {
    'clean': 'VOC clean',
    'fog': 'VOC fog',
    'rain': 'VOC rain',
    'snow': 'VOC snow',
    'mixed': 'VOC fog/rain/snow',
    'rtts': 'RTTS',
}

SPLIT_ORDER = ('clean', 'fog', 'rain', 'snow', 'mixed', 'rtts')
MODEL_ORDER = ('OW-OVD', 'Mamba-KAT')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize semantic matching statistic CSV files.')
    parser.add_argument('--stats-dir',
                        default='work_dirs/semantic_match_stats/full')
    return parser.parse_args()


def read_rows(stats_dir):
    rows = []
    for csv_path in sorted(Path(stats_dir).glob('*_gt_center_stats.csv')):
        with csv_path.open(newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['_source_csv'] = str(csv_path)
                for key in METRICS:
                    row[key] = float(row[key])
                row['class_id'] = int(float(row['class_id']))
                rows.append(row)
    return rows


def mean(values):
    return sum(values) / len(values) if values else math.nan


def stdev(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu)**2 for value in values) /
                     (len(values) - 1))


def describe(values):
    n = len(values)
    if n == 0:
        return {
            'n': 0,
            'mean': math.nan,
            'sd': math.nan,
            'se': math.nan,
            'ci95_low': math.nan,
            'ci95_high': math.nan,
        }
    mu = mean(values)
    sd = stdev(values)
    se = sd / math.sqrt(n) if n > 0 else math.nan
    half_width = 1.96 * se if n > 1 else 0.0
    return {
        'n': n,
        'mean': mu,
        'sd': sd,
        'se': se,
        'ci95_low': mu - half_width,
        'ci95_high': mu + half_width,
    }


def group_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['split'])].append(row)

    summaries = []
    for (model, split), items in sorted(
            grouped.items(),
            key=lambda item: (SPLIT_ORDER.index(item[0][1])
                              if item[0][1] in SPLIT_ORDER else 999,
                              MODEL_ORDER.index(item[0][0])
                              if item[0][0] in MODEL_ORDER else 999)):
        entry = {
            'model': model,
            'split': split,
            'n': len(items),
        }
        for metric in METRICS:
            stats = describe([item[metric] for item in items])
            for key, value in stats.items():
                if key == 'n':
                    continue
                entry[f'{metric}_{key}'] = value
        summaries.append(entry)
    return summaries


def pair_key_builder(rows):
    counters = Counter()
    keys = {}
    for idx, row in enumerate(rows):
        gt_index = row.get('gt_index', '')
        if gt_index in (None, ''):
            counter_key = (row.get('img_id', ''), row.get('class_id', ''))
            gt_index = counters[counter_key]
            counters[counter_key] += 1
        key = (str(row.get('img_id', '')), int(row['class_id']),
               str(gt_index))
        keys[idx] = key
    return keys


def index_records(rows):
    keys = pair_key_builder(rows)
    indexed = {}
    duplicates = 0
    for idx, row in enumerate(rows):
        key = keys[idx]
        if key in indexed:
            duplicates += 1
            key = key + (str(duplicates), )
        indexed[key] = row
    return indexed


def paired_delta_rows(grouped):
    output = []

    splits = sorted({split for _, split in grouped},
                    key=lambda split: SPLIT_ORDER.index(split)
                    if split in SPLIT_ORDER else 999)
    for split in splits:
        left = grouped.get(('OW-OVD', split))
        right = grouped.get(('Mamba-KAT', split))
        if not left or not right:
            continue
        left_index = index_records(left)
        right_index = index_records(right)
        common_keys = sorted(set(left_index) & set(right_index))
        for metric in METRICS:
            deltas = [
                right_index[key][metric] - left_index[key][metric]
                for key in common_keys
            ]
            stats = describe(deltas)
            output.append({
                'comparison': 'Mamba-KAT minus OW-OVD',
                'model': 'Mamba-KAT',
                'split': split,
                'baseline_split': split,
                'metric': metric,
                'n_pairs': len(deltas),
                'mean_delta': stats['mean'],
                'sd_delta': stats['sd'],
                'se_delta': stats['se'],
                'ci95_low': stats['ci95_low'],
                'ci95_high': stats['ci95_high'],
                'fraction_positive': mean([1.0 if d > 0 else 0.0
                                           for d in deltas])
                if deltas else math.nan,
            })

    for model in MODEL_ORDER:
        clean = grouped.get((model, 'clean'))
        if not clean:
            continue
        clean_index = index_records(clean)
        for split in ('fog', 'rain', 'snow'):
            weather = grouped.get((model, split))
            if not weather:
                continue
            weather_index = index_records(weather)
            common_keys = sorted(set(clean_index) & set(weather_index))
            for metric in METRICS:
                deltas = [
                    weather_index[key][metric] - clean_index[key][metric]
                    for key in common_keys
                ]
                stats = describe(deltas)
                output.append({
                    'comparison': 'weather minus clean',
                    'model': model,
                    'split': split,
                    'baseline_split': 'clean',
                    'metric': metric,
                    'n_pairs': len(deltas),
                    'mean_delta': stats['mean'],
                    'sd_delta': stats['sd'],
                    'se_delta': stats['se'],
                    'ci95_low': stats['ci95_low'],
                    'ci95_high': stats['ci95_high'],
                    'fraction_positive': mean([1.0 if d > 0 else 0.0
                                               for d in deltas])
                    if deltas else math.nan,
                })

    return output


def independent_condition_deltas(summaries):
    by_group = {(row['model'], row['split']): row for row in summaries}
    output = []
    for model in MODEL_ORDER:
        clean = by_group.get((model, 'clean'))
        if not clean:
            continue
        for split in SPLIT_ORDER:
            if split == 'clean':
                continue
            current = by_group.get((model, split))
            if not current:
                continue
            for metric in METRICS:
                output.append({
                    'comparison': 'split mean minus clean mean',
                    'model': model,
                    'split': split,
                    'baseline_split': 'clean',
                    'metric': metric,
                    'n': current['n'],
                    'baseline_n': clean['n'],
                    'mean_delta':
                    current[f'{metric}_mean'] - clean[f'{metric}_mean'],
                })
    return output


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value, digits):
    if isinstance(value, str):
        return value
    if value is None or not math.isfinite(value):
        return '--'
    return f'{value:.{digits}f}'


def best_by_split(summaries):
    grouped = defaultdict(list)
    for row in summaries:
        grouped[row['split']].append(row)
    best = {}
    for split, rows in grouped.items():
        for metric, _, _, scale in TABLE_METRICS:
            values = [(row['model'], row[f'{metric}_mean'] * scale)
                      for row in rows]
            if not values:
                continue
            if metric == 'entropy':
                target = min(value for _, value in values)
            else:
                target = max(value for _, value in values)
            for model, value in values:
                if abs(value - target) <= 1e-12:
                    best[(split, model, metric)] = True
    return best


def latex_table(summaries):
    best = best_by_split(summaries)
    header = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Full-sample auxiliary statistics for visual-semantic '
        r'matching reliability.}',
        r'\label{tab:semantic_match_full}',
        r'\begin{tabular}{llrrrrr}',
        r'\toprule',
        r'Dataset & Method & $N$ & $\overline{R}$ & $\overline{M}$ & '
        r'$\overline{A}$ (\%) & $\overline{H}$ \\',
        r'\midrule',
    ]
    lines = []
    rows = sorted(summaries,
                  key=lambda row: (SPLIT_ORDER.index(row['split'])
                                   if row['split'] in SPLIT_ORDER else 999,
                                   MODEL_ORDER.index(row['model'])
                                   if row['model'] in MODEL_ORDER else 999))
    for row in rows:
        cells = [
            SPLIT_LABELS.get(row['split'], row['split']),
            row['model'],
            str(row['n']),
        ]
        for metric, _, digits, scale in TABLE_METRICS:
            value = row[f'{metric}_mean'] * scale
            text = format_float(value, digits)
            if best.get((row['split'], row['model'], metric)):
                text = r'\textbf{' + text + '}'
            cells.append(text)
        lines.append(' & '.join(cells) + r' \\')
    footer = [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(header + lines + footer)


def compact_jsonable(rows):
    output = []
    for row in rows:
        entry = {}
        for key, value in row.items():
            if isinstance(value, float):
                entry[key] = None if not math.isfinite(value) else value
            else:
                entry[key] = value
        output.append(entry)
    return output


def main():
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    rows = read_rows(stats_dir)
    if not rows:
        raise SystemExit(f'No *_gt_center_stats.csv files found in {stats_dir}')

    summaries = group_summary(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['split'])].append(row)
    paired = paired_delta_rows(grouped)
    independent = independent_condition_deltas(summaries)

    summary_fields = ['model', 'split', 'n']
    for metric in METRICS:
        summary_fields.extend([
            f'{metric}_mean', f'{metric}_sd', f'{metric}_se',
            f'{metric}_ci95_low', f'{metric}_ci95_high'
        ])
    write_csv(stats_dir / 'aggregate_summary.csv', summaries, summary_fields)

    delta_fields = [
        'comparison', 'model', 'split', 'baseline_split', 'metric', 'n_pairs',
        'mean_delta', 'sd_delta', 'se_delta', 'ci95_low', 'ci95_high',
        'fraction_positive'
    ]
    write_csv(stats_dir / 'paired_deltas.csv', paired, delta_fields)

    independent_fields = [
        'comparison', 'model', 'split', 'baseline_split', 'metric', 'n',
        'baseline_n', 'mean_delta'
    ]
    write_csv(stats_dir / 'independent_condition_deltas.csv', independent,
              independent_fields)

    with (stats_dir / 'aggregate_summary.json').open('w') as f:
        json.dump(
            {
                'summary': compact_jsonable(summaries),
                'paired_deltas': compact_jsonable(paired),
                'independent_condition_deltas': compact_jsonable(independent),
            },
            f,
            indent=2)

    (stats_dir / 'latex_table_basic.tex').write_text(latex_table(summaries))
    print(json.dumps(
        {
            'stats_dir': str(stats_dir),
            'rows': len(rows),
            'groups': len(summaries),
            'paired_delta_rows': len(paired),
            'outputs': [
                'aggregate_summary.csv',
                'paired_deltas.csv',
                'independent_condition_deltas.csv',
                'aggregate_summary.json',
                'latex_table_basic.tex',
            ],
        },
        indent=2))


if __name__ == '__main__':
    main()
