import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASETS = {
    'clean': 'annotations/coco/task1/test_clean_ow.json',
    'fog': 'annotations/coco/task1/test_fog_ow.json',
    'rain': 'annotations/coco/task1/test_rain_ow.json',
    'snow': 'annotations/coco/task1/test_snow_ow.json',
    'mixed': 'annotations/coco/task1/test_fog_rain_snow_ow.json',
    'rtts': 'annotations/coco/task1/test_rtts_ow.json',
}


MODELS = {
    'OW-OVD': {
        'config': 'configs/open_world/weather_owod_mamba_kat/our/'
        'eval_task1_no_adapter.py',
        'checkpoint': 'pretrained_models/'
        'yolo_world_v2_l_obj365v1_goldg_pretrain-a82b1fe3.pth',
    },
    'Mamba-KAT': {
        'config': 'configs/open_world/weather_owod_mamba_kat/ablations/'
        'task1_full.py',
        'checkpoint': 'work_dirs/experiment7/task1_full/'
        'best_Known AP50_epoch_6.pth',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run full-sample semantic matching statistics.')
    parser.add_argument('--data-root', default='/data/weather/WeatherOWOD')
    parser.add_argument('--out-dir',
                        default='work_dirs/semantic_match_stats/full')
    parser.add_argument('--splits',
                        nargs='+',
                        default=['clean', 'fog', 'rain', 'snow', 'rtts'],
                        choices=sorted(DATASETS))
    parser.add_argument('--models',
                        nargs='+',
                        default=['OW-OVD', 'Mamba-KAT'],
                        choices=sorted(MODELS))
    parser.add_argument('--max-images', type=int, default=-1)
    parser.add_argument('--num-known', type=int, default=5)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--no-summary', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(f'{description} not found: {path}')


def run_one(repo_root, args, model_name, split):
    model_cfg = MODELS[model_name]
    out_dir = repo_root / args.out_dir
    stem = f'{model_name}_{split}'.replace('/', '_')
    csv_path = out_dir / f'{stem}_gt_center_stats.csv'
    json_path = out_dir / f'{stem}_summary.json'
    if args.skip_existing and csv_path.exists() and json_path.exists():
        print(f'[skip] {model_name} {split}: {csv_path}')
        return

    config = repo_root / model_cfg['config']
    checkpoint = repo_root / model_cfg['checkpoint']
    require_file(config, 'Config')
    require_file(checkpoint, 'Checkpoint')

    work_dir = out_dir / f'{stem}_work'
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'{stem}.log'

    cmd = [
        args.python,
        str(repo_root / 'tools/analysis/semantic_match_stats.py'),
        str(config),
        str(checkpoint),
        '--ann-file',
        DATASETS[split],
        '--split',
        split,
        '--model-name',
        model_name,
        '--data-root',
        args.data_root,
        '--work-dir',
        str(work_dir),
        '--out-dir',
        str(out_dir),
        '--num-known',
        str(args.num_known),
        '--max-images',
        str(args.max_images),
    ]

    print(f'[run] {model_name} {split}')
    print('      ' + ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with log_path.open('w') as log_file:
        log_file.write(' '.join(cmd) + '\n\n')
        log_file.flush()
        subprocess.run(cmd,
                       cwd=repo_root,
                       env=env,
                       stdout=log_file,
                       stderr=subprocess.STDOUT,
                       check=True)
    print(f'[done] {model_name} {split}: {csv_path}')


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    for model_name in args.models:
        for split in args.splits:
            run_one(repo_root, args, model_name, split)

    if not args.no_summary:
        summary_script = repo_root / 'tools/analysis/summarize_semantic_match_stats.py'
        require_file(summary_script, 'Summary script')
        cmd = [
            args.python,
            str(summary_script),
            '--stats-dir',
            str(repo_root / args.out_dir),
        ]
        print('[summary] ' + ' '.join(cmd))
        subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == '__main__':
    main()
