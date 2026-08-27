import argparse
import subprocess
import sys
from pathlib import Path


BRANCH_EXPERIMENTS = (
    'task1_full',
    'task1_wo_mamba',
    'task1_wo_kat',
    'task1_wo_conv',
)

LOSS_EXPERIMENTS = (
    'task1_det_only',
    'task1_cwd_only',
)

DATASETS = {
    'clean': (
        'annotations/coco/task1/test_clean_ow.json',
        'task1_test_clean',
        'test_clean.txt',
    ),
    'fog': (
        'annotations/coco/task1/test_fog_ow.json',
        'task1_test_fog',
        'test_fog.txt',
    ),
    'rain': (
        'annotations/coco/task1/test_rain_ow.json',
        'task1_test_rain',
        'test_rain.txt',
    ),
    'snow': (
        'annotations/coco/task1/test_snow_ow.json',
        'task1_test_snow',
        'test_snow.txt',
    ),
    'mixed': (
        'annotations/coco/task1/test_fog_rain_snow_ow.json',
        'task1_test_fog_rain_snow',
        'test_fog_rain_snow.txt',
    ),
    'rtts': (
        'annotations/coco/task1/test_rtts_ow.json',
        'task1_test_rtts',
        'test_rtts.txt',
    ),
    'foggy_driving': (
        'annotations/coco/task1/test_foggy_driving_ow.json',
        'task1_test_foggy_driving',
        'test_foggy_driving.txt',
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Experiment 7 checkpoints on each weather domain.')
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=BRANCH_EXPERIMENTS + LOSS_EXPERIMENTS,
        default=BRANCH_EXPERIMENTS)
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=tuple(DATASETS),
        default=('clean', 'fog', 'rain', 'snow', 'mixed', 'rtts'))
    return parser.parse_args()


def find_best_checkpoint(work_dir: Path) -> Path:
    checkpoints = sorted(work_dir.glob('best_*Known*AP50*.pth'))
    if len(checkpoints) != 1:
        names = ', '.join(path.name for path in checkpoints) or 'none'
        raise RuntimeError(
            f'Expected one best Known AP50 checkpoint in {work_dir}, '
            f'found: {names}')
    return checkpoints[0]


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = (
        repo_root / 'configs' / 'open_world' /
        'weather_owod_mamba_kat' / 'ablations')
    data_root = '/data/weather/WeatherOWOD'
    for experiment in args.experiments:
        config = config_dir / f'{experiment}.py'
        experiment_dir = repo_root / 'work_dirs' / 'experiment7' / experiment
        checkpoint = find_best_checkpoint(experiment_dir)

        for dataset_name in args.datasets:
            ann_file, eval_dir, file_name = DATASETS[dataset_name]
            output_dir = experiment_dir / 'tests' / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / 'test_stdout.log'
            command = [
                sys.executable,
                str(repo_root / 'tools' / 'test.py'),
                str(config),
                str(checkpoint),
                '--work-dir',
                str(output_dir),
                '--cfg-options',
                f'test_dataloader.dataset.ann_file={ann_file}',
                (
                    'test_evaluator.cfg.dataset_root='
                    f'{data_root}/eval_voc/{eval_dir}'
                ),
                f'test_evaluator.cfg.file_name={file_name}',
            ]
            print(
                f'Testing {experiment} on {dataset_name}: '
                f'{" ".join(command)}',
                flush=True)
            with log_path.open('w') as log_file:
                subprocess.run(
                    command,
                    cwd=repo_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True)


if __name__ == '__main__':
    main()
