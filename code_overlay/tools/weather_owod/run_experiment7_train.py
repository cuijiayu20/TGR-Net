import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = (
    'task1_full',
    'task1_wo_mamba',
    'task1_wo_kat',
    'task1_wo_conv',
    'task1_det_only',
    'task1_cwd_only',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the Experiment 7 ablation training matrix.')
    parser.add_argument(
        '--experiments',
        nargs='+',
        choices=EXPERIMENTS,
        default=EXPERIMENTS)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = (
        repo_root / 'configs' / 'open_world' /
        'weather_owod_mamba_kat' / 'ablations')

    for experiment in args.experiments:
        config = config_dir / f'{experiment}.py'
        work_dir = repo_root / 'work_dirs' / 'experiment7' / experiment
        work_dir.mkdir(parents=True, exist_ok=True)
        log_path = work_dir / 'train_stdout.log'
        command = [
            sys.executable,
            str(repo_root / 'tools' / 'train.py'),
            str(config),
            '--work-dir',
            str(work_dir),
        ]
        print(f'Running {experiment}: {" ".join(command)}', flush=True)
        with log_path.open('w') as log_file:
            subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True)


if __name__ == '__main__':
    main()
