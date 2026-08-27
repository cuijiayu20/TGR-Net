"""KADet-style snow synthesis for paired adverse-weather datasets.

This script adapts KADet/Snow_Synthesizing.py into a reusable CLI. It overlays
random snow masks on clean images and keeps annotations unchanged. Output
filenames default to ``<stem>_snow<ext>``.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def iter_images(image_dir: Path):
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def add_snow(clean_path: Path,
             snow_mask_path: Path,
             beta_min: float = 0.5,
             beta_max: float = 1.0) -> np.ndarray:
    clean = cv2.imread(str(clean_path))
    snow_mask = cv2.imread(str(snow_mask_path))
    if clean is None:
        raise ValueError(f'Failed to read clean image: {clean_path}')
    if snow_mask is None:
        raise ValueError(f'Failed to read snow mask: {snow_mask_path}')

    snow_mask = cv2.resize(snow_mask, (clean.shape[1], clean.shape[0]))
    beta = random.uniform(beta_min, beta_max)
    corrupt = cv2.addWeighted(clean, 1.0, snow_mask, beta, 0)
    return np.uint8(corrupt.clip(0, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate KADet-style snowy images from clean images.')
    parser.add_argument('--clean-dir', required=True, type=Path)
    parser.add_argument('--snow-mask-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--suffix', default='_snow')
    parser.add_argument('--beta-min', default=0.5, type=float)
    parser.add_argument('--beta-max', default=1.0, type=float)
    parser.add_argument('--seed', default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    masks = list(iter_images(args.snow_mask_dir))
    if not masks:
        raise FileNotFoundError(f'No snow masks found in {args.snow_mask_dir}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for clean_path in iter_images(args.clean_dir):
        mask_path = random.choice(masks)
        snow_image = add_snow(clean_path,
                              mask_path,
                              beta_min=args.beta_min,
                              beta_max=args.beta_max)
        out_name = f'{clean_path.stem}{args.suffix}{clean_path.suffix}'
        cv2.imwrite(str(args.output_dir / out_name), snow_image)
        count += 1

    print(f'Generated {count} snowy images in {args.output_dir}')


if __name__ == '__main__':
    main()
