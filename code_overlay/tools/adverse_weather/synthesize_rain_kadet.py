"""KADet-style rain synthesis for paired adverse-weather datasets.

This script adapts KADet/Rain_Synthesizing.py into a reusable CLI. It creates
rainy images from a clean image directory and keeps the original annotations
unchanged. Output filenames default to ``<stem>_rain<ext>``.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def get_noise(img: np.ndarray, value: int = 10) -> np.ndarray:
    """Generate sparse rain seed noise."""
    noise = np.random.uniform(0, 256, img.shape[:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
    kernel = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
    return cv2.filter2D(noise, -1, kernel)


def rain_blur(noise: np.ndarray,
              length: int = 50,
              angle: int = 0,
              width: int = 3) -> np.ndarray:
    """Apply a motion blur kernel to sparse noise to form rain streaks."""
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45,
                                    1 - length / 100.0)
    diag = np.diag(np.ones(length))
    kernel = cv2.warpAffine(diag, trans, (length, length))
    kernel = cv2.GaussianBlur(kernel, (width, width), 0)
    blurred = cv2.filter2D(noise, -1, kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return np.array(blurred, dtype=np.uint8)


def alpha_rain(rain: np.ndarray, img: np.ndarray, beta: float = 0.8) -> np.ndarray:
    """Blend rain streaks into a BGR image."""
    rain = np.expand_dims(rain.astype(np.float32), 2)
    rain_result = img.copy().astype(np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    return np.uint8(np.clip(rain_result, 0, 255))


def iter_images(clean_dir: Path):
    for path in sorted(clean_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate KADet-style rainy images from clean images.')
    parser.add_argument('--clean-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--suffix', default='_rain')
    parser.add_argument('--value-min', default=100, type=int)
    parser.add_argument('--value-max', default=200, type=int)
    parser.add_argument('--angle-min', default=-30, type=int)
    parser.add_argument('--angle-max', default=30, type=int)
    parser.add_argument('--length', default=50, type=int)
    parser.add_argument('--width', default=3, type=int)
    parser.add_argument('--beta', default=0.8, type=float)
    parser.add_argument('--seed', default=None, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in iter_images(args.clean_dir):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f'Skip unreadable image: {image_path}')
            continue

        value = random.randint(args.value_min, args.value_max)
        angle = random.randint(args.angle_min, args.angle_max)
        noise = get_noise(img, value=value)
        rain = rain_blur(noise,
                         length=args.length,
                         angle=angle,
                         width=args.width)
        rainy = alpha_rain(rain, img, beta=args.beta)

        out_name = f'{image_path.stem}{args.suffix}{image_path.suffix}'
        cv2.imwrite(str(args.output_dir / out_name), rainy)
        count += 1

    print(f'Generated {count} rainy images in {args.output_dir}')


if __name__ == '__main__':
    main()
