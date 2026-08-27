# Copyright (c) Tencent Inc. All rights reserved.
import json
from pathlib import Path
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag])

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class RandomWeatherVariant:
    """Randomly switch a synthetic weather variant before image loading.

    The annotation remains shared across variants. Use this transform before
    ``LoadImageFromFile`` with a single-weather annotation file, so each clean
    image contributes one sampled weather variant per epoch instead of three
    duplicated fog/rain/snow samples.
    """

    def __init__(self,
                 weather_names: Tuple[str, ...] = ('fog', 'rain', 'snow'),
                 suffixes: dict = None,
                 strict: bool = True) -> None:
        self.weather_names = tuple(weather_names)
        self.suffixes = suffixes or {'fog': '', 'rain': '', 'snow': ''}
        self.strict = strict

    def _switch_path(self, img_path: str, target_weather: str) -> Path:
        path = Path(img_path)
        parts = list(path.parts)
        for idx, part in enumerate(parts):
            if part in self.weather_names:
                parts[idx] = target_weather
                break
        else:
            raise ValueError(
                f'Cannot infer weather variant from image path: {img_path}')

        switched_path = Path(*parts)
        stem = switched_path.stem
        for suffix in self.suffixes.values():
            if suffix and stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        target_suffix = self.suffixes.get(target_weather, '')
        switched_path = switched_path.with_name(
            stem + target_suffix + switched_path.suffix)
        return switched_path

    def __call__(self, results: dict) -> dict:
        if 'img_path' not in results:
            raise KeyError('RandomWeatherVariant expects img_path before '
                           'LoadImageFromFile.')

        source_path = results['img_path']
        target_weather = random.choice(self.weather_names)
        target_path = self._switch_path(source_path, target_weather)
        if not target_path.exists():
            if self.strict:
                raise FileNotFoundError(
                    f'Weather variant not found for {source_path}: '
                    f'{target_path}')
            return results

        results['img_path'] = str(target_path)
        results['weather_type'] = target_weather
        return results


@TRANSFORMS.register_module()
class LoadPairedCleanImage:
    """Load the clean counterpart for a packed degraded training sample.

    This transform is intended to run after ``mmdet.PackDetInputs``. It keeps
    the degraded tensor as ``degraded_inputs`` and adds a resized clean tensor as
    ``clean_inputs`` so the model can compute clean-degraded feature alignment.
    """

    def __init__(self,
                 clean_weather_name: str = 'clean',
                 degraded_weather_names: Tuple[str, ...] = ('fog', 'rain',
                                                            'snow'),
                 remove_suffixes: Tuple[str, ...] = ('_rain', '_snow')) -> None:
        self.clean_weather_name = clean_weather_name
        self.degraded_weather_names = tuple(degraded_weather_names)
        self.remove_suffixes = tuple(remove_suffixes)

    def _clean_path_from_degraded(self, img_path: str) -> Path:
        path = Path(img_path)
        parts = list(path.parts)
        for idx, part in enumerate(parts):
            if part in self.degraded_weather_names:
                parts[idx] = self.clean_weather_name
                break
        else:
            raise ValueError(
                f'Cannot infer clean image path from degraded path: {img_path}')

        clean_path = Path(*parts)
        stem = clean_path.stem
        for suffix in self.remove_suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                clean_path = clean_path.with_name(stem + clean_path.suffix)
                break
        return clean_path

    @staticmethod
    def _resize_to_tensor_shape(image: np.ndarray,
                                reference: torch.Tensor) -> torch.Tensor:
        _, height, width = reference.shape
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height),
                               interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

    def __call__(self, results: dict) -> dict:
        if 'inputs' not in results or 'data_samples' not in results:
            raise KeyError('LoadPairedCleanImage expects packed results with '
                           'inputs and data_samples.')

        data_sample = results['data_samples']
        img_path = data_sample.metainfo.get('img_path', None)
        if img_path is None:
            raise KeyError('img_path must be included in PackDetInputs '
                           'meta_keys before LoadPairedCleanImage.')

        clean_path = self._clean_path_from_degraded(img_path)
        if not clean_path.exists():
            raise FileNotFoundError(
                f'Clean counterpart not found for {img_path}: {clean_path}')

        clean_img = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if clean_img is None:
            raise FileNotFoundError(f'Failed to read clean image: {clean_path}')

        degraded_inputs = results['inputs']
        results['degraded_inputs'] = degraded_inputs
        results['clean_inputs'] = self._resize_to_tensor_shape(
            clean_img, degraded_inputs)
        data_sample.set_metainfo({'clean_img_path': str(clean_path)})
        return results
