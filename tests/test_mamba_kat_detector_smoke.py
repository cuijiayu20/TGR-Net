import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def install_framework_stubs(adapter_cls, cwd_loss_cls):
    mmdet = types.ModuleType('mmdet')
    mmdet_structures = types.ModuleType('mmdet.structures')
    mmdet_structures.OptSampleList = object
    mmdet_structures.SampleList = object
    sys.modules['mmdet'] = mmdet
    sys.modules['mmdet.structures'] = mmdet_structures

    class Registry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator

        def build(self, cfg):
            cfg = dict(cfg)
            type_name = cfg.pop('type')
            if type_name == 'MambaKATDegradationAdapter':
                return adapter_cls(**cfg)
            if type_name == 'ChannelWiseDivergenceLoss':
                return cwd_loss_cls(**cfg)
            raise KeyError(type_name)

    mmyolo = types.ModuleType('mmyolo')
    mmyolo_registry = types.ModuleType('mmyolo.registry')
    mmyolo_registry.MODELS = Registry()
    mmyolo_models = types.ModuleType('mmyolo.models')
    mmyolo_detectors = types.ModuleType('mmyolo.models.detectors')

    class YOLODetector(nn.Module):
        def __init__(self, backbone=None, neck=None, bbox_head=None, *args,
                     **kwargs):
            super().__init__()
            self.backbone = backbone
            self.neck = neck
            self.bbox_head = bbox_head
            self.with_neck = neck is not None

        def add_pred_to_datasample(self, samples, results):
            return results

    mmyolo_detectors.YOLODetector = YOLODetector
    sys.modules['mmyolo'] = mmyolo
    sys.modules['mmyolo.registry'] = mmyolo_registry
    sys.modules['mmyolo.models'] = mmyolo_models
    sys.modules['mmyolo.models.detectors'] = mmyolo_detectors


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 16, 1)

    def forward_image(self, x):
        base = self.proj(x)
        return (base, nn.functional.avg_pool2d(base, 2),
                nn.functional.avg_pool2d(base, 4))

    def forward(self, x, texts):
        txt = torch.randn(x.shape[0], 3, 512, device=x.device)
        mask = torch.ones(x.shape[0], 3, dtype=torch.bool, device=x.device)
        return self.forward_image(x), (txt, mask)


class DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 0
        self.training_during_loss = None

    def loss(self, img_feats, txt_feats, txt_masks, batch_data_samples):
        self.training_during_loss = self.training
        return {'loss_det': sum(feat.square().mean() for feat in img_feats)}

    def predict(self,
                img_feats,
                txt_feats,
                txt_masks,
                batch_data_samples,
                rescale=True):
        return [{'ok': True, 'shape0': tuple(img_feats[0].shape)}]

    def forward(self, img_feats, txt_feats, txt_masks):
        return tuple(feat.mean() for feat in img_feats)


class DummyNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.received_grad_enabled_features = False

    def forward(self, img_feats):
        self.received_grad_enabled_features = any(
            feat.requires_grad for feat in img_feats)
        return tuple(feat * self.scale for feat in img_feats)


def test_detector_paired_loss_freezing_and_predict():
    repo_root = Path(__file__).resolve().parents[1]
    adapter_mod = load_module(
        'adapter_mod',
        repo_root / 'yolo_world' / 'models' / 'adapters' /
        'mamba_kat_adapter.py')
    install_framework_stubs(adapter_mod.MambaKATDegradationAdapter, None)
    cwd_mod = load_module(
        'cwd_loss_mod',
        repo_root / 'yolo_world' / 'models' / 'losses' /
        'channel_wise_divergence_loss.py')
    install_framework_stubs(
        adapter_mod.MambaKATDegradationAdapter,
        cwd_mod.ChannelWiseDivergenceLoss)

    pkg = types.ModuleType('fake_detectors')
    pkg.__path__ = [str(repo_root / 'yolo_world' / 'models' / 'detectors')]
    sys.modules['fake_detectors'] = pkg
    load_module('fake_detectors.yolo_world',
                repo_root / 'yolo_world' / 'models' / 'detectors' /
                'yolo_world.py')
    detector_mod = load_module(
        'fake_detectors.yolo_world_mamba_kat',
        repo_root / 'yolo_world' / 'models' / 'detectors' /
        'yolo_world_mamba_kat.py')

    model = detector_mod.YOLOWorldMambaKATDetector(
        backbone=DummyBackbone(),
        neck=DummyNeck(),
        bbox_head=DummyHead(),
        mm_neck=False,
        num_train_classes=3,
        num_test_classes=3,
        degradation_adapter=dict(type='MambaKATDegradationAdapter',
                                 in_channels=[16, 16, 16],
                                 gamma_init=0.1,
                                 use_mamba_ssm=False),
        alignment_loss=dict(
            type='ChannelWiseDivergenceLoss',
            temperature=1.0,
            loss_weight=0.1),
        freeze_yolo_world=True,
    )
    model.train()
    assert model.neck.training is False
    assert model.bbox_head.training is False
    adapter_calls = []
    hook = model.degradation_adapter.register_forward_hook(
        lambda module, inputs, output: adapter_calls.append(output))

    clean = torch.randn(2, 3, 64, 64)
    degraded = clean + 0.05 * torch.randn_like(clean)
    samples = {'texts': [['person', 'car', 'bus'],
                         ['person', 'car', 'bus']]}
    losses = model.loss({
        'clean_inputs': clean,
        'degraded_inputs': degraded
    }, samples)
    assert len(adapter_calls) == 1
    loss = sum(v for v in losses.values())
    loss.backward()

    trainable = [
        name for name, param in model.named_parameters()
        if param.requires_grad
    ]
    base_trainable = [
        name for name in trainable
        if not name.startswith('degradation_adapter')
    ]
    adapter_grads = [
        name for name, param in model.degradation_adapter.named_parameters()
        if param.grad is not None
    ]
    pred = model.predict({'degraded_inputs': degraded}, samples)
    hook.remove()

    assert sorted(losses.keys()) == ['loss_cwd', 'loss_det']
    assert len(adapter_calls) == 2
    assert model.bbox_head.training_during_loss is True
    assert model.neck.received_grad_enabled_features is True
    assert model.neck.scale.grad is None
    assert model.bbox_head.training is False
    assert len(trainable) > 0
    assert len(base_trainable) == 0
    assert len(adapter_grads) > 0
    assert pred[0]['ok'] is True


if __name__ == '__main__':
    test_detector_paired_loss_freezing_and_predict()
