import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def load_cwd_loss():
    registry_module = types.ModuleType('mmyolo.registry')

    class Registry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator

    registry_module.MODELS = Registry()
    mmyolo_module = types.ModuleType('mmyolo')
    sys.modules.setdefault('mmyolo', mmyolo_module)
    sys.modules['mmyolo.registry'] = registry_module

    path = (Path(__file__).resolve().parents[1] / 'yolo_world' / 'models' /
            'losses' / 'channel_wise_divergence_loss.py')
    spec = importlib.util.spec_from_file_location('cwd_loss_module', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ChannelWiseDivergenceLoss


def test_cwd_matches_reference_formula_and_gradient_direction():
    loss_cls = load_cwd_loss()
    temperature = 2.0
    loss_weight = 0.3
    loss_fn = loss_cls(
        temperature=temperature, loss_weight=loss_weight)

    student = torch.tensor(
        [[[[1.0, 0.0], [0.5, -0.5]], [[0.0, 1.0], [-1.0, 0.5]]]],
        requires_grad=True)
    teacher = torch.tensor(
        [[[[0.0, 1.0], [0.5, -0.5]], [[1.0, 0.0], [-0.5, 0.5]]]],
        requires_grad=True)

    loss = loss_fn(student, teacher)
    student_log_prob = F.log_softmax(
        student.flatten(2) / temperature, dim=2)
    teacher_prob = F.softmax(
        teacher.detach().flatten(2) / temperature, dim=2)
    reference = (
        loss_weight * F.kl_div(
            student_log_prob, teacher_prob, reduction='sum') *
        temperature**2 / (student.shape[0] * student.shape[1]))

    assert torch.allclose(loss, reference)
    assert loss.item() > 0
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_cwd_is_zero_for_identical_features():
    loss_cls = load_cwd_loss()
    loss_fn = loss_cls(temperature=1.0, loss_weight=1.0)
    feature = torch.randn(2, 4, 3, 3)

    loss = loss_fn((feature, feature), (feature.clone(), feature.clone()))

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
