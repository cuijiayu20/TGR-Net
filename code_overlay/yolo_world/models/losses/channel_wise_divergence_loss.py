from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmyolo.registry import MODELS


@MODELS.register_module()
class ChannelWiseDivergenceLoss(nn.Module):
    """Channel-wise spatial distribution distillation loss.

    For each channel, CWD converts the H x W responses into a probability
    distribution and minimizes KL(teacher || student). The clean feature is the
    detached teacher and the enhanced degraded feature is the student.
    """

    def __init__(self,
                 temperature: float = 1.0,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be greater than zero.')
        self.temperature = float(temperature)
        self.loss_weight = float(loss_weight)

    def _single_scale_loss(self, student: Tensor, teacher: Tensor) -> Tensor:
        if student.shape != teacher.shape:
            raise ValueError(
                f'Student and teacher shapes must match, got '
                f'{tuple(student.shape)} and {tuple(teacher.shape)}.')
        if student.ndim != 4:
            raise ValueError(
                f'CWD expects BCHW features, got {student.ndim} dimensions.')

        batch_size, channels = student.shape[:2]
        temperature = self.temperature
        student_log_prob = F.log_softmax(
            student.float().flatten(2) / temperature, dim=2)
        teacher_prob = F.softmax(
            teacher.detach().float().flatten(2) / temperature, dim=2)

        divergence = F.kl_div(
            student_log_prob, teacher_prob, reduction='sum')
        return divergence * temperature**2 / (batch_size * channels)

    def forward(
        self,
        student_features: Union[Tensor, Sequence[Tensor]],
        teacher_features: Union[Tensor, Sequence[Tensor]],
    ) -> Tensor:
        if isinstance(student_features, Tensor):
            student_features = (student_features, )
        if isinstance(teacher_features, Tensor):
            teacher_features = (teacher_features, )
        if len(student_features) != len(teacher_features):
            raise ValueError(
                f'Expected equal feature counts, got {len(student_features)} '
                f'and {len(teacher_features)}.')

        losses = [
            self._single_scale_loss(student, teacher)
            for student, teacher in zip(student_features, teacher_features)
        ]
        return self.loss_weight * torch.stack(losses).mean()
