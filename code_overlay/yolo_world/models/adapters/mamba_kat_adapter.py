from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from mmengine.model import BaseModule
    from mmyolo.registry import MODELS
except ImportError:  # Allows standalone unit tests before MMYOLO is installed.
    class BaseModule(nn.Module):
        def __init__(self, init_cfg=None):
            super().__init__()

    class _RegistryStub:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator

    MODELS = _RegistryStub()

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels,
                      channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MambaLikeFallback(nn.Module):
    """Pure PyTorch sequence mixer used when mamba-ssm is unavailable."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_channels * 2)
        self.dwconv = nn.Conv1d(hidden_channels,
                                hidden_channels,
                                kernel_size=5,
                                padding=2,
                                groups=hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = value.transpose(1, 2)
        value = self.dwconv(value).transpose(1, 2)
        x = value * torch.sigmoid(gate)
        return residual + self.out_proj(x)


class MambaFeatureBranch(nn.Module):
    def __init__(self,
                 channels: int,
                 use_mamba_ssm: bool = True,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2) -> None:
        super().__init__()
        if use_mamba_ssm and Mamba is not None:
            self.sequence_model = Mamba(d_model=channels,
                                        d_state=d_state,
                                        d_conv=d_conv,
                                        expand=expand)
        else:
            self.sequence_model = MambaLikeFallback(channels,
                                                    expansion=expand)
        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.sequence_model(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return self.out_norm(x)


class KATFeatureBranch(nn.Module):
    """Lightweight KAN/KAT-style nonlinear channel adapter."""

    def __init__(self, channels: int, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        hidden_channels = max(16, int(channels * hidden_ratio))
        self.pre = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.basis = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        ])
        self.post = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)
        linear = self.basis[0](x)
        quadratic = self.basis[1](x * x)
        sinusoidal = self.basis[2](torch.sin(x))
        return self.post(linear + quadratic + sinusoidal)


class MambaKATScaleAdapter(nn.Module):
    def __init__(self,
                 channels: int,
                 gamma_init: float = 0.1,
                 use_mamba_ssm: bool = True,
                 kat_hidden_ratio: float = 0.5,
                 use_mamba_branch: bool = True,
                 use_kat_branch: bool = True,
                 use_conv_branch: bool = True) -> None:
        super().__init__()
        self.use_mamba_branch = use_mamba_branch
        self.use_kat_branch = use_kat_branch
        self.use_conv_branch = use_conv_branch
        num_branches = sum(
            (use_mamba_branch, use_kat_branch, use_conv_branch))
        if num_branches == 0:
            raise ValueError('At least one adapter branch must be enabled.')

        self.mamba_branch = (
            MambaFeatureBranch(
                channels=channels, use_mamba_ssm=use_mamba_ssm)
            if use_mamba_branch else None)
        self.kat_branch = (
            KATFeatureBranch(
                channels=channels, hidden_ratio=kat_hidden_ratio)
            if use_kat_branch else None)
        self.conv_branch = (
            DepthwiseSeparableConv(channels) if use_conv_branch else None)
        fused_channels = channels * num_branches
        self.gate = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: Tensor) -> Tensor:
        branch_outputs = []
        if self.mamba_branch is not None:
            branch_outputs.append(self.mamba_branch(x))
        if self.kat_branch is not None:
            branch_outputs.append(self.kat_branch(x))
        if self.conv_branch is not None:
            branch_outputs.append(self.conv_branch(x))
        fused = torch.cat(branch_outputs, dim=1)
        return x + self.gamma * self.gate(fused) * self.fuse(fused)


@MODELS.register_module()
class MambaKATDegradationAdapter(BaseModule):
    """Multi-scale adapter inserted between backbone features and the neck."""

    def __init__(self,
                 in_channels: Sequence[int],
                 gamma_init: float = 0.1,
                 use_mamba_ssm: bool = True,
                 kat_hidden_ratio: float = 0.5,
                 use_mamba_branch: bool = True,
                 use_kat_branch: bool = True,
                 use_conv_branch: bool = True,
                 bypass: bool = False,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = tuple(in_channels)
        self.bypass = bypass
        self.blocks = nn.ModuleList([
            MambaKATScaleAdapter(channels=channels,
                                 gamma_init=gamma_init,
                                 use_mamba_ssm=use_mamba_ssm,
                                 kat_hidden_ratio=kat_hidden_ratio,
                                 use_mamba_branch=use_mamba_branch,
                                 use_kat_branch=use_kat_branch,
                                 use_conv_branch=use_conv_branch)
            for channels in self.in_channels
        ])

    def forward(self, feats: Sequence[Tensor]) -> Tuple[Tensor, ...]:
        if len(feats) != len(self.blocks):
            raise ValueError(
                f'Expected {len(self.blocks)} feature maps, got {len(feats)}.')
        if self.bypass:
            return tuple(feats)
        return tuple(block(feat) for block, feat in zip(self.blocks, feats))
