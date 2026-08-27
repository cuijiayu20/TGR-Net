import pytest
import torch

from yolo_world.models.adapters.mamba_kat_adapter import (
    MambaKATDegradationAdapter,
)


def test_adapter_shapes_and_zero_gamma_identity():
    adapter = MambaKATDegradationAdapter(
        in_channels=[16, 32, 64],
        gamma_init=0.0,
        use_mamba_ssm=False)
    adapter.train()
    feats = (
        torch.randn(2, 16, 80, 80),
        torch.randn(2, 32, 40, 40),
        torch.randn(2, 64, 20, 20),
    )
    outs = adapter(feats)

    assert len(outs) == len(feats)
    for out, feat in zip(outs, feats):
        assert out.shape == feat.shape
        torch.testing.assert_close(out, feat)

    loss = sum(out.mean() for out in outs)
    loss.backward()
    assert all(block.gamma.grad is not None for block in adapter.blocks)


def test_adapter_bypass_returns_input_features():
    adapter = MambaKATDegradationAdapter(
        in_channels=[16, 32, 64],
        use_mamba_ssm=False,
        bypass=True)
    feats = (
        torch.randn(2, 16, 80, 80),
        torch.randn(2, 32, 40, 40),
        torch.randn(2, 64, 20, 20),
    )

    outs = adapter(feats)

    assert all(out is feat for out, feat in zip(outs, feats))


@pytest.mark.parametrize(
    ('disabled_branch', 'expected_fused_channels'),
    [('mamba', 32), ('kat', 32), ('conv', 32)])
def test_adapter_single_branch_ablation(disabled_branch,
                                        expected_fused_channels):
    branch_flags = {
        'use_mamba_branch': disabled_branch != 'mamba',
        'use_kat_branch': disabled_branch != 'kat',
        'use_conv_branch': disabled_branch != 'conv',
    }
    adapter = MambaKATDegradationAdapter(
        in_channels=[16],
        use_mamba_ssm=False,
        **branch_flags)
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)

    output = adapter((feature,))[0]
    loss = output.mean()
    loss.backward()

    assert output.shape == feature.shape
    assert adapter.blocks[0].fuse[0].in_channels == expected_fused_channels
    assert feature.grad is not None


def test_adapter_rejects_disabling_all_branches():
    with pytest.raises(ValueError, match='At least one'):
        MambaKATDegradationAdapter(
            in_channels=[16],
            use_mamba_ssm=False,
            use_mamba_branch=False,
            use_kat_branch=False,
            use_conv_branch=False)


if __name__ == '__main__':
    test_adapter_shapes_and_zero_gamma_identity()
    test_adapter_bypass_returns_input_features()
