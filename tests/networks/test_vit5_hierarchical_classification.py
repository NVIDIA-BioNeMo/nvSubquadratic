"""Tests for ViT5HierarchicalClassificationNet.

Uses a minimal stand-in mixer (linear over channels) so the tests focus on
the network's hierarchical plumbing — stages, patch merging, register-row
preservation, GAP readout — rather than on any specific mixer's correctness.
"""

import pytest
import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patch_merging import PatchMerging
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_hierarchical_classification import (
    ViT5HierarchicalClassificationNet,
)


class _LinearMixer(nn.Module):
    """Trivial sequence mixer used to validate the network's plumbing.

    Applies the same per-channel linear at every position. No spatial mixing —
    we don't need it to test the hierarchy itself.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(x)

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        D = self.proj.in_features
        return 2 * num_tokens * D * D


def _make_block_cfg(hidden_dim: int) -> LazyConfig:
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(_LinearMixer)(hidden_dim=hidden_dim),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6, use_quack=False),
        mlp_cfg=LazyConfig(MLP)(
            dim=hidden_dim,
            activation="gelu",
            expansion_factor=4.0,
            bias=False,
            dropout_cfg=LazyConfig(nn.Dropout)(p=0.0),
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6, use_quack=False),
        hidden_dim=hidden_dim,
        layer_scale_init=1e-4,
        drop_path_rate=0.0,
    )


def _make_pm_cfg(in_dim: int, out_dim: int, grid: int, num_regs: int, has_reg_row: bool) -> LazyConfig:
    return LazyConfig(PatchMerging)(
        in_dim=in_dim,
        out_dim=out_dim,
        grid_h=grid,
        grid_w=grid,
        norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6, use_quack=False),
        num_registers=num_regs,
        has_register_row=has_reg_row,
    )


def _build_net(layout: str, image_size: int = 32, p0: int = 4, num_registers: int = 0):
    """Build a small 4-stage hierarchical net suitable for unit testing."""
    stage_dims = [16, 32, 64, 128]
    stage_depths = [1, 1, 2, 1]
    num_stages = len(stage_dims)
    initial_grid = image_size // p0  # 8
    grids = [initial_grid // (2**i) for i in range(num_stages)]  # 8,4,2,1

    block_cfgs = [_make_block_cfg(d) for d in stage_dims]
    pm_cfgs = [
        _make_pm_cfg(
            in_dim=stage_dims[i],
            out_dim=stage_dims[i + 1],
            grid=grids[i],
            num_regs=num_registers,
            has_reg_row=(layout == "register_row"),
        )
        for i in range(num_stages - 1)
    ]

    return ViT5HierarchicalClassificationNet(
        in_channels=3,
        num_classes=10,
        image_size=image_size,
        initial_patch_size=p0,
        stage_dims=stage_dims,
        stage_depths=stage_depths,
        stage_block_cfgs=block_cfgs,
        patch_merge_cfgs=pm_cfgs,
        norm_cfg=LazyConfig(RMSNorm)(dim=stage_dims[-1], eps=1e-6, use_quack=False),
        layout=layout,
        num_registers=num_registers,
    )


@pytest.mark.parametrize("layout,num_registers", [("pure", 0), ("register_row", 1)])
def test_forward_shape(device, layout: str, num_registers: int) -> None:
    """Forward returns logits of the expected shape regardless of layout."""
    net = _build_net(layout=layout, num_registers=num_registers).to(device)
    x = torch.randn(2, 32, 32, 3, device=device)
    out = net({"input": x})
    assert out["logits"].shape == (2, 10)


def test_pure_and_register_row_param_counts_close(device) -> None:
    """The two variants should differ only by reg-related params (small delta)."""
    net_pure = _build_net(layout="pure", num_registers=0).to(device)
    net_reg = _build_net(layout="register_row", num_registers=1).to(device)
    n_pure = sum(p.numel() for p in net_pure.parameters())
    n_reg = sum(p.numel() for p in net_reg.parameters())
    # The reg variant adds: reg_token (D0) + per-stage-transition reg_proj
    # weights ([in_dim * out_dim] for 3 transitions). Should be a small delta.
    assert n_reg > n_pure
    assert (n_reg - n_pure) < n_pure * 0.1  # < 10% extra


def test_flop_count_is_positive(device) -> None:
    net = _build_net(layout="pure").to(device)
    assert net.flop_count() > 0


def test_invalid_stage_lengths_raise() -> None:
    """Mismatched per-stage list lengths raise ValueError at construction."""
    with pytest.raises(ValueError, match="stage_depths"):
        _ = ViT5HierarchicalClassificationNet(
            in_channels=3,
            num_classes=10,
            image_size=32,
            initial_patch_size=4,
            stage_dims=[16, 32, 64, 128],
            stage_depths=[1, 1, 1],  # wrong length
            stage_block_cfgs=[_make_block_cfg(d) for d in [16, 32, 64, 128]],
            patch_merge_cfgs=[
                _make_pm_cfg(16, 32, 8, 0, False),
                _make_pm_cfg(32, 64, 4, 0, False),
                _make_pm_cfg(64, 128, 2, 0, False),
            ],
            norm_cfg=LazyConfig(RMSNorm)(dim=128, eps=1e-6, use_quack=False),
            layout="pure",
        )


def test_register_row_requires_positive_num_registers() -> None:
    with pytest.raises(ValueError, match="num_registers > 0"):
        _build_net(layout="register_row", num_registers=0)


def test_backward_runs(device) -> None:
    """A backward pass on a small batch updates params without error."""
    net = _build_net(layout="register_row", num_registers=1).to(device)
    x = torch.randn(2, 32, 32, 3, device=device)
    out = net({"input": x})
    loss = out["logits"].sum()
    loss.backward()
    # Spot-check: patch_embed and reg_proj have non-zero grads.
    assert net.patch_embed.weight.grad is not None
    assert net.patch_merges[0].reg_proj.weight.grad is not None
