"""Tests for the PatchMerging module."""

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.patch_merging import PatchMerging
from nvsubquadratic.modules.rms_norm import RMSNorm


@pytest.mark.parametrize("grid_h,grid_w", [(8, 8), (16, 16), (28, 28)])
def test_pure_spatial_output_shape(device, grid_h: int, grid_w: int) -> None:
    """Pure-spatial layout: [B, H*W, C] -> [B, (H/2)*(W/2), out_dim]."""
    B, in_dim, out_dim = 2, 96, 192
    pm = PatchMerging(
        in_dim=in_dim,
        out_dim=out_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6, use_quack=False),
        has_register_row=False,
    ).to(device)

    x = torch.randn(B, grid_h * grid_w, in_dim, device=device)
    y = pm(x)

    expected_T = (grid_h // 2) * (grid_w // 2)
    assert y.shape == (B, expected_T, out_dim)


@pytest.mark.parametrize("grid_h,grid_w,num_regs", [(8, 8, 4), (28, 28, 4), (14, 14, 3)])
def test_register_row_output_shape(device, grid_h: int, grid_w: int, num_regs: int) -> None:
    """Register-row layout: [B, W + H*W, C] -> [B, W/2 + (H/2)*(W/2), out_dim]."""
    B, in_dim, out_dim = 2, 96, 192
    pm = PatchMerging(
        in_dim=in_dim,
        out_dim=out_dim,
        grid_h=grid_h,
        grid_w=grid_w,
        norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6, use_quack=False),
        num_registers=num_regs,
        has_register_row=True,
    ).to(device)

    T_in = grid_w + grid_h * grid_w
    x = torch.randn(B, T_in, in_dim, device=device)
    y = pm(x)

    expected_T = (grid_w // 2) + (grid_h // 2) * (grid_w // 2)
    assert y.shape == (B, expected_T, out_dim)


def test_register_row_pad_is_zero(device) -> None:
    """Output register-row padding slots must remain zero after the projection."""
    B, in_dim, out_dim, grid, num_regs = 2, 32, 64, 28, 4
    pm = PatchMerging(
        in_dim=in_dim,
        out_dim=out_dim,
        grid_h=grid,
        grid_w=grid,
        norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6, use_quack=False),
        num_registers=num_regs,
        has_register_row=True,
    ).to(device)

    x = torch.randn(B, grid + grid * grid, in_dim, device=device)
    # The input pad slice (positions [num_regs:grid] of the register row) does
    # not have to be zero for this test — we only check that the *output* pad
    # slice is, since it is sourced from a non-persistent zero buffer.
    y = pm(x)

    out_grid_w = grid // 2
    pad_slice = y[:, num_regs:out_grid_w, :]
    assert torch.equal(pad_slice, torch.zeros_like(pad_slice))


def test_grid_must_be_even() -> None:
    """Odd grid dims raise during construction."""
    with pytest.raises(ValueError, match="must both be even"):
        PatchMerging(
            in_dim=8,
            out_dim=16,
            grid_h=7,
            grid_w=8,
            norm_cfg=LazyConfig(RMSNorm)(dim=32, eps=1e-6, use_quack=False),
        )


def test_num_registers_must_fit_halved_grid() -> None:
    """num_registers exceeding the halved grid_w raises."""
    with pytest.raises(ValueError, match="must fit in halved grid_w"):
        PatchMerging(
            in_dim=8,
            out_dim=16,
            grid_h=4,
            grid_w=4,  # halved -> 2
            norm_cfg=LazyConfig(RMSNorm)(dim=32, eps=1e-6, use_quack=False),
            num_registers=3,
            has_register_row=True,
        )


def test_register_tokens_route_through_reg_proj(device) -> None:
    """Zeroing the patch-reduction weight should leave register tokens non-zero.

    Sanity check that the register and patch paths are wired independently.
    """
    B, in_dim, out_dim, grid, num_regs = 2, 16, 32, 8, 4
    pm = PatchMerging(
        in_dim=in_dim,
        out_dim=out_dim,
        grid_h=grid,
        grid_w=grid,
        norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6, use_quack=False),
        num_registers=num_regs,
        has_register_row=True,
    ).to(device)

    with torch.no_grad():
        pm.reduction.weight.zero_()

    x = torch.randn(B, grid + grid * grid, in_dim, device=device)
    y = pm(x)

    out_grid_w = grid // 2
    reg_out = y[:, :num_regs, :]
    patch_out = y[:, out_grid_w:, :]

    assert reg_out.abs().sum() > 0  # reg path is alive
    assert torch.equal(patch_out, torch.zeros_like(patch_out))  # patch path is zeroed


def test_flop_count_pure(device) -> None:
    """FLOP count returns a positive int dominated by the reduction linear."""
    in_dim, out_dim, grid = 96, 192, 28
    pm = PatchMerging(
        in_dim=in_dim,
        out_dim=out_dim,
        grid_h=grid,
        grid_w=grid,
        norm_cfg=LazyConfig(RMSNorm)(dim=4 * in_dim, eps=1e-6, use_quack=False),
    ).to(device)
    f = pm.flop_count()
    T_out = (grid // 2) ** 2
    reduction_flops = 2 * T_out * 4 * in_dim * out_dim
    assert f > reduction_flops  # norm contributes a (small) positive amount
    assert isinstance(f, int)
