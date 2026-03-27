"""Sanity check: verify that the unified UNet with ConvNeXtBlock produces the
same output as the original standalone UNetConvNext implementation.

Usage:
    python tests/test_unet_convnext_parity.py
    # or via pytest:
    pytest tests/test_unet_convnext_parity.py -v
"""

from __future__ import annotations

import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet import ConvNeXtBlock, UNet
from nvsubquadratic.networks.baselines.unet_convnext import UNetConvNext


def _build_pair(
    dim_in=4,
    dim_out=2,
    n_spatial_dims=2,
    spatial_resolution=(32, 32),
    stages=2,
    blocks_per_stage=1,
    blocks_at_neck=1,
    init_features=8,
):
    """Build the original and unified UNets with the same hyperparameters."""
    original = UNetConvNext(
        dim_in=dim_in,
        dim_out=dim_out,
        n_spatial_dims=n_spatial_dims,
        spatial_resolution=spatial_resolution,
        stages=stages,
        blocks_per_stage=blocks_per_stage,
        blocks_at_neck=blocks_at_neck,
        init_features=init_features,
    )

    block_cfg = LazyConfig(ConvNeXtBlock)(n_spatial_dims=n_spatial_dims)
    unified = UNet(
        dim_in=dim_in,
        dim_out=dim_out,
        n_spatial_dims=n_spatial_dims,
        spatial_resolution=spatial_resolution,
        stages=stages,
        blocks_per_stage=blocks_per_stage,
        blocks_at_neck=blocks_at_neck,
        init_features=init_features,
        block_cfg=block_cfg,
    )

    return original, unified


def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module):
    """Copy all parameters from src to dst (must have identical state_dict keys)."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()

    missing = set(dst_sd.keys()) - set(src_sd.keys())
    unexpected = set(src_sd.keys()) - set(dst_sd.keys())
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch.\n"
            f"  Missing in source:     {missing or 'none'}\n"
            f"  Unexpected in source:  {unexpected or 'none'}"
        )

    dst.load_state_dict(src_sd)


def test_state_dict_keys_match():
    """The two models must have identical state_dict key sets."""
    original, unified = _build_pair()
    orig_keys = set(original.state_dict().keys())
    unif_keys = set(unified.state_dict().keys())
    assert orig_keys == unif_keys, (
        f"Key mismatch.\n  Only in original: {orig_keys - unif_keys}\n  Only in unified:  {unif_keys - orig_keys}"
    )


def test_forward_parity():
    """With identical weights, both models must produce bit-identical outputs."""
    torch.manual_seed(42)
    original, unified = _build_pair()
    _copy_weights(original, unified)

    original.eval()
    unified.eval()

    x = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        y_orig = original(x)
        y_unif = unified(x)

    assert torch.equal(y_orig, y_unif), f"Outputs differ! max abs diff = {(y_orig - y_unif).abs().max().item():.2e}"


def test_forward_parity_deeper():
    """Same test with more stages and blocks to exercise all code paths."""
    torch.manual_seed(123)
    original, unified = _build_pair(
        dim_in=6,
        dim_out=3,
        stages=3,
        blocks_per_stage=2,
        blocks_at_neck=2,
        init_features=16,
        spatial_resolution=(64, 64),
    )
    _copy_weights(original, unified)

    original.eval()
    unified.eval()

    x = torch.randn(2, 6, 64, 64)
    with torch.no_grad():
        y_orig = original(x)
        y_unif = unified(x)

    assert torch.equal(y_orig, y_unif), f"Outputs differ! max abs diff = {(y_orig - y_unif).abs().max().item():.2e}"


def test_backward_parity():
    """With identical weights and input, gradients must be bit-identical."""
    torch.manual_seed(42)
    original, unified = _build_pair()
    _copy_weights(original, unified)

    original.train()
    unified.train()

    x = torch.randn(1, 4, 32, 32)
    # Use the same input for both but with independent graph
    x_orig = x.clone().detach().requires_grad_(True)
    x_unif = x.clone().detach().requires_grad_(True)

    y_orig = original(x_orig)
    y_unif = unified(x_unif)

    # Scalar loss for backward
    loss_orig = y_orig.sum()
    loss_unif = y_unif.sum()

    loss_orig.backward()
    loss_unif.backward()

    # Check input gradients
    assert torch.equal(x_orig.grad, x_unif.grad), (
        f"Input gradients differ! max abs diff = {(x_orig.grad - x_unif.grad).abs().max().item():.2e}"
    )

    # Check parameter gradients
    for (name_o, p_o), (name_u, p_u) in zip(original.named_parameters(), unified.named_parameters()):
        assert name_o == name_u, f"Parameter name mismatch: {name_o} vs {name_u}"
        assert p_o.grad is not None and p_u.grad is not None, f"Missing grad for {name_o}"
        assert torch.equal(p_o.grad, p_u.grad), (
            f"Gradient mismatch for {name_o}! max abs diff = {(p_o.grad - p_u.grad).abs().max().item():.2e}"
        )


def test_backward_parity_deeper():
    """Backward parity with more stages and blocks."""
    torch.manual_seed(123)
    original, unified = _build_pair(
        dim_in=6,
        dim_out=3,
        stages=3,
        blocks_per_stage=2,
        blocks_at_neck=2,
        init_features=16,
        spatial_resolution=(64, 64),
    )
    _copy_weights(original, unified)

    original.train()
    unified.train()

    x = torch.randn(2, 6, 64, 64)
    x_orig = x.clone().detach().requires_grad_(True)
    x_unif = x.clone().detach().requires_grad_(True)

    y_orig = original(x_orig)
    y_unif = unified(x_unif)

    loss_orig = y_orig.sum()
    loss_unif = y_unif.sum()

    loss_orig.backward()
    loss_unif.backward()

    assert torch.equal(x_orig.grad, x_unif.grad), (
        f"Input gradients differ! max abs diff = {(x_orig.grad - x_unif.grad).abs().max().item():.2e}"
    )

    for (name_o, p_o), (name_u, p_u) in zip(original.named_parameters(), unified.named_parameters()):
        assert torch.equal(p_o.grad, p_u.grad), (
            f"Gradient mismatch for {name_o}! max abs diff = {(p_o.grad - p_u.grad).abs().max().item():.2e}"
        )


def test_param_count_match():
    """Both models must have exactly the same number of parameters."""
    original, unified = _build_pair()
    n_orig = sum(p.numel() for p in original.parameters())
    n_unif = sum(p.numel() for p in unified.parameters())
    assert n_orig == n_unif, f"Param count mismatch: original={n_orig}, unified={n_unif}"


if __name__ == "__main__":
    for test_fn in [
        test_state_dict_keys_match,
        test_forward_parity,
        test_forward_parity_deeper,
        test_backward_parity,
        test_backward_parity_deeper,
        test_param_count_match,
    ]:
        print(f"Running {test_fn.__name__}...", end=" ")
        test_fn()
        print("PASSED")
    print("\nAll parity tests passed!")
