"""Tests for the unified UNet with pluggable blocks.

Covers:
    - Individual block forward shapes and residual behaviour (ConvNeXt, Attention, Hyena).
    - UNet forward shapes for 2D and 3D inputs with each block type.
    - WellUNet channels-last dict interface.
    - LazyConfig-based block instantiation inside UNet.
    - Gradient flow through all block types.
    - Gradient checkpointing produces the same output.
    - Encoder/decoder dimension progression.
    - Skip-connection concatenation dimensions.

Usage:
    PYTHONPATH=. pytest tests/networks/baselines/test_unet.py -v
"""

from __future__ import annotations

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.networks.baselines.unet import (
    AttentionBlock,
    ConvNeXtBlock,
    HyenaBlock,
    UNet,
    WellUNet,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

SMALL_2D = dict(
    dim_in=4,
    dim_out=2,
    n_spatial_dims=2,
    spatial_resolution=(32, 32),
    stages=2,
    blocks_per_stage=1,
    blocks_at_neck=1,
    init_features=8,
)

SMALL_3D = dict(
    dim_in=4,
    dim_out=2,
    n_spatial_dims=3,
    spatial_resolution=(16, 16, 16),
    stages=2,
    blocks_per_stage=1,
    blocks_at_neck=1,
    init_features=8,
)


def _convnext_cfg(n_spatial_dims=2):
    return LazyConfig(ConvNeXtBlock)(n_spatial_dims=n_spatial_dims)


def _attention_cfg(n_spatial_dims=2):
    return LazyConfig(AttentionBlock)(n_spatial_dims=n_spatial_dims, num_heads=2, mlp_ratio=2)


def _hyena_cfg(n_spatial_dims=2):
    return LazyConfig(HyenaBlock)(
        n_spatial_dims=n_spatial_dims,
        mlp_ratio=2,
        omega_0=10.0,
        siren_layers=2,
        siren_hidden_dim=16,
    )


# ─── Block-level tests ──────────────────────────────────────────────────────


class TestConvNeXtBlock:
    """Tests for ConvNeXtBlock."""

    def test_output_shape_2d(self):
        """Output shape must match input shape (residual block)."""
        block = ConvNeXtBlock(dim=16, n_spatial_dims=2)
        x = torch.randn(2, 16, 8, 8)
        y = block(x)
        assert y.shape == x.shape

    def test_output_shape_3d(self):
        """Block works with 3D spatial inputs."""
        block = ConvNeXtBlock(dim=16, n_spatial_dims=3)
        x = torch.randn(1, 16, 8, 8, 8)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_at_init(self):
        """At initialization, the block should approximately preserve input (residual + small perturbation)."""
        block = ConvNeXtBlock(dim=8, n_spatial_dims=2)
        block.eval()
        x = torch.randn(1, 8, 4, 4)
        with torch.no_grad():
            y = block(x)
        # The residual connection means output ≈ input + small_perturbation
        assert (y - x).abs().mean() < (x.abs().mean() * 10), "Residual should keep output close to input at init"

    def test_spatial_res_ignored(self):
        """ConvNeXtBlock accepts spatial_res but ignores it."""
        block = ConvNeXtBlock(dim=8, n_spatial_dims=2, spatial_res=42)
        x = torch.randn(1, 8, 4, 4)
        y = block(x)
        assert y.shape == x.shape

    def test_no_layer_scale(self):
        """Block works with layer_scale disabled."""
        block = ConvNeXtBlock(dim=8, n_spatial_dims=2, layer_scale_init_value=0)
        assert block.gamma is None
        x = torch.randn(1, 8, 4, 4)
        y = block(x)
        assert y.shape == x.shape


class TestAttentionBlock:
    """Tests for AttentionBlock."""

    def test_output_shape_2d(self):
        """Output shape must match input shape."""
        block = AttentionBlock(dim=12, n_spatial_dims=2, num_heads=3)
        x = torch.randn(2, 12, 8, 8)
        y = block(x)
        assert y.shape == x.shape

    def test_output_shape_3d(self):
        """Block works with 3D spatial inputs."""
        block = AttentionBlock(dim=12, n_spatial_dims=3, num_heads=3)
        x = torch.randn(1, 12, 4, 4, 4)
        y = block(x)
        assert y.shape == x.shape

    def test_residual_at_init(self):
        """At initialization, the block should approximately preserve input."""
        block = AttentionBlock(dim=12, n_spatial_dims=2, num_heads=3)
        block.eval()
        x = torch.randn(1, 12, 4, 4)
        with torch.no_grad():
            y = block(x)
        assert (y - x).abs().mean() < (x.abs().mean() * 10)


class TestHyenaBlock:
    """Tests for HyenaBlock."""

    def test_output_shape_2d(self):
        """Output shape must match input shape."""
        block = HyenaBlock(dim=16, n_spatial_dims=2, spatial_res=8, siren_layers=2, siren_hidden_dim=16)
        x = torch.randn(2, 16, 8, 8)
        y = block(x)
        assert y.shape == x.shape

    def test_output_shape_3d(self):
        """Block works with 3D spatial inputs."""
        block = HyenaBlock(dim=16, n_spatial_dims=3, spatial_res=4, siren_layers=2, siren_hidden_dim=16)
        x = torch.randn(1, 16, 4, 4, 4)
        y = block(x)
        assert y.shape == x.shape

    def test_requires_spatial_res(self):
        """HyenaBlock must raise when spatial_res is None."""
        with pytest.raises(AssertionError, match="spatial_res"):
            HyenaBlock(dim=16, n_spatial_dims=2, spatial_res=None)

    def test_residual_at_init(self):
        """At initialization, the block should approximately preserve input."""
        block = HyenaBlock(dim=16, n_spatial_dims=2, spatial_res=4, siren_layers=2, siren_hidden_dim=16)
        block.eval()
        x = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            y = block(x)
        assert (y - x).abs().mean() < (x.abs().mean() * 10)


# ─── UNet-level tests ───────────────────────────────────────────────────────


class TestUNet:
    """Tests for the unified UNet class."""

    @pytest.mark.parametrize("block_cfg_fn", [_convnext_cfg, _attention_cfg, _hyena_cfg], ids=["convnext", "attention", "hyena"])
    def test_forward_shape_2d(self, block_cfg_fn):
        """UNet output shape must match [B, dim_out, *spatial] for all block types."""
        model = UNet(**SMALL_2D, block_cfg=block_cfg_fn(n_spatial_dims=2))
        model.eval()
        x = torch.randn(1, SMALL_2D["dim_in"], *SMALL_2D["spatial_resolution"])
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, SMALL_2D["dim_out"], *SMALL_2D["spatial_resolution"])

    @pytest.mark.parametrize("block_cfg_fn", [_convnext_cfg, _attention_cfg, _hyena_cfg], ids=["convnext", "attention", "hyena"])
    def test_forward_shape_3d(self, block_cfg_fn):
        """UNet works with 3D inputs for all block types."""
        model = UNet(**SMALL_3D, block_cfg=block_cfg_fn(n_spatial_dims=3))
        model.eval()
        x = torch.randn(1, SMALL_3D["dim_in"], *SMALL_3D["spatial_resolution"])
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, SMALL_3D["dim_out"], *SMALL_3D["spatial_resolution"])

    @pytest.mark.parametrize("block_cfg_fn", [_convnext_cfg, _attention_cfg, _hyena_cfg], ids=["convnext", "attention", "hyena"])
    def test_gradient_flow(self, block_cfg_fn):
        """Gradients must flow through to all parameters."""
        model = UNet(**SMALL_2D, block_cfg=block_cfg_fn(n_spatial_dims=2))
        model.train()
        x = torch.randn(1, SMALL_2D["dim_in"], *SMALL_2D["spatial_resolution"])
        y = model(x)
        y.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_checkpointing_output(self):
        """Gradient checkpointing must produce the same forward output."""
        torch.manual_seed(0)
        model_no_ckpt = UNet(**SMALL_2D, block_cfg=_convnext_cfg(2), gradient_checkpointing=False)
        torch.manual_seed(0)
        model_ckpt = UNet(**SMALL_2D, block_cfg=_convnext_cfg(2), gradient_checkpointing=True)
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())

        model_no_ckpt.eval()
        model_ckpt.eval()
        x = torch.randn(1, SMALL_2D["dim_in"], *SMALL_2D["spatial_resolution"])
        with torch.no_grad():
            y1 = model_no_ckpt(x)
            y2 = model_ckpt(x)
        assert torch.equal(y1, y2), f"Checkpointing changes output! max diff = {(y1 - y2).abs().max():.2e}"

    def test_gradient_checkpointing_grads(self):
        """Gradient checkpointing must produce the same gradients."""
        torch.manual_seed(0)
        model_no_ckpt = UNet(**SMALL_2D, block_cfg=_convnext_cfg(2), gradient_checkpointing=False)
        torch.manual_seed(0)
        model_ckpt = UNet(**SMALL_2D, block_cfg=_convnext_cfg(2), gradient_checkpointing=True)
        model_ckpt.load_state_dict(model_no_ckpt.state_dict())

        model_no_ckpt.train()
        model_ckpt.train()
        x = torch.randn(1, SMALL_2D["dim_in"], *SMALL_2D["spatial_resolution"])

        y1 = model_no_ckpt(x.clone())
        y1.sum().backward()
        y2 = model_ckpt(x.clone())
        y2.sum().backward()

        for (n1, p1), (n2, p2) in zip(model_no_ckpt.named_parameters(), model_ckpt.named_parameters()):
            assert torch.allclose(p1.grad, p2.grad, atol=1e-6), (
                f"Grad mismatch for {n1}! max diff = {(p1.grad - p2.grad).abs().max():.2e}"
            )

    def test_encoder_decoder_structure(self):
        """Encoder and decoder must have the correct number of stages."""
        model = UNet(**SMALL_2D, block_cfg=_convnext_cfg(2))
        assert len(model.encoder) == SMALL_2D["stages"]
        assert len(model.decoder) == SMALL_2D["stages"]

    def test_deeper_config(self):
        """UNet works with more stages and blocks per stage."""
        model = UNet(
            dim_in=4,
            dim_out=2,
            n_spatial_dims=2,
            spatial_resolution=(64, 64),
            stages=3,
            blocks_per_stage=2,
            blocks_at_neck=2,
            init_features=8,
            block_cfg=_convnext_cfg(2),
        )
        model.eval()
        x = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 2, 64, 64)

    def test_batch_size_independence(self):
        """Output is independent per sample in the batch."""
        model = UNet(**SMALL_2D, block_cfg=_convnext_cfg(2))
        model.eval()
        x1 = torch.randn(1, SMALL_2D["dim_in"], *SMALL_2D["spatial_resolution"])
        x2 = torch.randn(1, SMALL_2D["dim_in"], *SMALL_2D["spatial_resolution"])
        with torch.no_grad():
            y_single_1 = model(x1)
            y_single_2 = model(x2)
            y_batch = model(torch.cat([x1, x2], dim=0))
        assert torch.allclose(y_batch[0], y_single_1[0], atol=1e-6)
        assert torch.allclose(y_batch[1], y_single_2[0], atol=1e-6)


# ─── WellUNet wrapper tests ─────────────────────────────────────────────────


class TestWellUNet:
    """Tests for the WellUNet channels-last dict wrapper."""

    def test_dict_interface_2d(self):
        """WellUNet accepts dict input and returns dict output with channels-last layout."""
        model = WellUNet(**SMALL_2D, block_cfg=_convnext_cfg(2))
        model.eval()
        H, W = SMALL_2D["spatial_resolution"]
        inp = {"input": torch.randn(1, H, W, SMALL_2D["dim_in"]), "condition": None}
        with torch.no_grad():
            out = model(inp)
        assert "logits" in out
        assert out["logits"].shape == (1, H, W, SMALL_2D["dim_out"])

    def test_dict_interface_3d(self):
        """WellUNet works with 3D channels-last input."""
        model = WellUNet(**SMALL_3D, block_cfg=_convnext_cfg(3))
        model.eval()
        D, H, W = SMALL_3D["spatial_resolution"]
        inp = {"input": torch.randn(1, D, H, W, SMALL_3D["dim_in"]), "condition": None}
        with torch.no_grad():
            out = model(inp)
        assert out["logits"].shape == (1, D, H, W, SMALL_3D["dim_out"])

    def test_lazy_config_instantiation(self):
        """WellUNet can be instantiated via LazyConfig + instantiate()."""
        cfg = LazyConfig(WellUNet)(
            **SMALL_2D,
            block_cfg=LazyConfig(ConvNeXtBlock)(n_spatial_dims=2),
        )
        model = instantiate(cfg)
        assert isinstance(model, WellUNet)
        H, W = SMALL_2D["spatial_resolution"]
        inp = {"input": torch.randn(1, H, W, SMALL_2D["dim_in"]), "condition": None}
        with torch.no_grad():
            out = model(inp)
        assert out["logits"].shape == (1, H, W, SMALL_2D["dim_out"])

    @pytest.mark.parametrize("block_cfg_fn", [_convnext_cfg, _attention_cfg, _hyena_cfg], ids=["convnext", "attention", "hyena"])
    def test_all_block_types(self, block_cfg_fn):
        """WellUNet works with all block types."""
        model = WellUNet(**SMALL_2D, block_cfg=block_cfg_fn(n_spatial_dims=2))
        model.eval()
        H, W = SMALL_2D["spatial_resolution"]
        inp = {"input": torch.randn(1, H, W, SMALL_2D["dim_in"]), "condition": None}
        with torch.no_grad():
            out = model(inp)
        assert out["logits"].shape == (1, H, W, SMALL_2D["dim_out"])
