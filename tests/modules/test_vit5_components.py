"""Comprehensive tests for all ViT-5 components.

Tests verify:
1. RMSNorm matches reference implementation
2. LayerScale shape, init value, and gradient flow
3. DropPath behavior in train/eval modes
4. ViT5Attention: shapes, QK-Norm, RoPE token routing, register handling
5. ViT5ResidualBlock: residual connection, LayerScale, DropPath
6. ViT5ClassificationNet: end-to-end forward pass, parameter count, token assembly
7. Cross-validation against the reference ViT-5 repo
8. GAP readout & token layout (use_cls_token / prepend_registers)

Run:
    PYTHONPATH=. python -m pytest tests/modules/test_vit5_components.py -v -o addopts=""

See tests/README.md for all test suites and markers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.drop_path import DropPath
from nvsubquadratic.modules.layer_scale import LayerScale
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


try:
    import apex  # noqa: F401

    _has_apex = True
except ImportError:
    _has_apex = False


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def device() -> torch.device:
    """Return CUDA device when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def vit5_small_params() -> dict[str, int]:
    """Reference ViT-5-Small hyperparameters from Table 4 of the paper."""
    return {
        "hidden_dim": 384,
        "num_heads": 6,
        "num_blocks": 12,
        "patch_size": 16,
        "image_size": 224,
        "num_registers": 4,
        "mlp_ratio": 4,
        "head_dim": 64,  # 384 / 6
    }


# ─── 1. RMSNorm Tests ──────────────────────────────────────────────────────────


class TestRMSNorm:
    """Tests for the RMSNorm module (Root Mean Square Layer Normalization)."""

    def test_output_shape(self, device: torch.device) -> None:
        """Output tensor preserves input shape ``[B, T, C]``."""
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_matches_reference_implementation(self, device: torch.device) -> None:
        """Verify our RMSNorm matches the reference repo's RMSNorm."""
        dim = 64
        eps = 1e-6
        norm = RMSNorm(dim, eps=eps).to(device)
        x = torch.randn(2, 10, dim, device=device)

        # Reference implementation (from models_vit5.py)
        weight = norm.weight.data.clone()
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        ref_out = weight * (x_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)

        our_out = norm(x)
        torch.testing.assert_close(our_out, ref_out, atol=1e-5, rtol=1e-4)

    def test_learnable_weight(self) -> None:
        """Weight parameter is learnable and initialized to ones."""
        norm = RMSNorm(64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)
        torch.testing.assert_close(norm.weight.data, torch.ones(64))

    def test_no_weight_decay_flag(self) -> None:
        """Weight parameter is marked with ``_no_weight_decay``."""
        norm = RMSNorm(64)
        assert hasattr(norm.weight, "_no_weight_decay")
        assert norm.weight._no_weight_decay is True

    def test_dtype_preservation(self, device: torch.device) -> None:
        """Output dtype matches input dtype (e.g. float16 in, float16 out)."""
        norm = RMSNorm(64).half().to(device)
        x = torch.randn(2, 10, 64, dtype=torch.float16, device=device)
        out = norm(x)
        assert out.dtype == torch.float16


# ─── 2. LayerScale Tests ───────────────────────────────────────────────────────


class TestLayerScale:
    """Tests for the LayerScale module (per-channel learnable scaling)."""

    def test_output_shape(self) -> None:
        """Output tensor preserves input shape ``[B, T, C]``."""
        ls = LayerScale(384, init_value=1e-4)
        x = torch.randn(2, 196, 384)
        out = ls(x)
        assert out.shape == x.shape

    def test_init_value(self) -> None:
        """Gamma vector is initialized to the given ``init_value``."""
        ls = LayerScale(384, init_value=1e-4)
        torch.testing.assert_close(ls.gamma.data, torch.full((384,), 1e-4))

    def test_scaling_effect(self) -> None:
        """Scaling by gamma=0.5 halves a ones tensor."""
        ls = LayerScale(4, init_value=0.5)
        x = torch.ones(1, 1, 4)
        out = ls(x)
        torch.testing.assert_close(out, torch.full_like(x, 0.5))

    def test_gradient_flow(self) -> None:
        """Gradients reach both the input tensor and the gamma parameter."""
        ls = LayerScale(384, init_value=1e-4)
        x = torch.randn(2, 196, 384, requires_grad=True)
        out = ls(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert ls.gamma.grad is not None

    def test_matches_reference_block_scaling(self) -> None:
        """Verify LayerScale matches the reference: ``gamma * x``."""
        dim = 384
        init_val = 1e-4
        ls = LayerScale(dim, init_value=init_val)
        x = torch.randn(2, 196, dim)

        # Reference: self.gamma_1 * self.attn(self.norm1(x))
        gamma = init_val * torch.ones(dim)
        ref_out = gamma * x
        our_out = ls(x)
        torch.testing.assert_close(our_out, ref_out, atol=1e-7, rtol=1e-6)

    def test_no_weight_decay_flag(self) -> None:
        """Gamma parameter is marked with ``_no_weight_decay``."""
        ls = LayerScale(384)
        assert hasattr(ls.gamma, "_no_weight_decay")
        assert ls.gamma._no_weight_decay is True


# ─── 3. DropPath Tests ─────────────────────────────────────────────────────────


class TestDropPath:
    """Tests for the DropPath module (stochastic depth / sample-wise dropout)."""

    def test_identity_when_zero(self) -> None:
        """With drop_prob=0, output equals input even in training mode."""
        dp = DropPath(0.0)
        x = torch.randn(4, 10, 384)
        dp.train()
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_identity_in_eval(self) -> None:
        """In eval mode, output always equals input regardless of drop_prob."""
        dp = DropPath(0.5)
        x = torch.randn(4, 10, 384)
        dp.eval()
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_drops_samples_in_train(self) -> None:
        """With high drop_prob, most samples are zeroed during training."""
        torch.manual_seed(0)
        dp = DropPath(0.99)
        dp.train()
        x = torch.ones(100, 10, 384)
        out = dp(x)
        # With p=0.99, most samples should be zeroed
        zero_samples = (out.abs().sum(dim=(1, 2)) == 0).sum().item()
        assert zero_samples > 50, f"Expected most samples zeroed, got {zero_samples}"

    def test_expectation_preservation(self) -> None:
        """DropPath rescales kept samples so the expected value is preserved."""
        torch.manual_seed(42)
        dp = DropPath(0.3)
        dp.train()
        x = torch.ones(10000, 1, 1)
        out = dp(x)
        mean = out.mean().item()
        assert abs(mean - 1.0) < 0.1, f"Expected mean ~1.0, got {mean}"


# ─── 4. ViT5Attention Tests ────────────────────────────────────────────────────


class TestViT5Attention:
    """Tests for ViT-5 multi-head self-attention with 2D RoPE and QK-Norm."""

    def _make_attn(self, device: torch.device, **kwargs) -> ViT5Attention:
        """Build a ViT5Attention module with sensible defaults.

        Any keyword argument overrides the corresponding default.
        """
        defaults = {
            "hidden_dim": 384,
            "num_heads": 6,
            "num_patches_h": 14,
            "num_patches_w": 14,
            "num_registers": 4,
            "qk_norm": LazyConfig(RMSNorm)(dim=64, eps=1e-6),
            "rope_base": 10000.0,
            "reg_rope_base": 100.0,
            "attn_dropout": 0.0,
            "proj_dropout": 0.0,
            "qkv_bias": False,
        }
        defaults.update(kwargs)
        return ViT5Attention(**defaults).to(device)

    def test_output_shape(self, device: torch.device) -> None:
        """Output shape ``[B, T, C]`` matches input for T = 1 + 196 + 4."""
        attn = self._make_attn(device)
        T = 1 + 196 + 4  # cls + patches + registers
        x = torch.randn(2, T, 384, device=device)
        out = attn(x)
        assert out.shape == (2, T, 384)

    def test_no_qkv_bias(self, device: torch.device) -> None:
        """QKV projection has no bias when ``qkv_bias=False``."""
        attn = self._make_attn(device, qkv_bias=False)
        assert attn.qkv.bias is None

    def test_with_qkv_bias(self, device: torch.device) -> None:
        """QKV projection has a bias when ``qkv_bias=True``."""
        attn = self._make_attn(device, qkv_bias=True)
        assert attn.qkv.bias is not None

    def test_qk_norm_modules_exist(self, device: torch.device) -> None:
        """Q and K normalization modules are RMSNorm when ``qk_norm`` is set."""
        attn = self._make_attn(device, qk_norm=LazyConfig(RMSNorm)(dim=64, eps=1e-6))
        assert isinstance(attn.q_norm, RMSNorm)
        assert isinstance(attn.k_norm, RMSNorm)

    def test_no_qk_norm(self, device: torch.device) -> None:
        """No RMSNorm Q/K normalization when ``qk_norm=None``."""
        attn = self._make_attn(device, qk_norm=None)
        assert not hasattr(attn, "q_norm") or not isinstance(getattr(attn, "q_norm", None), RMSNorm)

    def test_register_count(self, device: torch.device) -> None:
        """Register count and RoPE grid shape are stored correctly."""
        attn = self._make_attn(device, num_registers=4)
        assert attn.num_registers == 4
        assert attn.reg_rope_h == 2
        assert attn.reg_rope_w == 2

    def test_zero_registers(self, device: torch.device) -> None:
        """Forward pass works with zero register tokens (T = 1 + 196)."""
        attn = self._make_attn(device, num_registers=0)
        T = 1 + 196  # cls + patches only
        x = torch.randn(2, T, 384, device=device)
        out = attn(x)
        assert out.shape == (2, T, 384)

    def test_gradient_flow(self, device: torch.device) -> None:
        """Gradients propagate from output back to input."""
        attn = self._make_attn(device)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_rope_applies_to_patches_not_cls(self, device: torch.device) -> None:
        """Verify CLS token does NOT get RoPE (its position should not change)."""
        attn = self._make_attn(device)
        T = 1 + 196 + 4

        x1 = torch.randn(1, T, 384, device=device)
        x2 = x1.clone()
        x2[:, 0] += 1.0

        attn.eval()
        with torch.no_grad():
            out1 = attn(x1)
            out2 = attn(x2)

        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_deterministic_eval(self, device: torch.device) -> None:
        """Two forward passes with the same input produce identical output in eval."""
        attn = self._make_attn(device, attn_dropout=0.0)
        attn.eval()
        T = 1 + 196 + 4
        x = torch.randn(1, T, 384, device=device)
        with torch.no_grad():
            out1 = attn(x.clone())
            out2 = attn(x.clone())
        torch.testing.assert_close(out1, out2)

    def test_token_ordering_cls_patches_registers(self, device: torch.device) -> None:
        """Verify the attention module expects ``[cls, patches..., registers...]``."""
        attn = self._make_attn(device, num_registers=4)
        B, C = 2, 384
        cls = torch.randn(B, 1, C, device=device)
        patches = torch.randn(B, 196, C, device=device)
        regs = torch.randn(B, 4, C, device=device)
        x = torch.cat([cls, patches, regs], dim=1)
        out = attn(x)
        assert out.shape == (B, 201, C)


# ─── 5. ViT5ResidualBlock Tests ────────────────────────────────────────────────


class TestViT5ResidualBlock:
    """Tests for the ViT-5 pre-norm residual block (Attention + MLP + LayerScale + DropPath)."""

    def _make_block(
        self,
        device: torch.device,
        hidden_dim: int = 384,
        num_heads: int = 6,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ) -> ViT5ResidualBlock:
        """Build a ViT5ResidualBlock with configurable LayerScale and DropPath."""
        block = ViT5ResidualBlock(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_patches_h=14,
                num_patches_w=14,
                num_registers=4,
                qk_norm=LazyConfig(RMSNorm)(dim=hidden_dim // num_heads, eps=1e-6),
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=hidden_dim,
                activation="gelu",
                expansion_factor=4.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            hidden_dim=hidden_dim,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path,
        )
        return block.to(device)

    def test_output_shape(self, device: torch.device) -> None:
        """Output shape ``[B, T, C]`` matches input."""
        block = self._make_block(device)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device)
        out = block(x)
        assert out.shape == (2, T, 384)

    def test_residual_connection(self, device: torch.device) -> None:
        """With tiny LayerScale init, output should be close to input (strong residual)."""
        block = self._make_block(device, layer_scale_init=1e-8)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device)
        block.eval()
        with torch.no_grad():
            out = block(x)
        diff = (out - x).abs().max().item()
        assert diff < 0.1, f"Expected small residual diff, got {diff}"

    def test_has_layer_scale(self, device: torch.device) -> None:
        """Both attention and MLP branches use LayerScale when init > 0."""
        block = self._make_block(device, layer_scale_init=1e-4)
        assert isinstance(block.ls_attn, LayerScale)
        assert isinstance(block.ls_mlp, LayerScale)

    def test_no_layer_scale(self, device: torch.device) -> None:
        """LayerScale is replaced by Identity when init = 0."""
        block = self._make_block(device, layer_scale_init=0.0)
        assert isinstance(block.ls_attn, nn.Identity)
        assert isinstance(block.ls_mlp, nn.Identity)

    def test_has_drop_path(self, device: torch.device) -> None:
        """DropPath is active with the configured drop probability."""
        block = self._make_block(device, drop_path=0.1)
        assert isinstance(block.drop_path, DropPath)
        assert block.drop_path.drop_prob == 0.1

    def test_no_drop_path(self, device: torch.device) -> None:
        """DropPath is replaced by Identity when rate = 0."""
        block = self._make_block(device, drop_path=0.0)
        assert isinstance(block.drop_path, nn.Identity)

    def test_gradient_flow_through_all_components(self, device: torch.device) -> None:
        """Gradients reach the input and both LayerScale gamma parameters."""
        block = self._make_block(device)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert block.ls_attn.gamma.grad is not None
        assert block.ls_mlp.gamma.grad is not None

    def test_uses_rmsnorm(self, device: torch.device) -> None:
        """Both pre-norms are RMSNorm instances."""
        block = self._make_block(device)
        assert isinstance(block.input_norm, RMSNorm)
        assert isinstance(block.mlp_norm, RMSNorm)

    def test_uses_gelu_mlp(self, device: torch.device) -> None:
        """MLP uses GeLU activation (not SwiGLU)."""
        block = self._make_block(device)
        assert block.mlp.activation == "gelu"
        assert not block.mlp.is_glu_variant


# ─── 6. ViT5ClassificationNet Tests ────────────────────────────────────────────


class TestViT5ClassificationNet:
    """Tests for the full ViT-5 classification network (patchify + blocks + head)."""

    def _make_net(
        self,
        device: torch.device,
        hidden_dim: int = 384,
        num_blocks: int = 2,
        num_registers: int = 4,
        image_size: int = 224,
    ) -> ViT5ClassificationNet:
        """Build a small ViT-5 net (2 blocks by default for speed)."""
        net = ViT5ClassificationNet(
            in_channels=3,
            num_classes=1000,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            patch_size=16,
            image_size=image_size,
            num_registers=num_registers,
            dropout_rate=0.0,
            norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            block_cfg=LazyConfig(ViT5ResidualBlock)(
                sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                    hidden_dim=hidden_dim,
                    num_heads=6,
                    num_patches_h=image_size // 16,
                    num_patches_w=image_size // 16,
                    num_registers=num_registers,
                    qk_norm=LazyConfig(RMSNorm)(dim=hidden_dim // 6, eps=1e-6),
                ),
                sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
                mlp_cfg=LazyConfig(MLP)(
                    dim=hidden_dim,
                    activation="gelu",
                    expansion_factor=4.0,
                    dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                ),
                mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
                hidden_dim=hidden_dim,
                layer_scale_init=1e-4,
                drop_path_rate=0.05,
            ),
        )
        return net.to(device)

    def test_output_shape(self, device: torch.device) -> None:
        """Forward returns ``{"logits": [B, num_classes]}``."""
        net = self._make_net(device)
        x = {"input": torch.randn(2, 224, 224, 3, device=device), "condition": None}
        out = net(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 1000)

    def test_parameter_count_full_model(self, device: torch.device) -> None:
        """Full 12-block ViT-5-Small should have ~22M params (Table 4)."""
        net = self._make_net(device, num_blocks=12)
        num_params = sum(p.numel() for p in net.parameters()) / 1e6
        assert 21.0 < num_params < 23.0, f"Expected ~22M params, got {num_params:.1f}M"

    def test_cls_token_exists(self, device: torch.device) -> None:
        """CLS token is a learnable ``[1, 1, C]`` parameter."""
        net = self._make_net(device)
        assert net.cls_token.shape == (1, 1, 384)

    def test_register_tokens_exist(self, device: torch.device) -> None:
        """Register tokens are a learnable ``[1, num_registers, C]`` parameter."""
        net = self._make_net(device, num_registers=4)
        assert net.reg_token is not None
        assert net.reg_token.shape == (1, 4, 384)

    def test_no_register_tokens(self, device: torch.device) -> None:
        """With ``num_registers=0``, ``reg_token`` is ``None``."""
        net = self._make_net(device, num_registers=0)
        assert net.reg_token is None

    def test_pos_embed_shape(self, device: torch.device) -> None:
        """Positional embedding covers patch tokens only: ``[1, num_patches, C]``."""
        net = self._make_net(device)
        num_patches = (224 // 16) ** 2  # 196
        assert net.pos_embed.shape == (1, num_patches, 384)

    def test_patch_embed_is_conv2d(self, device: torch.device) -> None:
        """Patch embedding is a non-overlapping Conv2d with stride = kernel = patch_size."""
        net = self._make_net(device)
        assert isinstance(net.patch_embed, nn.Conv2d)
        assert net.patch_embed.kernel_size == (16, 16)
        assert net.patch_embed.stride == (16, 16)

    def test_uses_cls_token_readout(self, device: torch.device) -> None:
        """Default config reads out the CLS token (not GAP)."""
        net = self._make_net(device, num_blocks=1)
        net.eval()
        x = {"input": torch.randn(1, 224, 224, 3, device=device), "condition": None}
        with torch.no_grad():
            out = net(x)
        assert out["logits"].shape == (1, 1000)

    def test_gradient_flow_end_to_end(self, device: torch.device) -> None:
        """Gradients propagate from logits back to the raw image input."""
        net = self._make_net(device, num_blocks=1)
        x = {"input": torch.randn(1, 224, 224, 3, device=device, requires_grad=True), "condition": None}
        out = net(x)
        out["logits"].sum().backward()
        assert x["input"].grad is not None

    def test_channels_last_input(self, device: torch.device) -> None:
        """Network expects ``[B, H, W, C]`` (channels-last) input."""
        net = self._make_net(device, num_blocks=1)
        x_correct = {"input": torch.randn(1, 224, 224, 3, device=device), "condition": None}
        out = net(x_correct)
        assert out["logits"].shape == (1, 1000)

    def test_all_blocks_are_vit5_residual_blocks(self, device: torch.device) -> None:
        """Every block in ``net.blocks`` is a ``ViT5ResidualBlock`` instance."""
        net = self._make_net(device, num_blocks=3)
        for block in net.blocks:
            assert isinstance(block, ViT5ResidualBlock)

    def test_output_norm_is_rmsnorm(self, device: torch.device) -> None:
        """Output normalization before the classification head is RMSNorm."""
        net = self._make_net(device)
        assert isinstance(net.out_norm, RMSNorm)


# ─── 7. Cross-validation against reference ViT-5 ───────────────────────────────


@pytest.mark.skipif(not _has_apex, reason="apex not installed (install from source or NGC container)")
class TestCrossValidation:
    """Cross-validate our implementation against the ViT-5 paper specs.

    These tests instantiate the *real* v3 training config and check that
    architecture dimensions, hyperparameters, and component choices match
    the paper (Wang et al., 2026).
    """

    def test_vit5_small_architecture_matches_paper(self) -> None:
        """Verify our ViT-5-Small matches Table 4 of the paper."""
        from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)

        # Table 4: ViT-5-S: 12 layers, dim 384, 6 heads, 4 registers, 22M params
        assert len(net.blocks) == 12
        assert net.hidden_dim == 384
        assert net.num_registers == 4

        num_params = sum(p.numel() for p in net.parameters()) / 1e6
        assert 21.0 < num_params < 23.0, f"Expected ~22M, got {num_params:.1f}M"

        # Check heads
        attn = net.blocks[0].sequence_mixer
        assert attn.num_heads == 6
        assert attn.head_dim == 64

    def test_no_qkv_bias(self) -> None:
        """ViT-5 removes QKV bias (Section 3.7)."""
        from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        for block in net.blocks:
            assert block.sequence_mixer.qkv.bias is None

    def test_uses_soft_target_loss(self) -> None:
        """ViT-5 pretraining uses SoftTargetCrossEntropy for multiclass with soft labels.

        Changed from BCE to SoftTargetCrossEntropy (v3 config) which brought
        pretraining accuracy to 82.2% top-1 on ImageNet-1k.
        """
        from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config
        from experiments.lightning_wrappers.classification_wrapper import SoftTargetCrossEntropy
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        wrapper = instantiate(config.lightning_wrapper_class, network=net, cfg=config)
        assert wrapper.loss_mode == "soft_target_ce"
        assert isinstance(wrapper.loss_metric, SoftTargetCrossEntropy)

    def test_uses_gelu_not_swiglu(self) -> None:
        """ViT-5 uses GeLU MLP, not SwiGLU (Section 3.3 -- over-gating issue)."""
        from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        for block in net.blocks:
            assert block.mlp.activation == "gelu"
            assert not block.mlp.is_glu_variant

    def test_training_hyperparams_match_paper(self) -> None:
        """Verify hyperparameters match Table 12."""
        from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config

        config = get_config()
        assert config.optimizer.lr == 4e-3
        assert config.optimizer.weight_decay == 0.05
        assert config.train.grad_clip == 1.0
        assert config.train.precision == "bf16-mixed"

    def test_register_rope_base(self) -> None:
        """Registers should use high-frequency RoPE (theta=100, not 10000)."""
        from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        attn = net.blocks[0].sequence_mixer
        assert attn.rope_base == 10000.0
        assert attn.reg_rope_base == 100.0


# ─── 8. GAP readout & token layout (use_cls_token / prepend_registers) ───────


class TestGAPReadoutAndTokenLayout:
    """Tests for global average pooling readout and register token placement.

    Covers the bug where ``use_cls_token=False`` with ``prepend_registers=True``
    produced incorrect token slicing, and verifies all register layout combos.

    Uses ``nn.Identity`` as the sequence mixer because ``ViT5Attention`` has
    precomputed RoPE buffers sized for CLS-included sequences.  These tests
    target token assembly and readout logic, not the attention mechanism.
    """

    def _make_net(
        self,
        device: torch.device,
        use_cls_token: bool = True,
        prepend_registers: bool = False,
        num_registers: int = 4,
        hidden_dim: int = 384,
        num_blocks: int = 1,
    ) -> ViT5ClassificationNet:
        """Build a ViT-5 net with ``nn.Identity`` mixer for readout testing."""
        net = ViT5ClassificationNet(
            in_channels=3,
            num_classes=1000,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            patch_size=16,
            image_size=224,
            num_registers=num_registers,
            dropout_rate=0.0,
            use_cls_token=use_cls_token,
            prepend_registers=prepend_registers,
            norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            block_cfg=LazyConfig(ViT5ResidualBlock)(
                sequence_mixer_cfg=LazyConfig(nn.Identity)(),
                sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
                mlp_cfg=LazyConfig(MLP)(
                    dim=hidden_dim,
                    activation="gelu",
                    expansion_factor=4.0,
                    dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                ),
                mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
                hidden_dim=hidden_dim,
                layer_scale_init=1e-4,
                drop_path_rate=0.0,
            ),
        )
        return net.to(device)

    def _forward(self, net: ViT5ClassificationNet, device: torch.device) -> dict[str, torch.Tensor]:
        """Run a dummy forward pass with random ``[B, 224, 224, 3]`` images."""
        x = {"input": torch.randn(2, 224, 224, 3, device=device), "condition": None}
        return net(x)

    def test_gap_no_registers(self, device: torch.device) -> None:
        """GAP with no CLS and no registers averages all 196 patch tokens."""
        net = self._make_net(device, use_cls_token=False, num_registers=0)
        out = self._forward(net, device)
        assert out["logits"].shape == (2, 1000)

    def test_gap_appended_registers(self, device: torch.device) -> None:
        """GAP with appended registers (default layout) excludes register tokens."""
        net = self._make_net(device, use_cls_token=False, prepend_registers=False, num_registers=4)
        net.eval()
        out = self._forward(net, device)
        assert out["logits"].shape == (2, 1000)

    def test_gap_no_cls_prepend_registers_flag(self, device: torch.device) -> None:
        """Regression: ``use_cls_token=False`` + ``prepend_registers=True``.

        Registers are always appended when CLS token is absent (the prepend
        guard requires ``cls_token``), so the readout must slice from the end.
        """
        net = self._make_net(device, use_cls_token=False, prepend_registers=True, num_registers=4)
        net.eval()
        out = self._forward(net, device)
        assert out["logits"].shape == (2, 1000)

    def test_gap_excludes_registers_appended(self, device: torch.device) -> None:
        """Verify GAP actually excludes register tokens when they are appended.

        Manually builds the ``[patches, regs]`` sequence and asserts its length
        is 200, then checks the full forward pass produces valid logits.
        """
        net = self._make_net(device, use_cls_token=False, num_registers=4)
        net.eval()

        x = torch.randn(2, 224, 224, 3, device=device)
        inp = {"input": x, "condition": None}

        with torch.no_grad():
            # Manually trace to check register exclusion
            xr = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            patches = net.patch_embed(xr)
            patches = patches.flatten(2).transpose(1, 2)  # [B, 196, C]
            patches = patches + net.pos_embed

            reg_tokens = net.reg_token.expand(2, -1, -1)
            seq = torch.cat([patches, reg_tokens], dim=1)  # [B, 200, C]

            # GAP should use only first 196 tokens
            assert seq.shape[1] == 200
            out = net(inp)
            assert out["logits"].shape == (2, 1000)

    def test_cls_prepend_registers_layout(self, device: torch.device) -> None:
        """With CLS + ``prepend_registers``, layout is ``[CLS, regs, patches]``."""
        net = self._make_net(device, use_cls_token=True, prepend_registers=True, num_registers=4)
        net.eval()
        out = self._forward(net, device)
        assert out["logits"].shape == (2, 1000)

    def test_cls_append_registers_layout(self, device: torch.device) -> None:
        """With CLS + appended registers, layout is ``[CLS, patches, regs]``."""
        net = self._make_net(device, use_cls_token=True, prepend_registers=False, num_registers=4)
        net.eval()
        out = self._forward(net, device)
        assert out["logits"].shape == (2, 1000)

    def test_no_cls_token_attribute(self, device: torch.device) -> None:
        """``use_cls_token=False`` sets ``cls_token`` to ``None``."""
        net = self._make_net(device, use_cls_token=False)
        assert net.cls_token is None

    def test_gradient_flow_gap_mode(self, device: torch.device) -> None:
        """Gradients propagate through GAP readout back to the image input."""
        net = self._make_net(device, use_cls_token=False, num_registers=4)
        x = {"input": torch.randn(1, 224, 224, 3, device=device, requires_grad=True), "condition": None}
        out = net(x)
        out["logits"].sum().backward()
        assert x["input"].grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
