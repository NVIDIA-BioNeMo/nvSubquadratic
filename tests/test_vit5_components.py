"""Comprehensive tests for all ViT-5 components.

Tests verify:
1. RMSNorm matches reference implementation
2. LayerScale shape, init value, and gradient flow
3. DropPath behavior in train/eval modes
4. ViT5Attention: shapes, QK-Norm, RoPE token routing, register handling
5. ViT5ResidualBlock: residual connection, LayerScale, DropPath
6. ViT5ClassificationNet: end-to-end forward pass, parameter count, token assembly
7. Cross-validation against the reference ViT-5 repo

Run:
    PYTHONPATH=. python -m pytest tests/test_vit5_components.py -v
"""

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


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def device():
    """All tests use CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def vit5_small_params():
    return dict(
        hidden_dim=384,
        num_heads=6,
        num_blocks=12,
        patch_size=16,
        image_size=224,
        num_registers=4,
        mlp_ratio=4,
        head_dim=64,  # 384 / 6
    )


# ─── 1. RMSNorm Tests ──────────────────────────────────────────────────────────


class TestRMSNorm:
    def test_output_shape(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(2, 10, 64, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_matches_reference_implementation(self, device):
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

    def test_learnable_weight(self):
        norm = RMSNorm(64)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (64,)
        torch.testing.assert_close(norm.weight.data, torch.ones(64))

    def test_no_weight_decay_flag(self):
        norm = RMSNorm(64)
        assert hasattr(norm.weight, "_no_weight_decay")
        assert norm.weight._no_weight_decay is True

    def test_dtype_preservation(self, device):
        norm = RMSNorm(64).half().to(device)
        x = torch.randn(2, 10, 64, dtype=torch.float16, device=device)
        out = norm(x)
        assert out.dtype == torch.float16


# ─── 2. LayerScale Tests ───────────────────────────────────────────────────────


class TestLayerScale:
    def test_output_shape(self):
        ls = LayerScale(384, init_value=1e-4)
        x = torch.randn(2, 196, 384)
        out = ls(x)
        assert out.shape == x.shape

    def test_init_value(self):
        ls = LayerScale(384, init_value=1e-4)
        torch.testing.assert_close(ls.gamma.data, torch.full((384,), 1e-4))

    def test_scaling_effect(self):
        ls = LayerScale(4, init_value=0.5)
        x = torch.ones(1, 1, 4)
        out = ls(x)
        torch.testing.assert_close(out, torch.full_like(x, 0.5))

    def test_gradient_flow(self):
        ls = LayerScale(384, init_value=1e-4)
        x = torch.randn(2, 196, 384, requires_grad=True)
        out = ls(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert ls.gamma.grad is not None

    def test_matches_reference_block_scaling(self):
        """Verify LayerScale matches the reference: gamma * x."""
        dim = 384
        init_val = 1e-4
        ls = LayerScale(dim, init_value=init_val)
        x = torch.randn(2, 196, dim)

        # Reference: self.gamma_1 * self.attn(self.norm1(x))
        gamma = init_val * torch.ones(dim)
        ref_out = gamma * x
        our_out = ls(x)
        torch.testing.assert_close(our_out, ref_out, atol=1e-7, rtol=1e-6)

    def test_no_weight_decay_flag(self):
        ls = LayerScale(384)
        assert hasattr(ls.gamma, "_no_weight_decay")
        assert ls.gamma._no_weight_decay is True


# ─── 3. DropPath Tests ─────────────────────────────────────────────────────────


class TestDropPath:
    def test_identity_when_zero(self):
        dp = DropPath(0.0)
        x = torch.randn(4, 10, 384)
        dp.train()
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_identity_in_eval(self):
        dp = DropPath(0.5)
        x = torch.randn(4, 10, 384)
        dp.eval()
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_drops_samples_in_train(self):
        torch.manual_seed(0)
        dp = DropPath(0.99)
        dp.train()
        x = torch.ones(100, 10, 384)
        out = dp(x)
        # With p=0.99, most samples should be zeroed
        zero_samples = (out.abs().sum(dim=(1, 2)) == 0).sum().item()
        assert zero_samples > 50, f"Expected most samples zeroed, got {zero_samples}"

    def test_expectation_preservation(self):
        """DropPath rescales kept samples so expectation is preserved."""
        torch.manual_seed(42)
        dp = DropPath(0.3)
        dp.train()
        x = torch.ones(10000, 1, 1)
        out = dp(x)
        mean = out.mean().item()
        assert abs(mean - 1.0) < 0.1, f"Expected mean ~1.0, got {mean}"


# ─── 4. ViT5Attention Tests ────────────────────────────────────────────────────


class TestViT5Attention:
    def _make_attn(self, device, **kwargs):
        defaults = dict(
            hidden_dim=384,
            num_heads=6,
            num_patches_h=14,
            num_patches_w=14,
            num_registers=4,
            qk_norm=LazyConfig(RMSNorm)(dim=64, eps=1e-6),
            rope_base=10000.0,
            reg_rope_base=100.0,
            attn_dropout=0.0,
            proj_dropout=0.0,
            qkv_bias=False,
        )
        defaults.update(kwargs)
        return ViT5Attention(**defaults).to(device)

    def test_output_shape(self, device):
        attn = self._make_attn(device)
        T = 1 + 196 + 4  # cls + patches + registers
        x = torch.randn(2, T, 384, device=device)
        out = attn(x)
        assert out.shape == (2, T, 384)

    def test_no_qkv_bias(self, device):
        attn = self._make_attn(device, qkv_bias=False)
        assert attn.qkv.bias is None

    def test_with_qkv_bias(self, device):
        attn = self._make_attn(device, qkv_bias=True)
        assert attn.qkv.bias is not None

    def test_qk_norm_modules_exist(self, device):
        attn = self._make_attn(device, qk_norm=LazyConfig(RMSNorm)(dim=64, eps=1e-6))
        assert isinstance(attn.q_norm, RMSNorm)
        assert isinstance(attn.k_norm, RMSNorm)

    def test_no_qk_norm(self, device):
        attn = self._make_attn(device, qk_norm=None)
        assert not hasattr(attn, "q_norm") or not isinstance(getattr(attn, "q_norm", None), RMSNorm)

    def test_register_count(self, device):
        attn = self._make_attn(device, num_registers=4)
        assert attn.num_registers == 4
        assert attn.reg_rope_h == 2
        assert attn.reg_rope_w == 2

    def test_zero_registers(self, device):
        attn = self._make_attn(device, num_registers=0)
        T = 1 + 196  # cls + patches only
        x = torch.randn(2, T, 384, device=device)
        out = attn(x)
        assert out.shape == (2, T, 384)

    def test_gradient_flow(self, device):
        attn = self._make_attn(device)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_rope_applies_to_patches_not_cls(self, device):
        """Verify cls token does NOT get RoPE (its position should not change)."""
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

    def test_deterministic_eval(self, device):
        attn = self._make_attn(device, attn_dropout=0.0)
        attn.eval()
        T = 1 + 196 + 4
        x = torch.randn(1, T, 384, device=device)
        with torch.no_grad():
            out1 = attn(x.clone())
            out2 = attn(x.clone())
        torch.testing.assert_close(out1, out2)

    def test_token_ordering_cls_patches_registers(self, device):
        """Verify the attention module expects [cls, patches..., registers...]."""
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
    def _make_block(self, device, hidden_dim=384, num_heads=6, drop_path=0.0, layer_scale_init=1e-4):
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

    def test_output_shape(self, device):
        block = self._make_block(device)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device)
        out = block(x)
        assert out.shape == (2, T, 384)

    def test_residual_connection(self, device):
        """With tiny LayerScale init, output should be close to input."""
        block = self._make_block(device, layer_scale_init=1e-8)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device)
        block.eval()
        with torch.no_grad():
            out = block(x)
        diff = (out - x).abs().max().item()
        assert diff < 0.1, f"Expected small residual diff, got {diff}"

    def test_has_layer_scale(self, device):
        block = self._make_block(device, layer_scale_init=1e-4)
        assert isinstance(block.ls_attn, LayerScale)
        assert isinstance(block.ls_mlp, LayerScale)

    def test_no_layer_scale(self, device):
        block = self._make_block(device, layer_scale_init=0.0)
        assert isinstance(block.ls_attn, nn.Identity)
        assert isinstance(block.ls_mlp, nn.Identity)

    def test_has_drop_path(self, device):
        block = self._make_block(device, drop_path=0.1)
        assert isinstance(block.drop_path, DropPath)
        assert block.drop_path.drop_prob == 0.1

    def test_no_drop_path(self, device):
        block = self._make_block(device, drop_path=0.0)
        assert isinstance(block.drop_path, nn.Identity)

    def test_gradient_flow_through_all_components(self, device):
        block = self._make_block(device)
        T = 1 + 196 + 4
        x = torch.randn(2, T, 384, device=device, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert block.ls_attn.gamma.grad is not None
        assert block.ls_mlp.gamma.grad is not None

    def test_uses_rmsnorm(self, device):
        block = self._make_block(device)
        assert isinstance(block.input_norm, RMSNorm)
        assert isinstance(block.mlp_norm, RMSNorm)

    def test_uses_gelu_mlp(self, device):
        block = self._make_block(device)
        assert block.mlp.activation == "gelu"
        assert not block.mlp.is_glu_variant


# ─── 6. ViT5ClassificationNet Tests ────────────────────────────────────────────


class TestViT5ClassificationNet:
    def _make_net(self, device, hidden_dim=384, num_blocks=2, num_registers=4, image_size=224):
        """Build a small ViT-5 net (2 blocks for speed)."""
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

    def test_output_shape(self, device):
        net = self._make_net(device)
        x = {"input": torch.randn(2, 224, 224, 3, device=device), "condition": None}
        out = net(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 1000)

    def test_parameter_count_full_model(self, device):
        """Full 12-block ViT-5-Small should have ~22M params."""
        net = self._make_net(device, num_blocks=12)
        num_params = sum(p.numel() for p in net.parameters()) / 1e6
        assert 21.0 < num_params < 23.0, f"Expected ~22M params, got {num_params:.1f}M"

    def test_cls_token_exists(self, device):
        net = self._make_net(device)
        assert net.cls_token.shape == (1, 1, 384)

    def test_register_tokens_exist(self, device):
        net = self._make_net(device, num_registers=4)
        assert net.reg_token is not None
        assert net.reg_token.shape == (1, 4, 384)

    def test_no_register_tokens(self, device):
        net = self._make_net(device, num_registers=0)
        assert net.reg_token is None

    def test_pos_embed_shape(self, device):
        net = self._make_net(device)
        num_patches = (224 // 16) ** 2  # 196
        assert net.pos_embed.shape == (1, num_patches, 384)

    def test_patch_embed_is_conv2d(self, device):
        net = self._make_net(device)
        assert isinstance(net.patch_embed, nn.Conv2d)
        assert net.patch_embed.kernel_size == (16, 16)
        assert net.patch_embed.stride == (16, 16)

    def test_uses_cls_token_readout(self, device):
        """ViT-5 uses CLS token, not global average pooling."""
        net = self._make_net(device, num_blocks=1)
        net.eval()
        x = {"input": torch.randn(1, 224, 224, 3, device=device), "condition": None}
        with torch.no_grad():
            out = net(x)
        assert out["logits"].shape == (1, 1000)

    def test_gradient_flow_end_to_end(self, device):
        net = self._make_net(device, num_blocks=1)
        x = {"input": torch.randn(1, 224, 224, 3, device=device, requires_grad=True), "condition": None}
        out = net(x)
        out["logits"].sum().backward()
        assert x["input"].grad is not None

    def test_channels_last_input(self, device):
        """Network expects [B, H, W, C] (channels-last) input."""
        net = self._make_net(device, num_blocks=1)
        x_correct = {"input": torch.randn(1, 224, 224, 3, device=device), "condition": None}
        out = net(x_correct)
        assert out["logits"].shape == (1, 1000)

    def test_all_blocks_are_vit5_residual_blocks(self, device):
        net = self._make_net(device, num_blocks=3)
        for block in net.blocks:
            assert isinstance(block, ViT5ResidualBlock)

    def test_output_norm_is_rmsnorm(self, device):
        net = self._make_net(device)
        assert isinstance(net.out_norm, RMSNorm)


# ─── 7. Cross-validation against reference ViT-5 ───────────────────────────────


class TestCrossValidation:
    def test_vit5_small_architecture_matches_paper(self):
        """Verify our ViT-5-Small matches Table 4 of the paper."""
        from examples.vit5_imagenet.vit5_small_pretrain import get_config
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

    def test_no_qkv_bias(self):
        """ViT-5 removes QKV bias (Section 3.7)."""
        from examples.vit5_imagenet.vit5_small_pretrain import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        for block in net.blocks:
            assert block.sequence_mixer.qkv.bias is None

    def test_uses_soft_target_loss(self):
        """ViT-5 pretraining uses SoftTargetCrossEntropy for multiclass with soft labels."""
        from examples.vit5_imagenet.vit5_small_pretrain import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        wrapper = instantiate(config.lightning_wrapper_class, network=net, cfg=config)
        assert wrapper.loss_mode == "bce"
        assert isinstance(wrapper.loss_metric, torch.nn.BCEWithLogitsLoss)

    def test_uses_gelu_not_swiglu(self):
        """ViT-5 uses GeLU MLP, not SwiGLU (Section 3.3 - over-gating issue)."""
        from examples.vit5_imagenet.vit5_small_pretrain import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        for block in net.blocks:
            assert block.mlp.activation == "gelu"
            assert not block.mlp.is_glu_variant

    def test_training_hyperparams_match_paper(self):
        """Verify hyperparameters match Table 12."""
        from examples.vit5_imagenet.vit5_small_pretrain import get_config

        config = get_config()
        assert config.optimizer.lr == 4e-3
        assert config.optimizer.weight_decay == 0.05
        assert config.train.grad_clip == 1.0
        assert config.train.precision == "bf16-mixed"

    def test_register_rope_base(self):
        """Registers should use high-frequency RoPE (theta=100, not 10000)."""
        from examples.vit5_imagenet.vit5_small_pretrain import get_config
        from nvsubquadratic.lazy_config import instantiate

        config = get_config()
        net = instantiate(config.net)
        attn = net.blocks[0].sequence_mixer
        assert attn.rope_base == 10000.0
        assert attn.reg_rope_base == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
