# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 0 sanity tests for the zero-init parallel bottleneck adapter (C0/C1/C2).

Tests verify the three Phase-0 claims from the experiment brief:

4.1  **Bit-exact identity** — after attaching any C0/C1/C2 adapter, the model
     output is *exactly* (``torch.equal``, not ``allclose``) equal to the
     pretrain output.  Tested for fp32 on CPU and on CUDA when available.

4.2  **Gradient unlock** — at step 0, ``W_up.grad`` is non-zero but
     ``W_down.grad`` is exactly zero.  After one optimiser step, ``W_down.grad``
     becomes non-zero (branch has unlocked).

4.3  **Null-effect training check** — training the adapter with LR=0 produces a
     loss curve indistinguishable from the frozen-base / no-adapter arm.

Additionally tests the ``readout="corner"`` extension to ``ViT5ClassificationNet``
and verifies that C0/C1/C2 all share the same forward/backward structure.

Run::

    PYTHONPATH=. python -m pytest tests/modules/test_vit5_parallel_hyena_adapter.py -v -o addopts=""
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


# ── Constants matching the toy experiment ──────────────────────────────────

HIDDEN_DIM = 64  # smaller than experiment for test speed
NUM_HEADS = 4
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
GRID_W = 7  # 7×7 = 49 tokens (small but tests index arithmetic)
NUM_PATCHES = GRID_W * GRID_W
PATCH_SIZE = 4
IMAGE_SIZE = GRID_W * PATCH_SIZE  # 28
IN_CHANNELS = 1
NUM_CLASSES = 10
RANK = 8
NUM_BLOCKS = 2


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request) -> torch.device:
    """Parametrize over CPU and CUDA when available."""
    return torch.device(request.param)


def _make_attn_cfg(has_cls: bool = False) -> LazyConfig:
    return LazyConfig(ViT5Attention)(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_patches_h=GRID_W,
        num_patches_w=GRID_W,
        num_registers=0,
        has_cls=has_cls,
        qk_norm=LazyConfig(RMSNorm)(dim=HEAD_DIM, eps=1e-6),
        rope_base=10000.0,
        reg_rope_base=100.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=False,
        out_proj_bias=False,
    )


def _make_hyena2d_inner_cfg() -> LazyConfig:
    """Build a minimal QKVSequenceMixer(Hyena2D) config at bottleneck width RANK."""
    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.hyena_nd import Hyena
    from nvsubquadratic.modules.kernels_nd import SIRENKernelND
    from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
    from nvsubquadratic.utils.init import small_init
    from nvsubquadratic.utils.qk_norm import L2Norm

    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=RANK,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=RANK,
                fft_padding="zero",
                grid_type="double",
                is_causal=False,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=RANK,
                    mlp_hidden_dim=32,
                    num_layers=2,
                    embedding_dim=32,
                    omega_0=30.0,
                    L_cache=GRID_W,
                    use_bias=True,
                    hidden_omega_0=1.0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * RANK,
                out_channels=3 * RANK,
                kernel_size=3,
                groups=3 * RANK,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=RANK),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=RANK),
            qk_norm_cfg=LazyConfig(L2Norm)(),
        ),
        init_method_in=small_init,
    )


def _make_parallel_mixer_cfg_c0() -> LazyConfig:
    """C0: no inner mixer (nn.Identity)."""
    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=_make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=None,
    )


def _make_parallel_mixer_cfg_c1() -> LazyConfig:
    """C1: local depthwise Conv2d inner mixer."""
    from nvsubquadratic.modules.vit5_depthwise_conv2d_mixer import ViT5DepthwiseConv2dMixer

    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=_make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=LazyConfig(ViT5DepthwiseConv2dMixer)(channels=RANK, grid_w=GRID_W, kernel_size=3),
    )


def _make_parallel_mixer_cfg_c2() -> LazyConfig:
    """C2: Hyena2D global implicit filter."""
    from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter

    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=_make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=_make_hyena2d_inner_cfg(),
            grid_w=GRID_W,
        ),
    )


# Keep backward-compat alias for tests that just need "the adapter net"
_make_parallel_mixer_cfg = _make_parallel_mixer_cfg_c2


def _make_block_cfg(mixer_cfg: LazyConfig) -> LazyConfig:
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=mixer_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=4.0,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=1e-4,
        drop_path_rate=0.0,
    )


def _make_pretrain_net(device: torch.device) -> ViT5ClassificationNet:
    """Build the pretrain attention-only network."""
    from nvsubquadratic.lazy_config import instantiate

    cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        readout="corner",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=_make_block_cfg(_make_attn_cfg()),
    )
    return instantiate(cfg).to(device)


def _make_adapter_net(device: torch.device, mixer_cfg_fn=None) -> ViT5ClassificationNet:
    """Build the net with a parallel bottleneck adapter.

    Args:
        device: Target device.
        mixer_cfg_fn: Callable returning a LazyConfig for the mixer.
            Defaults to ``_make_parallel_mixer_cfg`` (C2 / Hyena2D).
    """
    from nvsubquadratic.lazy_config import instantiate

    if mixer_cfg_fn is None:
        mixer_cfg_fn = _make_parallel_mixer_cfg

    cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        readout="corner",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=_make_block_cfg(mixer_cfg_fn()),
    )
    return instantiate(cfg).to(device)


def _make_input(device: torch.device, dtype: torch.dtype = torch.float32) -> dict:
    """Build a synthetic batch matching the toy config."""
    torch.manual_seed(0)
    x = torch.randn(2, IMAGE_SIZE, IMAGE_SIZE, IN_CHANNELS, device=device, dtype=dtype)
    return {"input": x, "condition": None}


def _copy_all_shared_weights(src: ViT5ClassificationNet, dst: ViT5ClassificationNet) -> None:
    """Copy ALL shared parameters from src into dst, remapping sequence_mixer keys.

    dst's blocks have ``sequence_mixer.attn.*``; src's have ``sequence_mixer.*``.
    After this call, every parameter in dst that has a counterpart in src (with
    identical shape) has the src value.  Adapter-only params (w_down, w_up,
    inner_mixer) are left at their own initialisation.
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for k, v in src_sd.items():
        # Try the remapped key first (attention lives inside .attn. in dst)
        adapted_key = k.replace(".sequence_mixer.", ".sequence_mixer.attn.")
        if adapted_key in dst_sd and dst_sd[adapted_key].shape == v.shape:
            dst_sd[adapted_key] = v
        elif k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v
    dst.load_state_dict(dst_sd, strict=False)


# Backward-compat alias
_copy_attn_weights = _copy_all_shared_weights


# ── Test: readout="corner" ──────────────────────────────────────────────────


class TestCornerReadout:
    """Tests for the readout='corner' extension to ViT5ClassificationNet."""

    def test_corner_readout_output_shape(self, device: torch.device) -> None:
        """Forward pass with readout='corner' returns [B, num_classes] logits."""
        net = _make_pretrain_net(device)
        x = _make_input(device)
        out = net(x)
        assert out["logits"].shape == (2, NUM_CLASSES)

    def test_corner_idx_correct(self, device: torch.device) -> None:
        """Bottom-right corner index is (grid_h-1)*grid_w + (grid_w-1)."""
        net = _make_pretrain_net(device)
        expected = (GRID_W - 1) * GRID_W + (GRID_W - 1)
        assert net._corner_idx == expected

    def test_corner_raises_with_registers(self) -> None:
        """readout='corner' raises ValueError when num_registers > 0."""
        from nvsubquadratic.lazy_config import instantiate

        with pytest.raises(ValueError, match="num_registers=0"):
            instantiate(
                LazyConfig(ViT5ClassificationNet)(
                    in_channels=1,
                    num_classes=10,
                    hidden_dim=HIDDEN_DIM,
                    num_blocks=1,
                    patch_size=4,
                    image_size=28,
                    num_registers=2,
                    readout="corner",
                    norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
                    block_cfg=_make_block_cfg(_make_attn_cfg()),
                )
            )

    def test_no_padding_when_t_divisible(self, device: torch.device) -> None:
        """With num_registers=0, no CLS, and T % grid_w == 0, no zero padding is added."""
        net = _make_pretrain_net(device)
        # NUM_PATCHES = GRID_W * GRID_W, no extra tokens → no padding
        assert net._pad_size == 0
        assert net._zero_pad is None


# ── Test 4.1: Bit-exact identity ────────────────────────────────────────────


class TestBitExactIdentity:
    """§4.1: Adapter must be a bit-exact no-op at initialisation."""

    def _get_y0_y1(self, device: torch.device, dtype: torch.dtype):
        torch.manual_seed(42)
        pretrain = _make_pretrain_net(device).to(dtype=dtype)
        adapter_net = _make_adapter_net(device).to(dtype=dtype)
        # Copy pretrained attention weights into adapter_net.attn
        _copy_attn_weights(pretrain, adapter_net)
        # Also copy head, norms, pos_embed, patch_embed
        src_sd = pretrain.state_dict()
        dst_sd = adapter_net.state_dict()
        for k in ("patch_embed.weight", "pos_embed", "out_proj.weight", "out_norm.weight"):
            if k in src_sd and k in dst_sd:
                dst_sd[k] = src_sd[k]
        adapter_net.load_state_dict(dst_sd, strict=False)

        pretrain.eval()
        adapter_net.eval()
        x = _make_input(device, dtype=dtype)
        with torch.no_grad():
            y0 = pretrain(x)["logits"]
            y1 = adapter_net(x)["logits"]
        return y0, y1

    def test_bit_exact_fp32_cpu(self) -> None:
        """Output is bit-for-bit identical (torch.equal, not allclose) on CPU fp32."""
        y0, y1 = self._get_y0_y1(torch.device("cpu"), torch.float32)
        assert torch.equal(y0, y1), (
            f"Outputs differ: max_abs_diff={(y0 - y1).abs().max().item():.6e}. "
            "Check that w_up.weight is exactly zero and bias=False."
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bit_exact_fp32_cuda(self) -> None:
        """Output is bit-for-bit identical on CUDA fp32."""
        y0, y1 = self._get_y0_y1(torch.device("cuda"), torch.float32)
        assert torch.equal(y0, y1)

    def test_w_up_initialised_to_zero(self) -> None:
        """w_up.weight must be exactly zero, not merely small."""
        net = _make_adapter_net(torch.device("cpu"))
        for block in net.blocks:
            mixer = block.sequence_mixer
            assert isinstance(mixer, ViT5ParallelHyenaSequenceMixer)
            assert torch.equal(mixer.w_up.weight, torch.zeros_like(mixer.w_up.weight)), (
                "w_up.weight is not exactly zero"
            )

    def test_no_bias_on_w_up(self) -> None:
        """w_up must have bias=False so zero weight means zero output unconditionally."""
        net = _make_adapter_net(torch.device("cpu"))
        for block in net.blocks:
            assert block.sequence_mixer.w_up.bias is None


# ── Test 4.2: Gradient unlock ────────────────────────────────────────────────


class TestGradientUnlock:
    """§4.2: W_up gets gradients at step 0; W_down/Hyena unlocks after one step."""

    def _build_and_step(self, device: torch.device):
        """Return (net, w_up_ref, w_down_ref) after setting up."""
        net = _make_adapter_net(device)
        net.train()
        return net

    def test_wup_grad_nonzero_at_step0(self, device: torch.device) -> None:
        """W_up.grad is non-zero after the first backward at step 0."""
        net = self._build_and_step(device)
        x = _make_input(device)
        labels = torch.zeros(2, dtype=torch.long, device=device)
        out = net(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], labels)
        loss.backward()

        for i, block in enumerate(net.blocks):
            mixer = block.sequence_mixer
            grad = mixer.w_up.weight.grad
            assert grad is not None, f"block {i}: w_up.grad is None"
            assert grad.abs().sum() > 0, f"block {i}: w_up.grad is all zero at step 0"

    def test_wdown_grad_zero_at_step0(self, device: torch.device) -> None:
        """W_down.grad is exactly zero at step 0 (W_up == 0 blocks the path).

        This is the expected LoRA-style behaviour: only W_up can move at step 0.
        W_down and the Hyena kernel unlock after W_up moves off zero.
        """
        net = self._build_and_step(device)
        x = _make_input(device)
        labels = torch.zeros(2, dtype=torch.long, device=device)
        out = net(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], labels)
        loss.backward()

        for i, block in enumerate(net.blocks):
            mixer = block.sequence_mixer
            grad = mixer.w_down.weight.grad
            assert grad is not None, f"block {i}: w_down.grad is None"
            assert torch.equal(grad, torch.zeros_like(grad)), (
                f"block {i}: w_down.grad is non-zero at step 0 ({grad.abs().max().item():.3e})"
            )

    def test_wdown_grad_nonzero_after_optimiser_step(self, device: torch.device) -> None:
        """W_down.grad becomes non-zero after one optimiser step (branch has unlocked)."""
        net = self._build_and_step(device)
        opt = torch.optim.SGD(net.parameters(), lr=1.0)
        x = _make_input(device)
        labels = torch.zeros(2, dtype=torch.long, device=device)

        # Step 0 — W_up moves off zero
        loss = torch.nn.functional.cross_entropy(net(x)["logits"], labels)
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Step 1 — W_down should now receive gradients
        loss = torch.nn.functional.cross_entropy(net(x)["logits"], labels)
        loss.backward()

        any_nonzero = False
        for block in net.blocks:
            grad = block.sequence_mixer.w_down.weight.grad
            if grad is not None and grad.abs().sum() > 0:
                any_nonzero = True
                break

        assert any_nonzero, (
            "w_down.grad is still zero after one optimiser step. "
            "W_up may not have moved off zero, or gradient path is broken."
        )


# ── Test: Ablation ladder C0/C1/C2 ──────────────────────────────────────────


class TestAblationLadder:
    """Structural tests shared across C0, C1, C2 mixer variants.

    All three must satisfy the zero-init and gradient properties.
    """

    @pytest.mark.parametrize(
        "mixer_cfg_fn",
        [_make_parallel_mixer_cfg_c0, _make_parallel_mixer_cfg_c1, _make_parallel_mixer_cfg_c2],
        ids=["C0_identity", "C1_depthwise_conv", "C2_hyena2d"],
    )
    def test_bit_exact_identity_all_variants(self, device: torch.device, mixer_cfg_fn) -> None:
        """All three ablation variants are bit-exact no-ops at init."""
        pretrain = _make_pretrain_net(device)
        adapter = _make_adapter_net(device, mixer_cfg_fn)
        _copy_attn_weights(pretrain, adapter)

        pretrain.eval()
        adapter.eval()
        x = _make_input(device)
        with torch.no_grad():
            y0 = pretrain(x)["logits"]
            y1 = adapter(x)["logits"]

        assert torch.equal(y0, y1), f"{mixer_cfg_fn.__name__}: max_diff={(y0 - y1).abs().max().item():.3e}"

    @pytest.mark.parametrize(
        "mixer_cfg_fn",
        [_make_parallel_mixer_cfg_c0, _make_parallel_mixer_cfg_c1, _make_parallel_mixer_cfg_c2],
        ids=["C0_identity", "C1_depthwise_conv", "C2_hyena2d"],
    )
    def test_w_up_zero_all_variants(self, mixer_cfg_fn) -> None:
        """W_up is exactly zero for all ablation variants."""
        net = _make_adapter_net(torch.device("cpu"), mixer_cfg_fn)
        for block in net.blocks:
            mixer = block.sequence_mixer
            assert torch.equal(mixer.w_up.weight, torch.zeros_like(mixer.w_up.weight))

    @pytest.mark.parametrize(
        "mixer_cfg_fn",
        [_make_parallel_mixer_cfg_c0, _make_parallel_mixer_cfg_c1, _make_parallel_mixer_cfg_c2],
        ids=["C0_identity", "C1_depthwise_conv", "C2_hyena2d"],
    )
    def test_output_shape_all_variants(self, device: torch.device, mixer_cfg_fn) -> None:
        """All variants return [B, num_classes] logits."""
        net = _make_adapter_net(device, mixer_cfg_fn)
        x = _make_input(device)
        out = net(x)
        assert out["logits"].shape == (2, NUM_CLASSES)


# ── Test 4.3: Null-effect training check ────────────────────────────────────


class TestNullEffectTraining:
    """§4.3: Training adapter with LR=0 must not change the loss trajectory."""

    def _run_n_steps(self, net: nn.Module, x: dict, labels: torch.Tensor, n: int, lr: float):
        """Run n training steps and return the final loss value."""
        opt = torch.optim.SGD(net.parameters(), lr=lr)
        net.train()
        for _ in range(n):
            opt.zero_grad()
            out = net(x)
            loss = torch.nn.functional.cross_entropy(out["logits"], labels)
            loss.backward()
            opt.step()
        return loss.item()

    def test_lr0_adapter_matches_frozen_base(self, device: torch.device) -> None:
        """Adapter with LR=0 is identical to the frozen base model at every step.

        Compares the FROZEN base (lr=0) against the FROZEN adapter (lr=0).
        Both models are frozen so loss stays constant; the adapter adds exactly
        0 at every step because w_up=0 and nothing changes.

        The key claim: ``loss(frozen_base) == loss(frozen_adapter)`` at every
        step.  A discrepancy would mean the adapter wiring itself (not its
        parameters) is altering the computation — a bug.
        """
        torch.manual_seed(0)
        base_net = _make_pretrain_net(device)

        # Build adapter net and copy ALL shared params so the only difference
        # is the structural addition of w_down/inner_mixer/w_up (which output 0).
        adapter_net = _make_adapter_net(device)
        _copy_all_shared_weights(base_net, adapter_net)

        x = _make_input(device)
        labels = torch.zeros(2, dtype=torch.long, device=device)

        N = 5
        # Both frozen (lr=0) — loss must be IDENTICAL at every step.
        loss_base = self._run_n_steps(base_net, x, labels, N, lr=0.0)
        loss_adapter_lr0 = self._run_n_steps(adapter_net, x, labels, N, lr=0.0)

        assert abs(loss_base - loss_adapter_lr0) < 1e-5, (
            f"LR=0 adapter loss {loss_adapter_lr0:.6f} != frozen base loss {loss_base:.6f}. "
            "Adapter wiring contributes to the loss even with LR=0."
        )
