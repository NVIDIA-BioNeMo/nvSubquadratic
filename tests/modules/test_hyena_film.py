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

"""Tests for Hyena-related FiLM conditioning components.

Tests verify:
1. RegisterPooling: shape, initialization, weighted averaging, gradient flow
2. KernelFiLMGenerator: output structure, identity init, zero-init weights, gradient flow
3. ViT5ResidualBlock with FiLM: register pooling creation, conditioning passthrough, gradients
4. ViT5HyenaAdapter: reshape round-trip, mixer_kwargs forwarding

Run:
    PYTHONPATH=. python -m pytest tests/modules/test_hyena_film.py -v -o addopts=""

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
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterCompressConcat, RegisterPooling
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock


# ─── Test helper modules ────────────────────────────────────────────────────────


class _KwargsPassthroughMixer(nn.Module):
    """Identity mixer that accepts ``**kwargs`` (e.g. conditioning).

    If ``conditioning`` is provided, it is broadcast-added to ``x`` so that
    gradients flow through both the main input and the conditioning path.
    Used by :class:`TestResidualBlockFiLM` to verify end-to-end gradient flow
    without needing a full Hyena stack.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        cond = kwargs.get("conditioning", None)
        if cond is not None:
            x = x + cond.unsqueeze(1)
        return x


class _KwargCaptureMixer(nn.Module):
    """Identity mixer that records received kwargs for test assertions.

    After a forward call, ``self.last_kwargs`` contains the ``**kwargs``
    that were passed.  Used by :class:`TestViT5HyenaAdapter` to verify
    that ``conditioning`` is forwarded through the adapter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.last_kwargs = kwargs
        return x


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def device() -> torch.device:
    """Return CUDA device when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 1. RegisterPooling Tests ───────────────────────────────────────────────


class TestRegisterPooling:
    """Tests for :class:`RegisterPooling` — learnable weighted average over register tokens.

    ``RegisterPooling`` maps ``[B, num_registers, C]`` to ``[B, C]`` via
    softmax-weighted combination.  At initialization the logits are zero,
    giving uniform (mean) pooling.
    """

    def test_output_shape(self, device: torch.device) -> None:
        """Output is ``[B, C]`` when input is ``[B, num_registers, C]``."""
        pool = RegisterPooling(num_registers=4).to(device)
        regs = torch.randn(2, 4, 384, device=device)
        out = pool(regs)
        assert out.shape == (2, 384)

    def test_uniform_init(self) -> None:
        """Logits initialized to zeros yield uniform weights ``1/num_registers``."""
        pool = RegisterPooling(num_registers=4)
        weights = torch.softmax(pool.logits, dim=0)
        expected = torch.full((4,), 0.25)
        torch.testing.assert_close(weights, expected)

    def test_uniform_weights_mean_registers(self, device: torch.device) -> None:
        """With uniform weights, output equals the mean of register tokens."""
        pool = RegisterPooling(num_registers=4).to(device)
        regs = torch.randn(2, 4, 64, device=device)
        out = pool(regs)
        expected = regs.mean(dim=1)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    def test_gradient_flow(self, device: torch.device) -> None:
        """Gradients reach both the input registers and the learnable logits."""
        pool = RegisterPooling(num_registers=4).to(device)
        regs = torch.randn(2, 4, 64, device=device, requires_grad=True)
        out = pool(regs)
        out.sum().backward()
        assert regs.grad is not None
        assert pool.logits.grad is not None

    def test_no_weight_decay_flag(self) -> None:
        """Logits parameter is marked with ``_no_weight_decay``."""
        pool = RegisterPooling(num_registers=4)
        assert hasattr(pool.logits, "_no_weight_decay")
        assert pool.logits._no_weight_decay is True

    def test_single_register(self, device: torch.device) -> None:
        """With one register, output equals that single register token."""
        pool = RegisterPooling(num_registers=1).to(device)
        regs = torch.randn(2, 1, 64, device=device)
        out = pool(regs)
        torch.testing.assert_close(out, regs.squeeze(1), atol=1e-6, rtol=1e-5)

    def test_learned_weights_shift_output(self, device: torch.device) -> None:
        """After setting logits to heavily favor one register, output matches it."""
        pool = RegisterPooling(num_registers=3).to(device)
        with torch.no_grad():
            pool.logits.fill_(-100.0)
            pool.logits[0] = 100.0  # heavily weight first register

        regs = torch.randn(2, 3, 64, device=device)
        out = pool(regs)
        torch.testing.assert_close(out, regs[:, 0, :], atol=1e-4, rtol=1e-3)


# ─── 1b. RegisterCompressConcat Tests ────────────────────────────────────────


class TestRegisterCompressConcat:
    """Tests for :class:`RegisterCompressConcat` — compress and concatenate register tokens.

    ``RegisterCompressConcat`` maps ``[B, num_registers, hidden_dim]`` to
    ``[B, num_registers * compressed_dim]`` via a shared linear compression
    followed by flattening.
    """

    def test_output_shape(self, device: torch.device) -> None:
        """Output is ``[B, num_registers * compressed_dim]``."""
        pool = RegisterCompressConcat(num_registers=14, hidden_dim=384, compressed_dim=32).to(device)
        regs = torch.randn(2, 14, 384, device=device)
        out = pool(regs)
        assert out.shape == (2, 14 * 32)

    def test_out_dim_property(self) -> None:
        """``out_dim`` matches ``num_registers * compressed_dim``."""
        pool = RegisterCompressConcat(num_registers=14, hidden_dim=384, compressed_dim=32)
        assert pool.out_dim == 14 * 32

    def test_gradient_flow(self, device: torch.device) -> None:
        """Gradients reach the input registers and the compression weights."""
        pool = RegisterCompressConcat(num_registers=4, hidden_dim=64, compressed_dim=16).to(device)
        regs = torch.randn(2, 4, 64, device=device, requires_grad=True)
        out = pool(regs)
        out.sum().backward()
        assert regs.grad is not None
        assert pool.compress.weight.grad is not None

    def test_no_bias(self) -> None:
        """Compression linear has no bias (intentional — FiLM MLP has its own)."""
        pool = RegisterCompressConcat(num_registers=4, hidden_dim=64, compressed_dim=16)
        assert pool.compress.bias is None

    def test_single_register(self, device: torch.device) -> None:
        """With one register, output equals the compressed single token."""
        pool = RegisterCompressConcat(num_registers=1, hidden_dim=64, compressed_dim=16).to(device)
        regs = torch.randn(2, 1, 64, device=device)
        out = pool(regs)
        expected = pool.compress(regs).squeeze(1)
        torch.testing.assert_close(out, expected)

    def test_different_registers_different_output_slices(self, device: torch.device) -> None:
        """Each register contributes a distinct slice of the output."""
        pool = RegisterCompressConcat(num_registers=4, hidden_dim=64, compressed_dim=16).to(device)
        regs = torch.randn(2, 4, 64, device=device)
        out = pool(regs)
        # Each register's contribution is a contiguous slice of compressed_dim
        for i in range(4):
            expected_slice = pool.compress(regs[:, i, :])
            torch.testing.assert_close(out[:, i * 16 : (i + 1) * 16], expected_slice)

    def test_shared_linear(self) -> None:
        """A single ``compress`` linear is shared across all registers (weight only, no bias)."""
        pool = RegisterCompressConcat(num_registers=14, hidden_dim=384, compressed_dim=32)
        linear_params = list(pool.parameters())
        assert len(linear_params) == 1

    def test_flop_count(self) -> None:
        """FLOP count matches R * 2 * hidden_dim * compressed_dim."""
        pool = RegisterCompressConcat(num_registers=14, hidden_dim=384, compressed_dim=32)
        expected = 14 * 2 * 384 * 32
        assert pool.flop_count(384) == expected


# ─── 2. KernelFiLMGenerator Tests ───────────────────────────────────────────


class TestKernelFiLMGenerator:
    """Tests for :class:`KernelFiLMGenerator` — MLP that produces per-layer FiLM params.

    Maps ``[B, cond_dim]`` conditioning to a list of ``(gamma, beta)`` tuples
    (one per SIREN hidden layer).  At initialization, ``gamma=1, beta=0``
    (identity modulation) thanks to zero-initialized output weights.  The
    output-layer bias (encoding the identity point) is always excluded from
    weight decay.
    """

    def test_output_structure(self, device: torch.device) -> None:
        """Returns a list of ``(gamma, beta)`` tuples, one per ``num_film_layers``."""
        gen = KernelFiLMGenerator(cond_dim=384, kernel_hidden_dim=32, num_film_layers=3).to(device)
        conditioning = torch.randn(2, 384, device=device)
        film_params = gen(conditioning)
        assert len(film_params) == 3
        for gamma, beta in film_params:
            assert gamma.shape == (2, 32)
            assert beta.shape == (2, 32)

    def test_identity_init(self, device: torch.device) -> None:
        """At initialization, gamma=1 and beta=0 for any conditioning input."""
        gen = KernelFiLMGenerator(cond_dim=64, kernel_hidden_dim=16, num_film_layers=2).to(device)
        conditioning = torch.randn(4, 64, device=device)
        film_params = gen(conditioning)
        for gamma, beta in film_params:
            torch.testing.assert_close(gamma, torch.ones_like(gamma), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(beta, torch.zeros_like(beta), atol=1e-6, rtol=1e-5)

    def test_zero_init_output_weights(self) -> None:
        """Final linear layer weights are zero-initialized."""
        gen = KernelFiLMGenerator(cond_dim=64, kernel_hidden_dim=16, num_film_layers=2)
        final_linear = gen.mlp[-1]
        torch.testing.assert_close(final_linear.weight, torch.zeros_like(final_linear.weight))

    def test_bias_init_gamma_one_beta_zero(self) -> None:
        """Final linear bias encodes ``(gamma=1, beta=0)`` for each film layer."""
        gen = KernelFiLMGenerator(cond_dim=64, kernel_hidden_dim=8, num_film_layers=3)
        bias = gen.mlp[-1].bias.data
        for i in range(3):
            offset = i * 2 * 8
            gamma_bias = bias[offset : offset + 8]
            beta_bias = bias[offset + 8 : offset + 16]
            torch.testing.assert_close(gamma_bias, torch.ones(8))
            torch.testing.assert_close(beta_bias, torch.zeros(8))

    def test_gradient_flow(self, device: torch.device) -> None:
        """Gradients reach the conditioning input and all generator parameters."""
        gen = KernelFiLMGenerator(cond_dim=64, kernel_hidden_dim=16, num_film_layers=2).to(device)
        conditioning = torch.randn(2, 64, device=device, requires_grad=True)
        film_params = gen(conditioning)
        loss = sum(g.sum() + b.sum() for g, b in film_params)
        loss.backward()
        assert conditioning.grad is not None
        for p in gen.parameters():
            assert p.grad is not None

    def test_single_film_layer(self, device: torch.device) -> None:
        """Works correctly with ``num_film_layers=1``."""
        gen = KernelFiLMGenerator(cond_dim=64, kernel_hidden_dim=8, num_film_layers=1).to(device)
        conditioning = torch.randn(2, 64, device=device)
        film_params = gen(conditioning)
        assert len(film_params) == 1
        gamma, _beta = film_params[0]
        assert gamma.shape == (2, 8)

    def test_bottleneck_dim(self) -> None:
        """``film_hidden_dim`` controls the MLP bottleneck width."""
        gen = KernelFiLMGenerator(cond_dim=384, kernel_hidden_dim=32, num_film_layers=2, film_hidden_dim=128)
        assert gen.mlp[0].out_features == 128
        assert gen.mlp[0].in_features == 384

    def test_different_inputs_different_outputs(self, device: torch.device) -> None:
        """After a gradient step (breaking zero-init symmetry), different
        conditioning vectors produce different FiLM parameters."""
        gen = KernelFiLMGenerator(cond_dim=32, kernel_hidden_dim=8, num_film_layers=1).to(device)

        # Take a gradient step to break zero-init symmetry
        opt = torch.optim.SGD(gen.parameters(), lr=0.1)
        c = torch.randn(1, 32, device=device)
        params = gen(c)
        (params[0][0].sum() + params[0][1].sum()).backward()
        opt.step()

        c1 = torch.randn(1, 32, device=device)
        c2 = torch.randn(1, 32, device=device)
        p1 = gen(c1)
        p2 = gen(c2)
        assert not torch.allclose(p1[0][0], p2[0][0], atol=1e-6)

    @pytest.mark.parametrize("no_weight_decay", [False, True, 1e-3])
    def test_all_biases_always_no_weight_decay(self, no_weight_decay: bool | float) -> None:
        """Every bias in the MLP has ``_no_weight_decay=True`` regardless of setting."""
        gen = KernelFiLMGenerator(
            cond_dim=64,
            kernel_hidden_dim=8,
            num_film_layers=2,
            no_weight_decay=no_weight_decay,
        )
        for module in gen.mlp.modules():
            if hasattr(module, "bias") and module.bias is not None:
                assert getattr(module.bias, "_no_weight_decay", False) is True, (
                    f"bias of {module} missing _no_weight_decay"
                )

    @pytest.mark.parametrize("no_weight_decay", [False, True, 1e-3])
    def test_biases_no_custom_weight_decay(self, no_weight_decay: bool | float) -> None:
        """No bias has a ``_weight_decay`` attribute (even with float WD)."""
        gen = KernelFiLMGenerator(
            cond_dim=64,
            kernel_hidden_dim=8,
            num_film_layers=2,
            no_weight_decay=no_weight_decay,
        )
        for module in gen.mlp.modules():
            if hasattr(module, "bias") and module.bias is not None:
                assert not hasattr(module.bias, "_weight_decay"), f"bias of {module} should not have _weight_decay"


# ─── 3. ViT5ResidualBlock with FiLM conditioning ────────────────────────────


class TestResidualBlockFiLM:
    """Tests for :class:`ViT5ResidualBlock` with register-based FiLM conditioning.

    When ``register_pooling_cfg`` is provided, the block extracts register
    tokens from the normalized input, pools them via :class:`RegisterPooling`,
    and threads the result as ``conditioning`` through the sequence mixer.
    """

    def _make_block_with_pooling(
        self,
        device: torch.device,
        hidden_dim: int = 384,
        num_registers: int = 4,
    ) -> ViT5ResidualBlock:
        """Build a residual block with register pooling enabled."""
        block = ViT5ResidualBlock(
            sequence_mixer_cfg=LazyConfig(_KwargsPassthroughMixer)(),
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
            register_pooling_cfg=LazyConfig(RegisterPooling)(num_registers=num_registers),
            num_registers=num_registers,
        )
        return block.to(device)

    def _make_block_without_pooling(
        self,
        device: torch.device,
        hidden_dim: int = 384,
    ) -> ViT5ResidualBlock:
        """Build a residual block without register pooling (default)."""
        block = ViT5ResidualBlock(
            sequence_mixer_cfg=LazyConfig(_KwargsPassthroughMixer)(),
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
        )
        return block.to(device)

    def test_register_pooling_created(self, device: torch.device) -> None:
        """``register_pooling`` is a ``RegisterPooling`` when configured."""
        block = self._make_block_with_pooling(device)
        assert isinstance(block.register_pooling, RegisterPooling)

    def test_no_register_pooling_by_default(self, device: torch.device) -> None:
        """``register_pooling`` is ``None`` when no pooling config is given."""
        block = self._make_block_without_pooling(device)
        assert block.register_pooling is None

    def test_output_shape_with_pooling(self, device: torch.device) -> None:
        """Block with register pooling preserves sequence length ``[B, T, C]``."""
        block = self._make_block_with_pooling(device, num_registers=4)
        T = 1 + 4 + 196  # [CLS, regs, patches]
        x = torch.randn(2, T, 384, device=device)
        out = block(x)
        assert out.shape == (2, T, 384)

    def test_gradient_flow_through_pooling(self, device: torch.device) -> None:
        """Gradients flow from the output through register pooling logits."""
        block = self._make_block_with_pooling(device, num_registers=4)
        T = 1 + 4 + 196
        x = torch.randn(2, T, 384, device=device, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert block.register_pooling.logits.grad is not None

    def test_pooling_not_created_with_zero_registers(self, device: torch.device) -> None:
        """``register_pooling_cfg`` is ignored when ``num_registers=0``."""
        block = ViT5ResidualBlock(
            sequence_mixer_cfg=LazyConfig(_KwargsPassthroughMixer)(),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=384, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=384,
                activation="gelu",
                expansion_factor=4.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=384, eps=1e-6),
            hidden_dim=384,
            register_pooling_cfg=LazyConfig(RegisterPooling)(num_registers=4),
            num_registers=0,
        ).to(device)
        assert block.register_pooling is None

    def test_register_extraction_positions(self, device: torch.device) -> None:
        """Register pooling extracts from positions ``[1, 1+num_registers)``
        assuming ``[CLS, regs, patches]`` layout."""
        block = self._make_block_with_pooling(device, num_registers=4)
        T = 1 + 4 + 196
        x = torch.randn(2, T, 384, device=device)

        x_normed = block.input_norm(x)
        expected_regs = x_normed[:, 1:5, :]  # positions 1..4
        pooled = block.register_pooling(expected_regs)
        assert pooled.shape == (2, 384)


# ─── 4. ViT5HyenaAdapter Tests ──────────────────────────────────────────────


class TestViT5HyenaAdapter:
    """Tests for :class:`ViT5HyenaAdapter` — bridges ``[B, T, C]`` and ``[B, H, W, C]``.

    The adapter reshapes the flat token sequence into a 2D spatial grid,
    applies the inner mixer, and reshapes back.  All mixer kwargs
    (e.g. ``conditioning``) are forwarded transparently.
    """

    def _make_passthrough_adapter(self, device: torch.device, grid_w: int = 15) -> ViT5HyenaAdapter:
        """Build an adapter with ``nn.Identity`` as the inner mixer for shape testing."""
        adapter = ViT5HyenaAdapter(
            inner_mixer_cfg=LazyConfig(nn.Identity)(),
            grid_w=grid_w,
        )
        return adapter.to(device)

    def test_output_shape(self, device: torch.device) -> None:
        """Output shape ``[B, T, C]`` matches input."""
        adapter = self._make_passthrough_adapter(device, grid_w=15)
        T = 15 * 14  # 210 = grid_w * grid_h
        x = torch.randn(2, T, 64, device=device)
        out = adapter(x)
        assert out.shape == (2, T, 64)

    def test_reshape_round_trip(self, device: torch.device) -> None:
        """With identity mixer, output equals input (reshape is lossless)."""
        adapter = self._make_passthrough_adapter(device, grid_w=15)
        T = 15 * 14
        x = torch.randn(2, T, 64, device=device)
        out = adapter(x)
        torch.testing.assert_close(out, x)

    def test_kwargs_forwarded(self, device: torch.device) -> None:
        """``mixer_kwargs`` (e.g. ``conditioning``) are forwarded to the inner mixer."""
        adapter = ViT5HyenaAdapter(
            inner_mixer_cfg=LazyConfig(nn.Identity)(),
            grid_w=15,
        ).to(device)
        # Replace inner mixer with a capture mock (bypass LazyConfig resolution)
        capture = _KwargCaptureMixer().to(device)
        adapter.inner_mixer = capture

        T = 15 * 14
        x = torch.randn(2, T, 64, device=device)
        cond = torch.randn(2, 64, device=device)
        adapter(x, conditioning=cond)

        assert "conditioning" in capture.last_kwargs
        torch.testing.assert_close(capture.last_kwargs["conditioning"], cond)

    def test_grid_w_stored(self, device: torch.device) -> None:
        """``grid_w`` constructor argument is stored as an attribute."""
        adapter = self._make_passthrough_adapter(device, grid_w=15)
        assert adapter.grid_w == 15

    def test_gradient_flow(self, device: torch.device) -> None:
        """Gradients propagate from output back to input through the reshape."""
        adapter = self._make_passthrough_adapter(device, grid_w=15)
        T = 15 * 14
        x = torch.randn(2, T, 64, device=device, requires_grad=True)
        out = adapter(x)
        out.sum().backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
