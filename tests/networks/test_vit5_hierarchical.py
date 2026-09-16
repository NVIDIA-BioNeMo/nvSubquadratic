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

# David W. Romero, 2025-09-09

"""Tests for ViT5HierarchicalNet.

Validates:
1. End-to-end forward pass: (B, 224, 224, 3) → (B, 1000)
2. Spatial shape progression: 56×56 → 28×28 → 14×14 → 7×7
3. Channel doubling at each merge: C1 → 2C1 → 4C1 → 8C1
4. No CLS token, pos_embed, or register tokens
5. DropPath rate injection (linear ramp)
6. flop_count returns a positive integer
7. Gradient flows end-to-end
8. Batch-size invariance
9. Config-level smoke test: build real Hyena net from _base_config.py (torch_fft backend)
   and verify forward pass on a tiny image with the full Hyena stack.
10. BlockDiag overrides: verify kernel/mask type after apply_block_diag_overrides()

Tests use a tiny synthetic model (no SIREN kernels, no DALI, CPU-only)
built from real ``ViT5ResidualBlock`` + ``nn.Identity`` mixers so we test
the *wiring* without needing GPU or the heavy kernel infrastructure.
"""

from itertools import pairwise

import pytest
import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_hierarchical import (
    StageSpec,
    ViT5HierarchicalNet,
    _compute_drop_path_rates,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def test_spatial_norm_decay_matches_vit5_classification():
    from experiments.lightning_wrappers.base_lightning_wrapper import _build_param_groups
    from nvsubquadratic.lazy_config import instantiate
    from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet

    # Use the runner's lazy-instantiation path, with no blocks to isolate stem,
    # mergers and classifier from the existing mixers' regularization policies.
    spatial = instantiate(
        LazyConfig(ViT5HierarchicalNet)(
            in_channels=3,
            num_classes=10,
            stage_specs=[StageSpec(0, d, LazyConfig(nn.Identity)()) for d in (8, 16, 32, 64)],
        )
    )
    flat = instantiate(
        LazyConfig(ViT5ClassificationNet)(
            in_channels=3,
            num_classes=10,
            hidden_dim=8,
            num_blocks=0,
            patch_size=4,
            image_size=32,
            num_registers=0,
            norm_cfg=LazyConfig(nn.LayerNorm)(normalized_shape=8),
            readout="gap",
            block_cfg=LazyConfig(nn.Identity)(),
        )
    )
    for net in (spatial, flat):
        groups = _build_param_groups(net, default_weight_decay=0.05)
        decay = {id(p): group["weight_decay"] for group in groups for p in group["params"]}
        assert len(decay) == len(list(net.parameters()))
        for module in net.modules():
            if isinstance(module, nn.LayerNorm):
                assert all(decay[id(p)] == 0 for p in module.parameters())
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                assert decay[id(module.weight)] == 0.05
                if module.bias is not None:
                    assert decay[id(module.bias)] == 0


class _IdentityMixer(nn.Module):
    """Trivial sequence mixer: returns input unchanged.  For shape-only tests."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x

    def flop_count(self, num_tokens, **kwargs) -> int:
        return 0


def _make_stage_spec(dim: int, num_blocks: int) -> StageSpec:
    """Build a StageSpec with ViT5ResidualBlock + identity mixer for shape tests."""
    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(_IdentityMixer)(dim=dim),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=dim, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=dim,
            activation="gelu",
            expansion_factor=4.0,
            bias=False,
            dropout_cfg=LazyConfig(nn.Dropout)(p=0.0),
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=dim, eps=1e-6),
        hidden_dim=dim,
        layer_scale_init=1e-4,
    )
    return StageSpec(num_blocks=num_blocks, hidden_dim=dim, block_cfg=block_cfg)


def _make_tiny_net(
    base_dim: int = 16,
    depths: list[int] | None = None,
    num_classes: int = 10,
    max_drop_path_rate: float = 0.1,
) -> ViT5HierarchicalNet:
    """Build a tiny 4-stage hierarchical net for CPU tests."""
    if depths is None:
        depths = [2, 2, 2, 2]
    dims = [base_dim * (2**i) for i in range(4)]  # [16, 32, 64, 128]
    stage_specs = [_make_stage_spec(dim, n) for dim, n in zip(dims, depths)]
    return ViT5HierarchicalNet(
        in_channels=3,
        num_classes=num_classes,
        stage_specs=stage_specs,
        max_drop_path_rate=max_drop_path_rate,
    )


# ── _compute_drop_path_rates ──────────────────────────────────────────────────


class TestDropPathRates:
    def test_linear_ramp(self):
        rates = _compute_drop_path_rates(0.2, [2, 2, 6, 2])
        flat = [r for stage in rates for r in stage]
        assert len(flat) == 12
        assert flat[0] == pytest.approx(0.0, abs=1e-6)
        assert flat[-1] == pytest.approx(0.2, abs=1e-6)
        # Monotonically non-decreasing
        for a, b in pairwise(flat):
            assert b >= a - 1e-9

    def test_constant_schedule(self):
        rates = _compute_drop_path_rates(0.1, [2, 2, 6, 2], schedule="constant")
        for stage in rates:
            for r in stage:
                assert r == pytest.approx(0.1, abs=1e-6)

    def test_stage_grouping(self):
        rates = _compute_drop_path_rates(0.2, [2, 2, 6, 2])
        assert len(rates) == 4
        assert len(rates[0]) == 2
        assert len(rates[1]) == 2
        assert len(rates[2]) == 6
        assert len(rates[3]) == 2

    def test_zero_max_rate(self):
        rates = _compute_drop_path_rates(0.0, [2, 2, 6, 2])
        for stage in rates:
            for r in stage:
                assert r == pytest.approx(0.0, abs=1e-6)


# ── ViT5HierarchicalNet ────────────────────────────────────────────────────────


class TestViT5HierarchicalNet:
    def test_forward_shape(self):
        net = _make_tiny_net(base_dim=16, depths=[1, 1, 1, 1], num_classes=10)
        net.eval()
        x = torch.randn(2, 224, 224, 3)
        out = net({"input": x, "condition": None})
        assert "logits" in out
        assert out["logits"].shape == (2, 10)

    def test_spatial_progression_56_7(self):
        """Verify 56×56 → 28×28 → 14×14 → 7×7 spatial schedule."""
        net = _make_tiny_net(base_dim=8, depths=[1, 1, 1, 1], num_classes=4)
        net.eval()

        activations = {}

        def hook_factory(name):
            def hook(module, input, output):
                activations[name] = output.shape

            return hook

        net.stem.register_forward_hook(hook_factory("stem"))
        for i, stage in enumerate(net.stages):
            stage[-1].register_forward_hook(hook_factory(f"stage{i}"))
        for i, ds in enumerate(net.downsamplers):
            ds.register_forward_hook(hook_factory(f"down{i}"))

        x = torch.randn(1, 224, 224, 3)
        with torch.no_grad():
            net({"input": x, "condition": None})

        assert activations["stem"][1:3] == (56, 56)
        assert activations["stage0"][1:3] == (56, 56)
        assert activations["down0"][1:3] == (28, 28)
        assert activations["stage1"][1:3] == (28, 28)
        assert activations["down1"][1:3] == (14, 14)
        assert activations["stage2"][1:3] == (14, 14)
        assert activations["down2"][1:3] == (7, 7)
        assert activations["stage3"][1:3] == (7, 7)

    def test_channel_doubling(self):
        """Channels double at each merge: C1 → 2C1 → 4C1 → 8C1."""
        base_dim = 8
        net = _make_tiny_net(base_dim=base_dim, depths=[1, 1, 1, 1], num_classes=4)
        net.eval()

        activations = {}

        def hook_factory(name):
            def hook(module, input, output):
                activations[name] = output.shape[-1]  # last dim = channels

            return hook

        net.stem.register_forward_hook(hook_factory("stem"))
        for i, ds in enumerate(net.downsamplers):
            ds.register_forward_hook(hook_factory(f"down{i}"))

        x = torch.randn(1, 224, 224, 3)
        with torch.no_grad():
            net({"input": x, "condition": None})

        assert activations["stem"] == base_dim
        assert activations["down0"] == base_dim * 2
        assert activations["down1"] == base_dim * 4
        assert activations["down2"] == base_dim * 8

    def test_no_cls_pos_reg_tokens(self):
        """Hierarchical net must not carry CLS / pos_embed / register tokens."""
        net = _make_tiny_net()
        assert not hasattr(net, "cls_token"), "cls_token should not exist"
        assert not hasattr(net, "pos_embed"), "pos_embed should not exist"
        assert not hasattr(net, "reg_token"), "reg_token should not exist"

    def test_gradient_flow(self):
        net = _make_tiny_net(base_dim=8, depths=[1, 1, 1, 1], num_classes=4)
        x = torch.randn(1, 32, 32, 3)  # smaller image for speed
        out = net({"input": x, "condition": None})
        out["logits"].sum().backward()
        # Check stem gradient
        assert net.stem.proj.weight.grad is not None
        assert net.stem.proj.weight.grad.abs().sum() > 0
        # Check head gradient
        assert net.head.weight.grad is not None

    def test_batch_size_invariant(self):
        net = _make_tiny_net(base_dim=8, depths=[1, 1, 1, 1], num_classes=4)
        net.eval()
        with torch.no_grad():
            for B in [1, 2, 4]:
                out = net({"input": torch.randn(B, 32, 32, 3), "condition": None})
                assert out["logits"].shape == (B, 4)

    def test_requires_4_stages(self):
        specs = [_make_stage_spec(16, 1)] * 3  # only 3 stages
        with pytest.raises(ValueError, match="4 stages"):
            ViT5HierarchicalNet(in_channels=3, num_classes=10, stage_specs=specs)

    def test_flop_count_positive(self):
        net = _make_tiny_net(base_dim=16, depths=[2, 2, 2, 2], num_classes=10)
        flops = net.flop_count(image_size=32)
        assert flops > 0

    def test_drop_path_rate_injection(self):
        """First block should have drop_path_rate=0, last should be max."""
        net = _make_tiny_net(base_dim=8, depths=[2, 2, 2, 2], num_classes=4, max_drop_path_rate=0.2)
        from nvsubquadratic.modules.drop_path import DropPath

        # Collect all drop_path rates from all blocks
        rates = []
        for stage in net.stages:
            for block in stage:
                dp = block.drop_path
                if isinstance(dp, DropPath):
                    rates.append(dp.drop_prob)
                else:
                    rates.append(0.0)

        assert rates[0] == pytest.approx(0.0, abs=1e-6)
        assert rates[-1] == pytest.approx(0.2, abs=1e-6)

    def test_eval_train_modes(self):
        """Forward pass works in both train and eval modes."""
        net = _make_tiny_net(base_dim=8, depths=[1, 1, 1, 1], num_classes=4)
        x = {"input": torch.randn(1, 32, 32, 3), "condition": None}
        net.train()
        out_train = net(x)
        assert out_train["logits"].shape == (1, 4)
        net.eval()
        with torch.no_grad():
            out_eval = net(x)
        assert out_eval["logits"].shape == (1, 4)


# ── Config-level smoke tests (real Hyena stack, torch_fft backend) ────────────


def _build_real_hyena_net(base_dim: int = 8, stage_depths: list[int] | None = None, num_classes: int = 4):
    """Build a tiny hierarchical Hyena net from _base_config, using torch_fft for local runs."""
    from examples.vit5_imagenet.v5_patchmerge._base_config import (
        build_hierarchical_net,
    )

    if stage_depths is None:
        stage_depths = [1, 1, 1, 1]

    net = build_hierarchical_net(
        base_dim=base_dim,
        stage_depths=stage_depths,
        num_classes=num_classes,
        fft_backend="torch_fft",
    )
    return net


class TestRealHyenaSmoke:
    """Smoke tests that instantiate real Hyena blocks from _base_config.py.

    Uses torch_fft backend (which works with arbitrary kernel/input sizes)
    instead of subq_ops (requires subq_ops >= 0.2.0 on GPU cluster).
    """

    def test_forward_shape_real_hyena(self):
        """End-to-end with real Hyena blocks: (1, 32, 32, 3) → (1, 4)."""
        net = _build_real_hyena_net(base_dim=8, num_classes=4)
        net.eval()
        with torch.no_grad():
            out = net({"input": torch.randn(1, 32, 32, 3), "condition": None})
        assert out["logits"].shape == (1, 4)

    def test_channel_schedule_real_hyena(self):
        """Stage channels double: C1 → 2C1 → 4C1 → 8C1."""
        base_dim = 8
        net = _build_real_hyena_net(base_dim=base_dim)
        net.eval()

        activations = {}

        def hook_factory(name):
            def hook(m, inp, out):
                activations[name] = out.shape[-1]

            return hook

        net.stem.register_forward_hook(hook_factory("stem"))
        for i, ds in enumerate(net.downsamplers):
            ds.register_forward_hook(hook_factory(f"down{i}"))

        with torch.no_grad():
            net({"input": torch.randn(1, 32, 32, 3), "condition": None})

        assert activations["stem"] == base_dim
        assert activations["down0"] == base_dim * 2
        assert activations["down1"] == base_dim * 4
        assert activations["down2"] == base_dim * 8

    def test_no_rope_in_hyena_blocks(self):
        """Hyena blocks must have use_rope=False (no RoPE in hierarchical setup)."""
        from nvsubquadratic.modules.hyena_nd import Hyena

        net = _build_real_hyena_net(base_dim=8)
        for module in net.modules():
            if isinstance(module, Hyena):
                assert not any("rope" in type(child).__name__.lower() for child in module.modules())

    def test_grn_present_in_all_hyena_blocks(self):
        """Every Hyena residual block should have a GRN module."""

        net = _build_real_hyena_net(base_dim=8)
        for stage in net.stages:
            for block in stage:
                assert block.grn is not None, "Every Hyena block should have GRN"
                assert isinstance(block.grn, GlobalResponseNorm)

    def test_qk_norm_disabled(self):
        """QK-norm must be None/disabled for Hyena (no attention)."""
        from nvsubquadratic.modules.hyena_nd import Hyena

        net = _build_real_hyena_net(base_dim=8)
        for module in net.modules():
            if isinstance(module, Hyena):
                assert module.q_norm is None, "q_norm should be None (no QK-norm for Hyena)"


class TestBlockDiagOverrides:
    """Verify apply_block_diag_overrides swaps kernels and masks correctly."""

    def test_kernel_type_after_override(self):
        """Kernels become BlockDiagonalMultiOmegaSIRENKernelND after override."""
        from examples.vit5_imagenet.v5_patchmerge._base_config import build_hierarchical_net
        from examples.vit5_imagenet.v5_patchmerge._blockdiag import apply_block_diag_overrides
        from nvsubquadratic.modules.kernels_nd import BlockDiagonalMultiOmegaSIRENKernelND

        net = build_hierarchical_net(base_dim=8, stage_depths=[1, 1, 1, 1], fft_backend="torch_fft")
        apply_block_diag_overrides(net)

        for stage in net.stages:
            for block in stage:
                gconv = block.sequence_mixer.mixer.global_conv
                assert isinstance(gconv.kernel, BlockDiagonalMultiOmegaSIRENKernelND), (
                    f"Expected BlockDiagonalMultiOmegaSIRENKernelND, got {type(gconv.kernel).__name__}"
                )

    def test_mask_type_after_override(self):
        """Masks become BlockAlignedGaussianModulationND after override."""
        from examples.vit5_imagenet.v5_patchmerge._base_config import build_hierarchical_net
        from examples.vit5_imagenet.v5_patchmerge._blockdiag import apply_block_diag_overrides
        from nvsubquadratic.modules.masks_nd import BlockAlignedGaussianModulationND

        net = build_hierarchical_net(base_dim=8, stage_depths=[1, 1, 1, 1], fft_backend="torch_fft")
        apply_block_diag_overrides(net)

        for stage in net.stages:
            for block in stage:
                gconv = block.sequence_mixer.mixer.global_conv
                assert isinstance(gconv.mask, BlockAlignedGaussianModulationND), (
                    f"Expected BlockAlignedGaussianModulationND, got {type(gconv.mask).__name__}"
                )

    def test_omega_scaling_per_stage(self):
        """ω₀_max at stage 0 should be 4× that at stage 3 (56/7 = 8, but from ref 14: 4× and 0.5×)."""
        from examples.vit5_imagenet.v5_patchmerge._base_config import build_hierarchical_net
        from examples.vit5_imagenet.v5_patchmerge._blockdiag import (
            _REF_HEIGHT,
            OMEGA_0_MAX_BASE,
            apply_block_diag_overrides,
        )
        from nvsubquadratic.modules.kernels_nd import BlockDiagonalMultiOmegaSIRENKernelND

        net = build_hierarchical_net(base_dim=8, stage_depths=[1, 1, 1, 1], fft_backend="torch_fft")
        apply_block_diag_overrides(net)

        from examples.vit5_imagenet.v5_patchmerge._base_config import STAGE_HEIGHTS

        for stage_idx, (stage_blocks, stage_h) in enumerate(zip(net.stages, STAGE_HEIGHTS)):
            expected_max = OMEGA_0_MAX_BASE * (stage_h / _REF_HEIGHT)
            kernel = stage_blocks[0].sequence_mixer.mixer.global_conv.kernel
            assert isinstance(kernel, BlockDiagonalMultiOmegaSIRENKernelND)
            actual_max = float(kernel.omega_0_per_row.max())
            assert actual_max == pytest.approx(expected_max, rel=0.05), (
                f"Stage {stage_idx} (h={stage_h}): expected ω₀_max≈{expected_max}, got {actual_max}"
            )


@pytest.mark.parametrize("block_diag", [False, True])
def test_lazy_hierarchy_seed_logging_and_optimizer_policy(block_diag):
    import json
    import warnings

    from examples.vit5_imagenet.v5_patchmerge._base_config import get_hierarchical_net_config
    from examples.vit5_imagenet.v5_patchmerge._blockdiag import apply_block_diag_config_overrides
    from experiments.lightning_wrappers.base_lightning_wrapper import _build_param_groups
    from experiments.utils.cli import config_to_dict
    from nvsubquadratic.lazy_config import instantiate

    rng_before = torch.get_rng_state().clone()
    cfg = get_hierarchical_net_config(base_dim=8, stage_depths=[1, 1, 1, 1], fft_backend="torch_fft", num_classes=4)
    if block_diag:
        apply_block_diag_config_overrides(cfg)
    assert torch.equal(rng_before, torch.get_rng_state())
    serialized = json.loads(json.dumps(config_to_dict(cfg)))
    assert serialized["stage_specs"][0]["hidden_dim"] == 8
    kernel = serialized["stage_specs"][0]["block_cfg"]["sequence_mixer_cfg"]["mixer_cfg"]["global_conv_cfg"][
        "kernel_cfg"
    ]
    assert kernel["L_cache"] == 56
    assert ("BlockDiagonal" in kernel["__target__"]) == block_diag

    torch.manual_seed(42)
    first = instantiate(cfg)
    torch.manual_seed(42)
    repeated = instantiate(cfg)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)
    torch.manual_seed(43)
    different = instantiate(cfg)
    assert not torch.equal(first.stem.proj.weight, different.stem.proj.weight)
    with warnings.catch_warnings():
        # Existing GRN/shortcut warnings are unrelated to the stem/merger policy.
        warnings.simplefilter("ignore", UserWarning)
        groups = _build_param_groups(first, default_weight_decay=0.05)
    decay = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    for module in first.modules():
        if isinstance(module, nn.LayerNorm):
            assert all(decay[id(p)] == 0 for p in module.parameters())
    assert decay[id(first.head.bias)] == 0
    assert decay[id(first.head.weight)] == 0.05
    first.eval()
    first({"input": torch.randn(1, 32, 32, 3)})["logits"].sum().backward()
    assert first.stem.proj.weight.grad is not None
    assert first.head.weight.grad is not None


@pytest.mark.parametrize(
    "recipe,base_helper",
    [
        ("examples.patch_merging.cifar10.hierarchical_hyena", None),
        ("examples.vit5_imagenet.local_comparison.hierarchical_hyena", "get_local_base_config"),
        ("examples.vit5_imagenet.v5_patchmerge.hierarchical_full_hyena", "get_base_config"),
        ("examples.vit5_imagenet.v5_patchmerge.hierarchical_full_hyena_blockdiag", "get_base_config"),
    ],
)
def test_hierarchical_leaf_recipes_defer_model_creation(monkeypatch, recipe, base_helper):
    from importlib import import_module

    from omegaconf import DictConfig

    from experiments.default_cfg import ExperimentConfig
    from experiments.utils.cli import apply_config_overrides, config_to_dict

    module = import_module(recipe)
    if base_helper:
        # Only replace the unrelated Apex/DALI training recipe dependency.
        monkeypatch.setattr(module, base_helper, lambda **kwargs: ExperimentConfig())
    before = torch.get_rng_state().clone()
    config = module.get_config()
    assert torch.equal(before, torch.get_rng_state())
    assert isinstance(config.net, DictConfig)
    assert isinstance(config_to_dict(config)["net"]["stage_specs"], list)
    config = apply_config_overrides(config, ["net.num_classes=7"])
    assert config.net.num_classes == 7


@pytest.mark.parametrize("schedule,expected", [("linear", 0.0), ("constant", 0.2)])
def test_single_block_drop_path_schedule(schedule, expected):
    rates = _compute_drop_path_rates(0.2, [0, 1, 0, 0], schedule)
    assert rates == [[], [expected], [], []]
    assert _compute_drop_path_rates(0.2, [0, 0, 0, 0], schedule) == [[], [], [], []]
    net = ViT5HierarchicalNet(
        in_channels=3,
        num_classes=4,
        stage_specs=[_make_stage_spec(dim, depth) for dim, depth in zip([8, 16, 32, 64], [0, 1, 0, 0])],
        max_drop_path_rate=0.2,
        drop_path_schedule=schedule,
    )
    assert getattr(net.stages[1][0].drop_path, "drop_prob", 0.0) == expected


def test_stage_count_validation_survives_optimized_python():
    import subprocess
    import sys

    code = """
from examples.vit5_imagenet.v5_patchmerge._base_config import get_hierarchical_net_config
from nvsubquadratic.networks.vit5_hierarchical import ViT5HierarchicalNet
for call in (
    lambda: get_hierarchical_net_config(stage_depths=[1, 1, 1]),
    lambda: ViT5HierarchicalNet(in_channels=3, num_classes=4, stage_specs=[]),
):
    try:
        call()
    except ValueError as error:
        if 'Expected 4' not in str(error):
            raise
    else:
        raise RuntimeError('Invalid stage count was accepted under python -O')
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("compiled_source", [False, True])
def test_pretrained_hierarchy_can_replace_classifier(compiled_source):
    from experiments.utils.checkpointing import DropKeysFromCheckpoint, StripCompiledPrefix
    from nvsubquadratic.lazy_config import instantiate

    source = nn.Module()
    source.network = _make_tiny_net(base_dim=8, depths=[0, 0, 0, 0], num_classes=4)
    target = nn.Module()
    target.network = _make_tiny_net(base_dim=8, depths=[0, 0, 0, 0], num_classes=7)
    initial_head = target.network.head.weight.detach().clone()
    state = source.state_dict()
    assert "network.head.weight" in state
    assert "network.out_proj.weight" not in state
    if compiled_source:
        state = {key.replace("network.", "network._orig_mod."): value for key, value in state.items()}
    # Exercise the exact callback sequence documented in examples/patch_merging.
    for cfg in [
        LazyConfig(StripCompiledPrefix)(),
        LazyConfig(DropKeysFromCheckpoint)(prefixes=("network.head",)),
    ]:
        state = instantiate(cfg)(state, model=target)
    missing, unexpected = target.load_state_dict(state, strict=False)
    assert set(missing) == {"network.head.weight", "network.head.bias"}
    assert unexpected == []
    torch.testing.assert_close(target.network.head.weight, initial_head)
    torch.testing.assert_close(target.network.stem.proj.weight, source.network.stem.proj.weight)
    assert target.network({"input": torch.randn(1, 32, 32, 3)})["logits"].shape == (1, 7)
