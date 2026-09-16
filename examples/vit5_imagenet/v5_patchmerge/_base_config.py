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


"""Shared base config for hierarchical Hyena (Swin/VMamba-style) experiments.

Builds a ``ViT5HierarchicalNet`` with 4 stages and discrete Patch Merging:

    Stem (4×4 conv) → Stage1 (N1 × H) → PatchMerging
                    → Stage2 (N2 × H) → PatchMerging
                    → Stage3 (N3 × H) → PatchMerging
                    → Stage4 (N4 × H) → LayerNorm → GAP → Linear

Architecture follows Swin Transformer (Liu et al., 2021) and VMamba (Liu et al., 2024).

Key design choices vs. the isotropic vit5_hybrid configs:
  - Pure Hyena (no attention blocks) — enables a RoPE-free setup.
  - No CLS token, register tokens, or absolute positional embeddings.
  - Global Average Pooling readout (matching Swin/VMamba).
  - Per-stage hidden dim, L_cache, and ω₀ — no OmegaConf ${eval:...} interpolations
    since each stage has different spatial size and channel width.
  - ω₀ scaled proportionally with spatial resolution to preserve Nyquist coverage:
        ω₀_stage = ω₀_base × (stage_h / 14)
    Reference: _blockdiag.py header note ("scale by m when resolution changes by m").
  - GRN (Global Response Normalization) on all Hyena blocks, matching the
    isotropic full-hyena configs.

Typical usage::

    from examples.vit5_imagenet.v5_patchmerge._base_config import (
        get_hierarchical_net_config, get_base_config
    )
    config = get_base_config()
    config.net = get_hierarchical_net_config()
"""

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_hierarchical import StageSpec, ViT5HierarchicalNet
from nvsubquadratic.utils.init import trunc_normal_init_factory


# Architectural constants are defined locally so this module is importable
# without the apex dependency that v5._base drags in at module load.
# Training-recipe constants (optimizer, scheduler, etc.) live in v5._base
# and are only needed when building the full ExperimentConfig.
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4


def get_base_config(*args, **kwargs):
    """Thin wrapper — defers the apex import to call time for non-training uses."""
    from examples.vit5_imagenet.v5._base import get_base_config as _get

    return _get(*args, **kwargs)


__all__ = ["build_hierarchical_net", "get_base_config", "get_hierarchical_net_config"]


# ── Architecture constants ──────────────────────────────────────────────────

# Swin-T depths; 12 total blocks — matches the isotropic NUM_BLOCKS=12 baseline.
STAGE_DEPTHS = [2, 2, 6, 2]

# Base channel width (Swin-T default: 96).
# C₁=96 → dims [96, 192, 384, 768] → ~28 M params.
# C₁=64 → dims [64, 128, 256, 512] → ~13 M params (use base_dim kwarg for a sweep).
BASE_DIM = 96

# Spatial dimensions per stage for a 224×224 input with 4×4 stem.
STAGE_HEIGHTS = [56, 28, 14, 7]

# ── SIREN kernel hyperparameters ────────────────────────────────────────────

KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_HIDDEN_OMEGA_0 = 1.0

# Reference ω₀ at 14×14 resolution (matching the isotropic vit5_hybrid config).
# For each stage with spatial height h: ω₀_stage = KERNEL_OMEGA_0_BASE × (h / 14).
KERNEL_OMEGA_0_BASE = 10.0
_REF_HEIGHT = 14

DROP_PATH_RATE = 0.1


# ── Init helper ─────────────────────────────────────────────────────────────

INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


# ── Per-stage Hyena block builder ────────────────────────────────────────────


def _make_hyena_block_cfg(
    hidden_dim: int,
    stage_h: int,
    omega_0: float,
    fft_backend: str = "subq_ops",
) -> LazyConfig:
    """Build a ViT5ResidualBlock config for one hierarchical stage.

    All dimensions are concrete Python scalars (no OmegaConf interpolations)
    since each stage has a different spatial size and channel width.

    Args:
        hidden_dim: Channel width for this stage.
        stage_h: Spatial height (= width) at this stage (e.g. 56, 28, 14, 7).
        omega_0: SIREN first-layer frequency for this stage.
        fft_backend: CKConvND backend; use torch_fft for CPU execution.

    Returns:
        LazyConfig for ViT5ResidualBlock.
    """
    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=hidden_dim,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=hidden_dim,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=hidden_dim,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=omega_0,
                    L_cache=stage_h,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(GaussianModulationND)(
                    data_dim=2,
                    num_channels=hidden_dim,
                    min_attenuation_at_step=0.1,
                    max_attenuation_at_limit=0.95,
                    init_extent=1.0,
                    parametrization="direct",
                ),
                grid_type="double",
                fft_padding="zero",
                fft_backend=fft_backend,
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * hidden_dim,
                out_channels=3 * hidden_dim,
                kernel_size=3,
                groups=3 * hidden_dim,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            qk_norm_cfg=None,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=mixer_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=hidden_dim,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hidden_dim, eps=1e-6),
        hidden_dim=hidden_dim,
        layer_scale_init=LAYER_SCALE_INIT,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=hidden_dim),
    )


# ── Network builder ──────────────────────────────────────────────────────────


def get_hierarchical_net_config(
    base_dim: int = BASE_DIM,
    stage_depths: list[int] | None = None,
    max_drop_path_rate: float = DROP_PATH_RATE,
    drop_path_schedule: str = "linear",
    num_classes: int = NUM_CLASSES,
    fft_backend: str = "subq_ops",
) -> LazyConfig:
    """Configure a ViT5HierarchicalNet with pure-Hyena blocks and Swin-style Patch Merging.

    Args:
        base_dim: Channel width of stage 1 (C₁). Doubles at each merge.
            Default 96 (Swin-T). Use 64 for a param-matched ~22 M variant.
        stage_depths: Number of blocks per stage, e.g. [2, 2, 6, 2].
            Default: Swin-T depths.
        max_drop_path_rate: Maximum stochastic depth drop probability.
        drop_path_schedule: ``"linear"`` (ramp 0→max) or ``"constant"``.
        num_classes: Number of output classes.
        fft_backend: CKConvND backend; use torch_fft for CPU execution.

    Returns:
        Lazy network configuration; no parameters are allocated or initialized.
    """
    if stage_depths is None:
        stage_depths = STAGE_DEPTHS

    if len(stage_depths) != 4:
        raise ValueError(f"Expected 4 stage depths, got {len(stage_depths)}")

    stage_specs = []
    for i, (n_blocks, stage_h) in enumerate(zip(stage_depths, STAGE_HEIGHTS)):
        hidden_dim = base_dim * (2**i)
        omega_0 = KERNEL_OMEGA_0_BASE * (stage_h / _REF_HEIGHT)
        block_cfg = _make_hyena_block_cfg(
            hidden_dim=hidden_dim,
            stage_h=stage_h,
            omega_0=omega_0,
            fft_backend=fft_backend,
        )
        stage_specs.append(
            StageSpec(
                num_blocks=n_blocks,
                hidden_dim=hidden_dim,
                block_cfg=block_cfg,
            )
        )

    return LazyConfig(ViT5HierarchicalNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=num_classes,
        stage_specs=stage_specs,
        max_drop_path_rate=max_drop_path_rate,
        drop_path_schedule=drop_path_schedule,
    )


def build_hierarchical_net(
    base_dim: int = BASE_DIM,
    stage_depths: list[int] | None = None,
    max_drop_path_rate: float = DROP_PATH_RATE,
    drop_path_schedule: str = "linear",
    num_classes: int = NUM_CLASSES,
    fft_backend: str = "subq_ops",
) -> ViT5HierarchicalNet:
    """Instantiate a hierarchy for direct use; recipes use get_hierarchical_net_config.

    Arguments match :func:`get_hierarchical_net_config`. Callers of this eager
    convenience API must set their seed before calling it.
    """
    return instantiate(
        get_hierarchical_net_config(
            base_dim=base_dim,
            stage_depths=stage_depths,
            max_drop_path_rate=max_drop_path_rate,
            drop_path_schedule=drop_path_schedule,
            num_classes=num_classes,
            fft_backend=fft_backend,
        )
    )
