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

"""Post-creation override: swap Hyena kernels for the learnable-ω₀ variants.

Importable helpers applied to a config whose ``net`` was built by
:func:`build_hybrid_net` from ``_base_config``.  Two flavors are exposed:

* :func:`apply_learnable_omega_overrides` swaps the per-block Hyena kernel
  for :class:`LearnableOmegaSIRENKernelND`.  Same scalar ω₀ as the baseline,
  but ``2π·ω₀`` is taken out of the first-layer weight init and applied as
  a runtime factor, with an additional learnable per-row scale clamped to
  ``[omega_0_scale_min, omega_0_scale_max]`` (default ``[1e-2, 2]``, so the
  effective per-row ω₀ ranges from ``0.01·ω₀`` to ``2·ω₀`` and no row's
  first-layer sine ever collapses to a constant).  The mask is left as the
  baseline :class:`GaussianModulationND`.

* :func:`apply_learnable_omega_blockdiag_overrides` swaps the kernel for
  :class:`BlockDiagonalLearnableOmegaSIRENKernelND` (block-diagonal MLP
  init + per-block ω₀ schedule, both made learnable in the same way) and
  the mask for :class:`BlockAlignedGaussianModulationND`, mirroring the
  ``_blockdiag.py`` defaults but with the per-row ω₀ now trainable.

Both helpers also propagate ``apply_lr_scale=True`` to the kernel, which
attaches ``_lr_scale = 1/(2π·ω₀)`` to the first-layer weight (or
``1/(2π·omega_0_max)`` in the block-diag case) so the per-step weight
update size matches the standard SIREN init.  ``_lr_scale`` is honoured
end-to-end by :func:`construct_optimizer` in
``experiments/lightning_wrappers/base_lightning_wrapper.py``.

Override the clamp bounds via CLI when desired (single line)::

    +net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0_scale_max=4.0
"""

from examples.vit5_imagenet.vit5_hybrid._base_config import (
    _GRID_H,
    KERNEL_EMBEDDING_DIM,
    KERNEL_HIDDEN_OMEGA_0,
    KERNEL_MLP_HIDDEN_DIM,
    KERNEL_NUM_LAYERS,
    KERNEL_OMEGA_0,
)
from examples.vit5_imagenet.vit5_hybrid._blockdiag import (
    KERNEL_BLOCK_DIAG_NUM_BLOCKS,
    KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE,
    KERNEL_BLOCK_DIAG_OMEGA_0_MAX,
    KERNEL_BLOCK_DIAG_OMEGA_0_MIN,
    KERNEL_BLOCK_DIAG_SCHEDULE,
)
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.kernels_nd import (
    BlockDiagonalLearnableOmegaSIRENKernelND,
    LearnableOmegaSIRENKernelND,
)
from nvsubquadratic.modules.masks_nd import BlockAlignedGaussianModulationND


# ─── Learnable-ω₀ scale defaults ─────────────────────────────────────────────
# The clamp lets every row up to double its effective ω₀ during training
# (max=2.0) while keeping a small *strictly positive* floor (min=1e-2) so
# no row can collapse to ω₀_eff = 0 — at zero the first-layer sine of that
# row becomes a constant ``sin(bias)`` and the gradient through the row's
# scale parameter largely vanishes (the row contributes no frequency
# information back to the loss), making recovery hard.  ``1e-2`` keeps the
# softest row at ~1% of the nominal ω₀, which still dampens the kernel
# substantially without irrecoverably killing it.
KERNEL_LEARNABLE_OMEGA_0_SCALE_MIN = 1e-2
KERNEL_LEARNABLE_OMEGA_0_SCALE_MAX = 2.0

# Enable the SIREN-paper LR compensation by default: 2π·ω₀ was pulled out
# of the first-layer weight init, so without scaling the optimizer would
# take effective steps ~2π·ω₀× smaller than a vanilla SIREN.  Setting
# _lr_scale = 1/(2π·ω₀) restores the original step size.
KERNEL_LEARNABLE_OMEGA_APPLY_LR_SCALE = True


# Same data_dim interpolation used in ``_make_hyena_block_cfg`` and
# ``_blockdiag.py``.  Inlined here so this module stays self-contained.
_DATA_DIM_REF = "${net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}"


def apply_learnable_omega_overrides(config) -> None:
    """Replace every Hyena block's kernel with :class:`LearnableOmegaSIRENKernelND`.

    Mutates ``config.net`` in place.  Must be called *after*
    :func:`build_hybrid_net`, before :func:`apply_config_overrides`.

    The mask is left untouched (baseline :class:`GaussianModulationND`).
    """
    gconv = config.net.layer_types["H"].sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg
    gconv.kernel_cfg = LazyConfig(LearnableOmegaSIRENKernelND)(
        data_dim=_DATA_DIM_REF,
        out_dim="${net.hidden_dim}",
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KERNEL_NUM_LAYERS,
        embedding_dim=KERNEL_EMBEDDING_DIM,
        omega_0=KERNEL_OMEGA_0,
        L_cache=_GRID_H,
        use_bias=True,
        hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
        omega_0_scale_init=1.0,
        omega_0_scale_min=KERNEL_LEARNABLE_OMEGA_0_SCALE_MIN,
        omega_0_scale_max=KERNEL_LEARNABLE_OMEGA_0_SCALE_MAX,
        apply_lr_scale=KERNEL_LEARNABLE_OMEGA_APPLY_LR_SCALE,
    )


def apply_learnable_omega_blockdiag_overrides(config) -> None:
    """Replace kernel + mask in every Hyena block with the block-diagonal learnable-ω₀ variant.

    Mutates ``config.net`` in place.  Must be called *after*
    :func:`build_hybrid_net`, before :func:`apply_config_overrides`.

    Uses the same block-diagonal schedule defaults as ``_blockdiag.py``
    (``num_blocks=8``, linear ω₀ schedule in ``[1, 12]``,
    ``off_block_scale=0.1``) but with the per-row ω₀ now learnable in
    ``[omega_0_scale_min, omega_0_scale_max]``.  Mask is upgraded to
    :class:`BlockAlignedGaussianModulationND` to match the block layout.
    """
    gconv = config.net.layer_types["H"].sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg
    gconv.kernel_cfg = LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
        data_dim=_DATA_DIM_REF,
        out_dim="${net.hidden_dim}",
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KERNEL_NUM_LAYERS,
        embedding_dim=KERNEL_EMBEDDING_DIM,
        L_cache=_GRID_H,
        use_bias=True,
        hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
        num_blocks=KERNEL_BLOCK_DIAG_NUM_BLOCKS,
        omega_0_min=KERNEL_BLOCK_DIAG_OMEGA_0_MIN,
        omega_0_max=KERNEL_BLOCK_DIAG_OMEGA_0_MAX,
        schedule=KERNEL_BLOCK_DIAG_SCHEDULE,
        off_block_scale=KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE,
        omega_0_scale_min=KERNEL_LEARNABLE_OMEGA_0_SCALE_MIN,
        omega_0_scale_max=KERNEL_LEARNABLE_OMEGA_0_SCALE_MAX,
        apply_lr_scale=KERNEL_LEARNABLE_OMEGA_APPLY_LR_SCALE,
    )
    gconv.mask_cfg = LazyConfig(BlockAlignedGaussianModulationND)(
        data_dim=_DATA_DIM_REF,
        num_channels="${net.hidden_dim}",
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=1.0,
        parametrization="direct",
    )
