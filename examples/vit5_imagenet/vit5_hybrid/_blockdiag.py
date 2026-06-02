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

"""Post-creation override: swap Hyena kernels + masks for the block-diagonal variants.

Importable helper applied to a config whose ``net`` was built by
:func:`build_hybrid_net` from ``_base_config``.  Replaces the kernel and mask
in every Hyena block with::

    kernel_cfg → BlockDiagonalMultiOmegaSIRENKernelND
    mask_cfg   → BlockAlignedGaussianModulationND

with the production defaults selected from the spectrum-coverage study at the
base 29×29 kernel grid (``PR_BASE_L=15``).  See
``reports/ckconv_block_diagonal_kernel/`` for the full sweep and write-up.

When changing the kernel grid resolution by a factor ``m`` (e.g. via
``patch_size`` overrides), the schedule should be scaled uniformly by ``m``
(``omega_0_min``, ``omega_0_max`` ← ``m·omega_0_min``, ``m·omega_0_max``) to
preserve Nyquist-normalized spectral coverage.  Override these via CLI when
needed (single line)::

    +net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0_max=24.0
"""

from examples.vit5_imagenet.vit5_hybrid._base_config import (
    _GRID_H,
    KERNEL_EMBEDDING_DIM,
    KERNEL_HIDDEN_OMEGA_0,
    KERNEL_MLP_HIDDEN_DIM,
    KERNEL_NUM_LAYERS,
)
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.kernels_nd import BlockDiagonalMultiOmegaSIRENKernelND
from nvsubquadratic.modules.masks_nd import BlockAlignedGaussianModulationND


# ─── Block-diagonal multi-ω₀ SIREN defaults ──────────────────────────────────
KERNEL_BLOCK_DIAG_NUM_BLOCKS = 8
KERNEL_BLOCK_DIAG_OMEGA_0_MIN = 1.0
KERNEL_BLOCK_DIAG_OMEGA_0_MAX = 12.0
KERNEL_BLOCK_DIAG_SCHEDULE = "linear"
KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE = 0.1


# Same data_dim interpolation that ``_make_hyena_block_cfg`` uses.  Copying
# rather than importing keeps the helper self-contained and avoids depending
# on any private symbol beyond what we already need.
_DATA_DIM_REF = "${net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}"


def apply_block_diag_overrides(config) -> None:
    """Replace the kernel + mask in every Hyena block with block-diagonal variants.

    Mutates ``config.net`` in place.  Must be called *after*
    :func:`build_hybrid_net`, before :func:`apply_config_overrides`.

    ``layer_types["H"]`` is shared across all H blocks in the layer pattern, so
    a single override here applies to every Hyena block.
    """
    gconv = config.net.layer_types["H"].sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg
    gconv.kernel_cfg = LazyConfig(BlockDiagonalMultiOmegaSIRENKernelND)(
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
    )
    gconv.mask_cfg = LazyConfig(BlockAlignedGaussianModulationND)(
        data_dim=_DATA_DIM_REF,
        num_channels="${net.hidden_dim}",
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=1.0,
        parametrization="direct",
    )
