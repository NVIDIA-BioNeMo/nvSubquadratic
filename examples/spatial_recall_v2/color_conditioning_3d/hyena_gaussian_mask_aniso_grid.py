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

"""3D Color Conditioning -- Hyena XS with anisotropic kernel grid.

Intervention isolated by this config:
    - SIREN kernel ``L_cache = (D, H, W) = (CANVAS_DEPTH, CANVAS_SIZE, CANVAS_SIZE)``
      i.e. per-axis kernel grid that matches the actual volume.  Each axis spans
      the full ``[-1, 1]`` range at its own resolution; no axis-0 region is
      wasted in the cube cache.
    - Gaussian mask ``init_extent = (DEPTH_INIT_EXTENT, 1.0, 1.0)`` where the
      depth scale is large enough to saturate the entire depth-axis logspace
      ramp at ``max_std`` (depth axis is essentially **unmasked at init** —
      every channel starts at the widest possible Gaussian on depth).  H and
      W keep the reference ramp ``[min_std, init_std_high_unit]``.

      Why the depth axis must be max-saturated: the mask's ``min_step`` is
      driven by the densest grid axis (here grid_size=127, min_step≈0.0159
      → min_std≈0.0075).  The depth kernel grid, in contrast, has only 15
      points spanning ``[-1, 1]`` (physical step ≈0.143).  A channel with
      std≈0.0075 masks the depth axis to ≈0 at the very first cell, so the
      bottom half of any depth-axis logspace ramp anchored at min_std is
      unusable.  Lifting the entire depth ramp to ``max_std`` removes this
      regime mismatch — depth is left to be shaped by the (small)
      ``L_cache_d=8`` kernel grid, which already constrains its frequency
      content.
    - ``omega_0 = KERNEL_OMEGA_0`` (default = 10), unchanged.

Compare against:
    - ``hyena_gaussian_mask_baseline.py``:  isotropic L_cache=64 (cube cache),
      isotropic init_extent=1.0.  Differs from this config in **both** the
      kernel grid and the depth-axis mask init — measures the combined
      effect of the anisotropic-grid intervention.
    - ``hyena_gaussian_mask_peraxis.py``:   isotropic L_cache=64,
      ``init_extent=(0.125, 1.0, 1.0)`` (per-axis mask, narrowed depth,
      previous ablation winner).  This config takes the opposite tack on
      depth: the kernel grid is shrunk to size 8 on depth and the mask is
      maximally relaxed on depth — anisotropy is moved entirely from the
      mask to the grid.

The full kernel cache for ``L_cache=(8, 64, 64)`` is
``(1, 15, 127, 127, embedding_dim)`` instead of the cube
``(1, 127, 127, 127, embedding_dim)`` used by the baseline — ~8.5x fewer
SIREN forward FLOPs on the kernel grid.
"""

import math

from examples.spatial_recall_v2.color_conditioning_3d._base import CANVAS_DEPTH, CANVAS_SIZE
from examples.spatial_recall_v2.color_conditioning_3d.hyena import HIDDEN_DIM
from examples.spatial_recall_v2.color_conditioning_3d.hyena import get_config as _get_hyena_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


DATA_DIM = 3

# Per-axis kernel grid: matches the (D, H, W) of the input volume so each
# axis is sampled at its own native resolution.  With grid_type="double"
# this produces a kernel cache of shape (1, 15, 127, 127, C).
L_CACHE_PER_AXIS = [CANVAS_DEPTH, CANVAS_SIZE, CANVAS_SIZE]

# Mask: ``init_extent`` multiplicatively scales the per-axis logspace ramp.
# We want the entire depth-axis ramp to saturate at ``max_std``, i.e.
# ``min_std * extent ≥ max_std``.  At our config (min_attn=0.1,
# max_attn=0.95, grid_size=127):
#     min_step = 2/126 ≈ 0.01587
#     min_std  = min_step * sqrt(-1/(2 ln 0.1)) ≈ 0.0075
#     max_std  =        1 * sqrt(-1/(2 ln 0.95)) ≈ 3.121
#     max_std / min_std ≈ 416
# We compute the saturation threshold symbolically so it stays correct if
# the attenuation parameters or grid_size ever change.
_MASK_GRID_SIZE = 2 * max(L_CACHE_PER_AXIS) - 1  # auto-injected by CKConvND
_MIN_STEP = 2.0 / (_MASK_GRID_SIZE - 1)
_MIN_STD = _MIN_STEP * math.sqrt(-1.0 / (2.0 * math.log(0.1)))
_MAX_STD = 1.0 * math.sqrt(-1.0 / (2.0 * math.log(0.95)))
DEPTH_INIT_EXTENT = _MAX_STD / _MIN_STD  # ≈ 416 → both ends clamp at max_std
INIT_EXTENT_PER_AXIS = (DEPTH_INIT_EXTENT, 1.0, 1.0)


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = _get_hyena_config()

    # Per-axis Gaussian-mask init: depth saturates at max_std (essentially
    # unmasked along depth at init); H and W use the reference ramp.
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=DATA_DIM,
        num_channels=HIDDEN_DIM,
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=INIT_EXTENT_PER_AXIS,
        parametrization="direct",
    )

    # Anisotropic SIREN kernel grid: one extent per axis instead of the
    # default scalar ``${dataset.canvas_size}``.
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.L_cache = L_CACHE_PER_AXIS

    return config
