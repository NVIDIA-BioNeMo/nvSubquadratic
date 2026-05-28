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

# TODO: Add license header here

"""3D Color Conditioning -- Hyena XS with per-axis Gaussian mask init only.

Ablation corner:
    - ``init_extent = (0.125, 1.0, 1.0)``  (per-axis, matching D/H/W over L_cache=64)
    - ``omega_0 = KERNEL_OMEGA_0`` (default = 10)

Isolates the effect of per-axis mask init_extent (without also bumping
omega_0). Paired with ``hyena_gaussian_mask_baseline.py``,
``hyena_gaussian_mask_omega30.py`` and ``hyena_gaussian_mask.py`` to form
a 2x2 design.
"""

from examples.spatial_recall_v2.color_conditioning_3d.hyena import HIDDEN_DIM
from examples.spatial_recall_v2.color_conditioning_3d.hyena import get_config as _get_hyena_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


DATA_DIM = 3
# Per-axis init_extent proportional to the axis size relative to L_cache (=64):
# depth = 8/64 = 0.125, H = W = 64/64 = 1.0
INIT_EXTENT_PER_AXIS = (0.125, 1.0, 1.0)


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = _get_hyena_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=DATA_DIM,
        num_channels=HIDDEN_DIM,
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=INIT_EXTENT_PER_AXIS,
        parametrization="direct",
    )

    return config
