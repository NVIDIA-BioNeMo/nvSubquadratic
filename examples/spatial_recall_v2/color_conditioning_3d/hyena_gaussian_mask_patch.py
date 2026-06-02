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

"""3D Color Conditioning -- patched Hyena XS with per-axis Gaussian mask init.

Identical to ``hyena_patch.py`` but replaces the ``nn.Identity`` mask with a
``GaussianModulationND`` mask on the CKConv global convolution kernel.

The volume is [D=8, H=64, W=64] and the kernel cache spans L_cache=64 per
axis.  The depth axis therefore only uses 8/64 = 0.125 of the kernel grid
at runtime, so the mask init_extent is set per-axis to match: the widest
init channel on depth reaches the 10% mark at 0.125 (≈ full depth), while
on H and W it reaches 10% at 1.0 (full spatial extent), giving a
meaningful logspace ramp of bandwidths along every axis.

Also bumps the kernel's ``omega_0`` to 30 (from the default 10) to
sharpen the SIREN's initial frequency content.
"""

from examples.spatial_recall_v2.color_conditioning_3d.hyena_patch import HIDDEN_DIM
from examples.spatial_recall_v2.color_conditioning_3d.hyena_patch import get_config as _get_hyena_patch_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


DATA_DIM = 3
# Per-axis init_extent proportional to the axis size relative to L_cache (=64):
# depth = 8/64 = 0.125, H = W = 64/64 = 1.0
INIT_EXTENT_PER_AXIS = (0.125, 1.0, 1.0)
KERNEL_OMEGA_0 = 30.0


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = _get_hyena_patch_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=DATA_DIM,
        num_channels=HIDDEN_DIM,
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=INIT_EXTENT_PER_AXIS,
        parametrization="direct",
    )

    # Sharpen the SIREN kernel's first-layer frequency
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0 = KERNEL_OMEGA_0

    return config
