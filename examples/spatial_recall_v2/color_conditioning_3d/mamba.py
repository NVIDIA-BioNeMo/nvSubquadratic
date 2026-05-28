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

"""3D Color Conditioning — Mamba XS (v2).

Hidden dim: 160, 4 blocks, headdim=32, expand=2, bidirectional.
~1.90M params (matching Attention ~1.84M / Hyena ~1.89M).
Bidirectional doubles Mamba2 layers, so hidden_dim is reduced from 208 → 160.
3D volume [D=8, H=64, W=64] with coloured bounding boxes.

Note: hidden_dim must be a multiple of 16 for Mamba2.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_3d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 160
HEADDIM = 32
EXPAND = 2


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.optimizer.lr = 1e-3

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=True,
    )

    return config
