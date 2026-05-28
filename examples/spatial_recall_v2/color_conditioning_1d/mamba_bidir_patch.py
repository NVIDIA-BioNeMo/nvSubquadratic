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

"""1D Color Conditioning — Mamba XS with Patchify (v2, bidirectional).

Hidden dim: 160, 4 blocks, headdim=32, expand=2, bidirectional.
~1.90M params (matching Attention ~1.84M / Hyena ~1.89M).
Patch_size=64 (4096 → 64 tokens).
4 items on 1D canvas with coloured boundary markers, output digit in matching colour.

Patchification compresses the 4096-element 1D canvas to 64 tokens via Conv1d
with kernel_size=stride=patch_size.

Patch-size CLI override
-----------------------
Only ``net.in_proj_cfg.patch_size=P`` is needed; stride and out_proj
patch/stride are derived via interpolators.

Note: hidden_dim must be a multiple of 16 for Mamba2 (d_inner = expand*d_model
must be divisible by headdim=32).
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_1d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


HIDDEN_DIM = 160
HEADDIM = 32
EXPAND = 2
PATCH_SIZE = 64  # 4096 / 64 = 64 tokens


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.optimizer.lr = 1e-3

    # ── Patchify / Unpatchify projections ─────────────────────────────
    config.net.in_proj_cfg = LazyConfig(Patchify)(
        in_features="${net.in_channels}",
        out_features="${net.hidden_dim}",
        data_dim="${net.data_dim}",
        patch_size=PATCH_SIZE,
        stride="${net.in_proj_cfg.patch_size}",
    )
    config.net.out_proj_cfg = LazyConfig(Unpatchify)(
        in_features="${net.hidden_dim}",
        out_features="${net.out_channels}",
        data_dim="${net.data_dim}",
        patch_size="${net.in_proj_cfg.patch_size}",
        stride="${net.in_proj_cfg.patch_size}",
    )

    # ── Mixer ─────────────────────────────────────────────────────────
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=True,
    )

    return config
