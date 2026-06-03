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

"""3D Simple Copy — Attention XS with Patchify (v2).

Hidden dim: 240, 4 blocks, num_heads=8, head_dim=30.
Patch_size=4 (8×64×64 → 2×16×16 = 512 tokens).
3D RoPE enabled (head_dim=30, 30 % 6 == 0).

Attention cost drops from O(32768²) to O(512²).
RoPE spatial dims are set to the patchified grid size.

Patch-size CLI override
-----------------------
Only ``net.in_proj_cfg.patch_size=P`` is needed; stride, out_proj, and
rope_spatial_dims are derived via interpolators.

Note: patch_size must evenly divide canvas_depth (8).  Max patch_size = 8.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.simple_copy_3d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


HIDDEN_DIM = 240
NUM_HEADS = 8  # head_dim = 30 (divisible by 6 for 3D RoPE)
PATCH_SIZE = 4  # 8/4 × 64/4 × 64/4 = 2×16×16 = 512 tokens


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_mode = "max-autotune-no-cudagraphs"

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
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        rope_spatial_dims=(
            "${eval:'${dataset.canvas_depth} // ${net.in_proj_cfg.patch_size}'}",
            "${eval:'${dataset.canvas_size} // ${net.in_proj_cfg.patch_size}'}",
            "${eval:'${dataset.canvas_size} // ${net.in_proj_cfg.patch_size}'}",
        ),
    )

    return config
