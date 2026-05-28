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

"""1D Color Conditioning — Attention XS (v2, causal).

Hidden dim: 256, 4 blocks, num_heads=8, head_dim=32.
4 items on 1D canvas with coloured boundary markers, output digit in matching colour.
Causal by default; pass is_causal=False for bidirectional variant.
Compile: max-autotune.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_1d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 256
NUM_HEADS = 8  # head_dim = 32


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_mode = "max-autotune"

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        is_causal=True,
        rope_spatial_dims=("${eval:'${dataset.canvas_size} * ${dataset.canvas_size}'}",),
    )

    return config
