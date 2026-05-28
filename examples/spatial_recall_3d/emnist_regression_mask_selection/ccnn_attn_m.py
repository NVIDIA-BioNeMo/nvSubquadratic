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

"""EMNIST Spatial Recall 3D - Mask Selection - Attention M (Medium).

3D Spatial Recall Task with Mask Selection:
- Multiple 2D images placed on depth slices of a 3D volume [D, H, W]
- One target, multiple distractors (all on different depth slices or positions)
- Mask channel indicates which item is the target
- Must recall target at back-bottom-right corner (last depth slice)

Model Size: M (Medium)
- Hidden dim: 384
- Params: ~4.4M
- num_heads: 12, head_dim: 32 (consistent with S-size)

Size Reference:
- XS: ~160 channels, 8 heads, head_dim=20
- S:  ~256 channels, 8 heads, head_dim=32
- M:  ~384 channels, 12 heads, head_dim=32
"""

import examples.spatial_recall_3d.mixer_defaults as spatial_recall_3d_mixer_defaults
from examples.spatial_recall_3d.base_config import (
    base_emnist_spatial_recall_3d_dataset_config,
)
from examples.spatial_recall_3d.base_config import (
    base_experiment_config as spatial_recall_3d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


# Dataset-specific parameters
BATCH_SIZE = 8  # Smaller batch due to larger model
TARGET_SIZE = 16
CANVAS_SIZE = 64  # H and W dimensions
CANVAS_DEPTH = 8  # D dimension

# Network parameters - M size
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 384
NUM_HEADS = 12  # head_dim = 384/12 = 32 (consistent with S-size)

NUM_ITEMS = 4  # target + 3 distractors

# Training parameters
TRAINING_ITERATIONS = 50_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST spatial recall 3D mask selection with Attention M."""
    config = spatial_recall_3d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_3d_emnist_mask_selection_m",
        target_size=TARGET_SIZE,
    )

    # Mixer: Attention (no RoPE for 3D - head_dim must be divisible by 6)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_3d_mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        use_rope=False,  # Disable RoPE for 3D (head_dim=32 divisible by 6, but keeping consistent with XS)
    )

    # Dataset
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_3d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        canvas_depth=CANVAS_DEPTH,
        batch_size=BATCH_SIZE,
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=True,
        normalize_input=True,
    )

    return config
