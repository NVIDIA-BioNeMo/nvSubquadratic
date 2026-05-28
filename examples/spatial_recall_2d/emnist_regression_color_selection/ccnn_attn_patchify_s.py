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

"""EMNIST Spatial Recall 2D (Color Selection) - Attention + Patchify S.

Model Size: S (Small)
- Hidden dim: 256
- num_heads=8, head_dim=32

Task: 4 items on canvas with colored frames, select digit by color.
Input: 3-channel RGB with colored bounding boxes.
"""

import examples.spatial_recall_2d.mixer_defaults as spatial_recall_2d_mixer_defaults
from examples.spatial_recall_2d.base_config import (
    base_emnist_spatial_recall_2d_dataset_config,
)
from examples.spatial_recall_2d.base_config import (
    base_experiment_config as spatial_recall_2d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16
CANVAS_SIZE = 64
NUM_ITEMS = 4

# Network parameters - S size
INPUT_CHANNELS = 3  # RGB with colored frames
OUTPUT_CHANNELS = 1
HIDDEN_DIM = 256
NUM_HEADS = 8  # head_dim = 256/8 = 32

# Patchification parameters
PATCH_SIZE = 8
STRIDE = 8

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST color selection with Attention + Patchify S."""
    config = spatial_recall_2d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_2d_emnist_color_selection_s",
    )

    config.net.in_proj_cfg = LazyConfig(Patchify)(
        in_features="${net.in_channels}",
        out_features="${net.hidden_dim}",
        data_dim="${net.data_dim}",
        patch_size=PATCH_SIZE,
        stride=STRIDE,
    )
    config.net.out_proj_cfg = LazyConfig(Unpatchify)(
        in_features="${net.hidden_dim}",
        out_features="${net.out_channels}",
        data_dim="${net.data_dim}",
        patch_size="${net.in_proj_cfg.patch_size}",
        stride="${net.in_proj_cfg.stride}",
    )

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_2d_mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
    )

    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_2d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        use_colored_frames=True,  # RGB with colored bounding boxes
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=False,  # No mask, selection by color
        normalize_input=True,
    )

    return config
