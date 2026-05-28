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

"""EMNIST Spatial Recall 1D (Mask Selection) - Attention M Causal + Patchify.

1D version of mask selection with patchification:
- Input: [B, 4096, 2] → Patchify (patch_size=64) → [B, 64, hidden_dim]
- Process: Self-attention on 64 tokens (O(64²) = 4096 vs O(4096²) = 16.7M)
- Output: [B, 64, hidden_dim] → Unpatchify → [B, 4096, 1]

Task:
- 4 flattened images placed in 1D canvas (4096 elements)
- Binary mask channel indicates which digit to recall
- Model must regress the target region for the masked digit (causal)

Model Size: M (Medium)
- Hidden dim: 384
- Num heads: 12 (head_dim=32)
"""

import examples.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from examples.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from examples.spatial_recall_1d.base_config import (
    base_experiment_config as spatial_recall_1d_base_experiment_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
NUM_ITEMS = 4  # 1 target + 3 distractors

# Patchification parameters
PATCH_SIZE = 64  # 4096 / 64 = 64 tokens
STRIDE = PATCH_SIZE  # Non-overlapping patches

# Network parameters - M size
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 384
NUM_HEADS = 12  # head_dim = 384 / 12 = 32

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST mask selection 1D with Attention M + Patchify."""
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_mask_selection_m_patchify",
    )

    # Override in/out projections with Patchify/Unpatchify
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

    # Mixer: Multi-head self-attention (causal)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        is_causal=True,  # Causal!
    )

    # Dataset: 1D spatial recall with mask selection
    assert config.dataset == PLACEHOLDER
    config.dataset = base_emnist_spatial_recall_1d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        num_items=NUM_ITEMS,
        placement="random",
        with_mask=True,
        normalize_input=True,
    )

    return config
