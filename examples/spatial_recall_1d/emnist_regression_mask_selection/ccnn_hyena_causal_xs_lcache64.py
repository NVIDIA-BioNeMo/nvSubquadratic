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

"""EMNIST Spatial Recall 1D (Mask Selection) - Hyena XS Causal with L_cache=64.

1D version of mask selection where:
1. Images are flattened FIRST (16×16 → 256 elements)
2. 4 flattened images placed as contiguous segments in 1D canvas (4096 elements)
3. Binary mask channel indicates which digit to recall
4. Model must regress the target region for the masked digit (causal)

Key insight from simple_copy experiments:
- L_cache=64 means SIREN uses coarser frequency grid (step_size=1/63 vs 1/4095)
- This dramatically improved performance in simple_copy task
- Testing if this transfers to mask selection

Model Size: XS (Extra-Small)
- Hidden dim: 160
- SIREN kernel with 3 layers
- L_cache: 64 (NOT 4096!)
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
from nvsubquadratic.utils.qk_norm import L2Norm


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
NUM_ITEMS = 4  # 1 target + 3 distractors

# Network parameters - XS size
INPUT_CHANNELS = 2  # Grayscale + Mask
OUTPUT_CHANNELS = 1  # Grayscale target
HIDDEN_DIM = 160

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for EMNIST mask selection 1D with Hyena XS (L_cache=64)."""
    config = spatial_recall_1d_base_experiment_config(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        training_iterations=TRAINING_ITERATIONS,
        wandb_job_group="spatial_recall_1d_emnist_mask_selection_xs",
    )

    # Mixer: Hyena with causal convolutions and L_cache=canvas_size (64)
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_hyena_mixer_cfg(
        is_causal=True,
        qk_norm_cfg=LazyConfig(L2Norm)(),
        L_cache="${dataset.canvas_size}",  # Key: Use 64 instead of 4096!
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
