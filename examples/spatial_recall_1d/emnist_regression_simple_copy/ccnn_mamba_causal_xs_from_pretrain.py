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

"""EMNIST Spatial Recall 1D - Mamba XS - Fine-tuning from Pretrained Checkpoint.

This config starts from weights pretrained with autoregressive pretraining
(ccnn_mamba_causal_xs_pretrain.py) and fine-tunes on the spatial recall task.

The pretrained checkpoint provides:
- Backbone weights (blocks, embeddings)
- These are loaded with strict=False since output heads may differ

Usage:
    Set start_from_checkpoint.run_path to your pretrain W&B run path, e.g.:
    --start_from_checkpoint.run_path=dromeroguzma/nvsubquadratic/abc123xyz
"""

from examples.spatial_recall_1d.emnist_regression_simple_copy.ccnn_mamba_causal_xs import (
    get_config as get_base_config,
)
from experiments.default_cfg import StartFromCheckpointConfig


def get_config():
    """Get config for fine-tuning Mamba XS from pretrained checkpoint."""
    # Start from the base regression config
    config = get_base_config()

    # Update wandb group to indicate fine-tuning
    config.wandb.job_group = "spatial_recall_1d_emnist_simple_copy_xs_finetune"

    # Configure checkpoint loading
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path="implicit-long-convs/nvsubquadratic/q1wklbij",
        alias="latest",  # or "latest"
        strict=True,  # Allow mismatches (output head may differ)
        partial_load=False,  # Set to True if shapes differ and you want overlapping slice loading
    )

    return config
