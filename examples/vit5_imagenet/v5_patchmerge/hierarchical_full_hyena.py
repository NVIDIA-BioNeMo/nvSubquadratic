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


"""Hierarchical Hyena — 12 blocks / 4 stages, scalar-ω₀ SIREN kernels.

Stage layout (Swin-T depths):
    Stem (4×4 conv, C₁=96)
    Stage 1: 2 × Hyena blocks  @ 56×56, dim=96
    PatchMerging → 28×28, dim=192
    Stage 2: 2 × Hyena blocks  @ 28×28, dim=192
    PatchMerging → 14×14, dim=384
    Stage 3: 6 × Hyena blocks  @ 14×14, dim=384
    PatchMerging → 7×7,   dim=768
    Stage 4: 2 × Hyena blocks  @ 7×7,   dim=768
    LayerNorm → GAP → Linear(768, 1000)

Compared with the isotropic vit5_hybrid configs:
  - 4-stage hierarchical pyramid vs. single-resolution
  - Patch Merging (Swin/VMamba recipe) instead of fixed patch_size
  - No CLS token, registers, or absolute position embeddings
  - Same pure-Hyena, no-RoPE, GRN setup as full_hyena.py
"""

from examples.vit5_imagenet.v5_patchmerge._base_config import (
    get_base_config,
    get_hierarchical_net_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Build the hierarchical all-Hyena config with scalar-ω₀ SIREN kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = get_hierarchical_net_config(base_dim=96)
    config.wandb.job_group = "v5_patchmerge"
    return config
