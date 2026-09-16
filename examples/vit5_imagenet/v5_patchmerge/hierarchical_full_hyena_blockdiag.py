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


"""Hierarchical Hyena — 12 blocks / 4 stages, block-diagonal multi-ω₀ SIREN kernels.

Same architecture as ``hierarchical_full_hyena.py`` but with
``BlockDiagonalMultiOmegaSIRENKernelND`` + ``BlockAlignedGaussianModulationND``
replacing the scalar-ω₀ SIREN at every block.

ω₀ schedule per stage (Nyquist-scaled, reference: vit5_hybrid/_blockdiag.py):
    Stage 1 @ 56×56: ω₀_min=4.0,  ω₀_max=48.0
    Stage 2 @ 28×28: ω₀_min=2.0,  ω₀_max=24.0
    Stage 3 @ 14×14: ω₀_min=1.0,  ω₀_max=12.0
    Stage 4 @  7×7:  ω₀_min=0.5,  ω₀_max=6.0
"""

from examples.vit5_imagenet.v5_patchmerge._base_config import (
    get_base_config,
    get_hierarchical_net_config,
)
from examples.vit5_imagenet.v5_patchmerge._blockdiag import apply_block_diag_config_overrides
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Build hierarchical all-Hyena config with block-diagonal SIREN kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = get_hierarchical_net_config(base_dim=96)
    apply_block_diag_config_overrides(config.net)
    config.wandb.job_group = "v5_patchmerge"
    return config
