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

"""Hybrid Hyena/Attention – alternating HA pattern (12 blocks), block-diagonal kernel.

Layout: H A H A H A H A H A H A
         └─────── 6 pairs ─────┘

Same architecture as ``hybrid_ha`` but every Hyena block uses
``BlockDiagonalMultiOmegaSIRENKernelND`` paired with
``BlockAlignedGaussianModulationND`` instead of the scalar-ω₀ SIREN + standard
Gaussian mask.

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from examples.vit5_imagenet.vit5_hybrid._blockdiag import apply_block_diag_overrides
from experiments.default_cfg import ExperimentConfig


LAYER_PATTERN = "HA" * (NUM_BLOCKS // 2)


def get_config() -> ExperimentConfig:
    """Build the alternating HA hybrid config with block-diagonal Hyena kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    apply_block_diag_overrides(config)
    config.wandb.job_group = "vit5_hybrid_blockdiag"
    return config
