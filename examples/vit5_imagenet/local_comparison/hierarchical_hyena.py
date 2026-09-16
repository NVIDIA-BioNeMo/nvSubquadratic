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


r"""Hierarchical Hyena (Swin-style patch merging) for local patch-merging comparison.

Architecture: ViT5HierarchicalNet — 4-stage Hyena pyramid with PatchMerging.
  - Swin-T depths [2, 2, 6, 2], base_dim=96 → dims [96,192,384,768] (~28.5 M params)
  - Stem: 4×4 stride-4 conv → 56×56 spatial
  - PatchMerging2D between stages: 56→28→14→7
  - GAP readout, no CLS token, no APE, no RoPE
  - Per-stage SIREN kernels with Nyquist-scaled ω₀:
      Stage 1 @ 56×56: ω₀=40   Stage 2 @ 28×28: ω₀=20
      Stage 3 @ 14×14: ω₀=10   Stage 4 @  7×7:  ω₀=5
  - GRN (Global Response Normalization) on all blocks
  - Drop-path rate linearly ramped 0 → 0.1 across all 12 blocks

Compare against ``isotropic_hyena.py`` (flat 12-block, patch_size=16, ~22 M params).

Run::

    PYTHONPATH=. python experiments/run.py \\
        --config examples/vit5_imagenet/local_comparison/hierarchical_hyena.py
"""

from examples.vit5_imagenet.local_comparison._base_local import get_local_base_config
from examples.vit5_imagenet.v5_patchmerge._base_config import get_hierarchical_net_config
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Build hierarchical pure-Hyena config for local single-GPU training."""
    config = get_local_base_config(epochs=100)
    config.net = get_hierarchical_net_config(base_dim=96)
    return config
