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


r"""Hierarchical Hyena (Swin-style patch merging) — CIFAR-10 patch-merging ablation.

Architecture: ViT5HierarchicalNet — 4-stage Hyena pyramid.
  - Swin-T depths [2, 2, 6, 2], base_dim=80 → dims [80,160,320,640] (~20.0 M params)
  - Stem: 4×4 stride-4 conv → 56×56 spatial (input resized to 224×224)
  - PatchMerging2D between stages: 56→28→14→7
  - GAP readout, no CLS token, no APE, no RoPE
  - Per-stage SIREN kernels with Nyquist-scaled ω₀
  - GRN on all blocks, drop-path linearly ramped 0 → 0.1

base_dim=80 chosen to match the isotropic baseline (~22.3 M) as closely as
possible while keeping dims as multiples of 8 and a meaningful pyramid width.
  base_dim=80 → 20.0 M  (chosen)
  base_dim=88 → 24.1 M
  base_dim=96 → 28.5 M  (Swin-T default)

Compare against isotropic_hyena.py as an architectural comparison.
Widths, normalization and compute also differ; this does not isolate
the effect of patch merging alone.

Run::

    PYTHONPATH=. python experiments/run.py \\
        --config examples/patch_merging/cifar10/hierarchical_hyena.py
"""

from examples.patch_merging.cifar10._base import NUM_CLASSES, get_base_config
from examples.vit5_imagenet.v5_patchmerge._base_config import get_hierarchical_net_config
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Build hierarchical Hyena config for CIFAR-10 single-GPU training."""
    config = get_base_config()
    config.net = get_hierarchical_net_config(base_dim=80, fft_backend="torch_fft", num_classes=NUM_CLASSES)
    return config
