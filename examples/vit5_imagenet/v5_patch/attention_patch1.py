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

"""v5_patch ablation — Attention baseline, patch_size=1.

Grid: 224x224 = 50176 patches + 1 CLS + 4 registers = 50181 tokens.
Batch: 1/gpu x 256 accum x 8 gpus = 2048 effective.

WARNING: O(n^2) attention on ~50K tokens is almost certainly infeasible
on H100 80GB. This config exists for completeness — expect OOM.
Consider activation checkpointing or skip this config for attention.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_attention_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 1


def get_config() -> ExperimentConfig:
    """Return Attention patch-1 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_attention_net(PATCH_SIZE)
    config.compile_compatible_fftconv = False
    return config
