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

"""v5_patch ablation — Attention baseline, patch_size=8.

Grid: 28x28 = 784 patches + 1 CLS + 4 registers = 789 tokens.
Batch: 64/gpu x 4 accum x 8 gpus = 2048 effective.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_attention_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 8


def get_config() -> ExperimentConfig:
    """Return Attention patch-8 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_attention_net(PATCH_SIZE)
    config.compile_compatible_fftconv = False
    return config
