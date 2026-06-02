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

"""Full Mamba – 12 bidirectional Mamba blocks.

Layout: M M M M M M M M M M M M

Replaces every sequence mixer with a ``Mamba`` (mamba_ssm core, bidirectional)
block.  Everything else — training recipe, MLP, norms, drop-path, patch size,
CLS token, registers — is identical to ``full_hyena.py`` and ``full_attention.py``.

Mamba hyperparameters (mamba_ssm defaults):
  d_state=16, d_conv=4, expand=2, bidirectional=True

hidden_dim is reduced from 384 → 320 to compensate for the extra parameters
introduced by the bidirectional second Mamba core, targeting a ~27M param budget
comparable to full_hyena.py.

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


LAYER_PATTERN = "M" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Return the full-Mamba (12 M) hybrid config."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    config.net.hidden_dim = 320
    config.wandb.job_group = "vit5_hybrid"
    return config
