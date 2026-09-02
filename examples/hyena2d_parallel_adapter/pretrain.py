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

"""Phase 1: pretrain attention-only ViT on fixed-placement spatial recall.

The digit is always placed top-left (``placement="fixed"``).  The model learns
one routing path.  Save the checkpoint; all fine-tuning arms start here.

Run::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/pretrain.py \
        seed=0
"""

from examples.hyena2d_parallel_adapter._base import (
    LR_PRETRAIN,
    PRETRAIN_ITERS,
    make_attn_net_cfg,
    make_base_config,
)
from experiments.default_cfg import ExperimentConfig


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the Phase 1 pretrain config.

    Args:
        seed: Random seed (override with CLI ``seed=N``).

    Returns:
        ExperimentConfig for pretrain.
    """
    config = make_base_config(
        placement="fixed",
        num_iters=PRETRAIN_ITERS,
        lr=LR_PRETRAIN,
        seed=seed,
    )
    config.net = make_attn_net_cfg()
    config.comment = "pretrain_fixed"
    return config
