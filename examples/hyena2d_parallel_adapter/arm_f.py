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

"""Arm F: from-scratch interleaved Hyena+Attention hybrid, random placement.

Trainable: everything (trained from scratch, no pretrain checkpoint).
Architecture: ViT5ClassificationNet with layer_pattern="HAHAHA"
(alternating Hyena / Attention blocks), ``placement="random"`` throughout.

This is the from-scratch upper bound: if interleaved outperforms arm D (full
fine-tune from pretrain), the inductive bias of Hyena matters independent of
zero-init.  The nvSubquadratic paper establishes layer-interleaved hybrids beat
pure attention; arm F validates that result still holds on this task.

No checkpoint loading — trained from scratch on random placement.

Run::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/arm_f.py \
        seed=0
"""

from examples.hyena2d_parallel_adapter._base import (
    FINETUNE_ITERS,
    LR_PRETRAIN,
    make_base_config,
    make_interleaved_net_cfg,
)
from experiments.default_cfg import ExperimentConfig


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm F config (from-scratch interleaved hybrid).

    Args:
        seed: Random seed.

    Returns:
        ExperimentConfig for arm F.
    """
    # Arm F is trained from scratch; use pretrain LR and more iterations.
    config = make_base_config(
        placement="random",
        num_iters=FINETUNE_ITERS,
        lr=LR_PRETRAIN,
        seed=seed,
    )
    config.net = make_interleaved_net_cfg()
    config.comment = "arm_f_scratch_interleaved_HA"
    return config
