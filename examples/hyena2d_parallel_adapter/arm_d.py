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

"""Arm D: full fine-tune (all params unfrozen), no adapter, random placement.

Trainable: everything.
Architecture: same as pretrain (attention only).
Loads pretrain checkpoint with strict=True.

This is the unconstrained upper bound for fine-tuning without any structural
change.  If arm D doesn't beat arm A, the task is too easy.

Run::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/arm_d.py \
        start_from_checkpoint.run_path=<entity>/<project>/<run_id> \
        seed=0
"""

from examples.hyena2d_parallel_adapter._base import (
    FINETUNE_ITERS,
    LR_FINETUNE,
    make_attn_net_cfg,
    make_base_config,
)
from experiments.default_cfg import ExperimentConfig, StartFromCheckpointConfig


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm D config (full fine-tune, attention only).

    Args:
        seed: Random seed.

    Returns:
        ExperimentConfig for arm D.
    """
    config = make_base_config(
        placement="random",
        num_iters=FINETUNE_ITERS,
        lr=LR_FINETUNE,
        seed=seed,
    )
    config.net = make_attn_net_cfg()

    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        alias="best",
        strict=True,
        partial_load=False,
        run_path="",  # Set via CLI
    )

    config.comment = "arm_d_full_finetune_attn"
    return config
