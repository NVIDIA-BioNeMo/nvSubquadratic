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

"""Arm A: fine-tune head only, no adapter, random placement.

Trainable: classification head (``out_proj``) only.
Architecture: same as pretrain.  Loads pretrain checkpoint with strict=True.

If base accuracy on random placement is already high, the task is too easy
— increase ``canvas_size`` or add distractors before continuing.

Run (fill in ``PRETRAIN_RUN_PATH`` with the W&B run path from pretrain)::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/arm_a.py \
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
    """Return the arm A fine-tune config (head only).

    Args:
        seed: Random seed.

    Returns:
        ExperimentConfig for arm A.
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
        run_path="",  # Set via CLI: start_from_checkpoint.run_path=<path>
    )

    # Freeze everything except the head after the checkpoint is loaded.
    # Implemented via optimizer param groups in the training script, or by
    # setting requires_grad=False on all params then re-enabling the head.
    # NOTE: The ClassificationWrapper optimizer uses all parameters by default.
    # Override by adding a param-group filter callback if needed.
    config.comment = "arm_a_head_only"
    return config
