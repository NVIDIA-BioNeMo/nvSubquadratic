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

"""Arm C2: head + parallel Hyena2D adapter (global implicit filter), random placement.

Trainable: head (``out_proj``) + w_down/w_up/hyena_adapter inside each block.
Architecture: ViT5ParallelHyenaSequenceMixer with ViT5HyenaAdapter inner mixer.

C2 is the main arm under test.  Compare vs:
  - C0 (no mixer): does mixing matter at all?
  - C1 (local conv): does *global* mixing matter?

The inductive-bias signature: C2 > C0 on far-displacement buckets, not near.
A uniform win means capacity, not inductive bias — report that if it happens.

Run::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/arm_c2.py \
        start_from_checkpoint.run_path=<entity>/<project>/<run_id> \
        seed=0
"""

from examples.hyena2d_parallel_adapter._base import (
    FINETUNE_ITERS,
    HIDDEN_DIM,
    IMAGE_SIZE,
    IN_CHANNELS,
    LR_FINETUNE,
    NUM_BLOCKS,
    NUM_CLASSES,
    PATCH_SIZE,
    RANK,
    make_base_config,
    make_block_cfg,
    make_parallel_mixer_cfg_c2,
)
from examples.hyena2d_parallel_adapter.modules import RemapPretrainKeys
from experiments.callbacks.wup_norm_monitor import WUpNormMonitorCallback
from experiments.default_cfg import ExperimentConfig, StartFromCheckpointConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm C fine-tune config (head + Hyena2D adapter).

    Args:
        seed: Random seed.

    Returns:
        ExperimentConfig for arm C.
    """
    config = make_base_config(
        placement="random",
        num_iters=FINETUNE_ITERS,
        lr=LR_FINETUNE,
        seed=seed,
    )

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        readout="corner",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=make_block_cfg(make_parallel_mixer_cfg_c2()),
    )

    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        alias="best",
        strict=False,
        partial_load=True,
        run_path="",  # Set via CLI: start_from_checkpoint.run_path=<path>
        callbacks=[LazyConfig(RemapPretrainKeys)()],
    )

    # Monitor ‖W_up‖ per layer — key diagnostic for the parallel adapter.
    config.callbacks = [
        LazyConfig(WUpNormMonitorCallback)(log_every_n_steps=100),
    ]

    config.comment = f"arm_c2_parallel_hyena_rank{RANK}"
    return config
