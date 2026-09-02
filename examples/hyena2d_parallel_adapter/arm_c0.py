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

"""Arm C0: head + parallel bottleneck, no inner mixer.

Trainable: head + W_down/W_up in each block (no mixer between them).
Architecture: ViT5ParallelHyenaSequenceMixer with inner_mixer=Identity.

Baseline: does simply adding a rank-r linear per token (W_up ∘ W_down) help?
If C2 > C0, the Hyena filter's structure is contributing something.
If C2 ≈ C0, the gain is just from having the bottleneck, not from mixing.

Run::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/arm_c0.py \
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
    make_parallel_mixer_cfg_c0,
)
from examples.hyena2d_parallel_adapter.modules import RemapPretrainKeys
from experiments.default_cfg import ExperimentConfig, StartFromCheckpointConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm C0 config (capacity ablation, no inner mixer).

    Args:
        seed: Random seed.

    Returns:
        ExperimentConfig for arm C0.
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
        block_cfg=make_block_cfg(make_parallel_mixer_cfg_c0()),
    )

    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        alias="best",
        strict=False,
        partial_load=True,
        run_path="",  # Set via CLI
        callbacks=[LazyConfig(RemapPretrainKeys)()],
    )

    config.comment = f"arm_c0_no_mixer_rank{RANK}"
    return config
