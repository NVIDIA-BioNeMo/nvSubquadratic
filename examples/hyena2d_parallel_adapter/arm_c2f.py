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

"""Arm C2f at 196 tokens — FiLM-conditioned Hyena (HyenaND proper) at the ORIGINAL scale.

Confound-decomposition arm.  The first run changed nothing about FiLM and used a
196-token grid; the 784-token rerun changes both at once.  This arm holds the grid
at 196 and turns FiLM on, so the two variables can be separated:

    C2s @196 (original)  vs  C2f @196 (this)   -> did FiLM alone matter?
    C2f @196 (this)      vs  C2f @784          -> did grid size matter?

Reuses the existing 196-token pretrain checkpoint.
"""

from examples.hyena2d_parallel_adapter._base import (
    FINETUNE_ITERS,
    GRID_W,
    HIDDEN_DIM,
    IMAGE_SIZE,
    IN_CHANNELS,
    LR_FINETUNE,
    NUM_BLOCKS,
    NUM_CLASSES,
    PATCH_SIZE,
    RANK,
    make_attn_cfg,
    make_base_config,
    make_block_cfg,
    make_hyena2d_inner_mixer_cfg,
)
from examples.hyena2d_parallel_adapter.modules import RemapPretrainKeys
from experiments.callbacks.wup_norm_monitor import WUpNormMonitorCallback
from experiments.default_cfg import ExperimentConfig, StartFromCheckpointConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm C2f config (FiLM-conditioned Hyena at 196 tokens)."""
    config = make_base_config(
        placement="random",
        num_iters=FINETUNE_ITERS,
        lr=LR_FINETUNE,
        seed=seed,
    )

    mixer_cfg = LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=make_hyena2d_inner_mixer_cfg(use_film=True),
            grid_w=GRID_W,
        ),
        conditioning_source="token_mean",
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
        block_cfg=make_block_cfg(mixer_cfg),
    )

    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        alias="best",
        strict=False,
        partial_load=True,
        run_path="",
        callbacks=[LazyConfig(RemapPretrainKeys)()],
    )
    config.callbacks = [LazyConfig(WUpNormMonitorCallback)(log_every_n_steps=100)]
    config.comment = f"arm_c2f_film_hyena_rank{RANK}"
    return config
