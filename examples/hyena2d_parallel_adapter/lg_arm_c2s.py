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

"""Arm c2s at 784 tokens — fine-tune on random placement from the lg pretrain checkpoint."""

from examples.hyena2d_parallel_adapter._base_lg import (
    FINETUNE_ITERS,
    LR_FINETUNE,
    make_config,
    make_net_cfg,
)
from examples.hyena2d_parallel_adapter.modules import RemapPretrainKeys
from experiments.callbacks.freeze_backbone import (
    ADAPTER_PATTERNS,
    HEAD_PATTERNS,
    FreezeBackboneCallback,
)
from experiments.callbacks.wup_norm_monitor import WUpNormMonitorCallback
from experiments.default_cfg import ExperimentConfig, StartFromCheckpointConfig
from nvsubquadratic.lazy_config import LazyConfig


ARM = "c2s"


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm c2s fine-tune config at 784 tokens."""
    config = make_config("random", FINETUNE_ITERS, LR_FINETUNE, seed)
    config.net = make_net_cfg(ARM)

    # Arm 'a' keeps the pretrain architecture exactly; the adapter arms nest
    # ViT5Attention one level deeper and need the key remap.
    if ARM == "a":
        config.start_from_checkpoint = StartFromCheckpointConfig(
            load=True,
            alias="best",
            strict=True,
            partial_load=False,
            run_path="",
        )
        # Head-only baseline: backbone frozen, classifier trains.
        trainable = list(HEAD_PATTERNS)
        config.callbacks = [LazyConfig(FreezeBackboneCallback)(trainable_patterns=trainable)]
    else:
        config.start_from_checkpoint = StartFromCheckpointConfig(
            load=True,
            alias="best",
            strict=False,
            partial_load=True,
            run_path="",
            callbacks=[LazyConfig(RemapPretrainKeys)()],
        )
        # PEFT arms: backbone frozen, head + adapter train.
        trainable = list(HEAD_PATTERNS) + list(ADAPTER_PATTERNS)
        config.callbacks = [
            LazyConfig(FreezeBackboneCallback)(trainable_patterns=trainable),
            LazyConfig(WUpNormMonitorCallback)(log_every_n_steps=100),
        ]

    config.comment = f"lg_arm_{ARM}_784"
    return config
