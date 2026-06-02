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

"""CNextU-net baseline for euler_multi_quadrants_periodicBC.

Reproduces the best baseline from the Well paper (Table 2: VRMSE = 0.1531).
The paper ran fp32 on 1× H100 and completed 1 epoch in 12h.
We use bf16-mixed + torch.compile for higher throughput.

Table 6: best LR for CNextU-net on this dataset = 5e-3 (1 epoch in 12h).
"""

from examples.well.v1.euler_multi_quadrants_periodicBC._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL_RESOLUTION,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet_convnext import WellUNetConvNext


# ─── Model hyperparameters (configs/model/unet_convnext.yaml) ────────────────
BATCH_SIZE = 24  # configs/data/euler_multi_quadrants_periodicBC.yaml
LEARNING_RATE = 5e-3  # Table 6: best LR for CNextU-net
WEIGHT_DECAY = 1e-4  # configs/optimizer/adam.yaml

INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1
GRADIENT_CHECKPOINTING = False  # bf16 halves memory; not needed on 80GB


def get_config() -> ExperimentConfig:
    """Build CNextU-net experiment config for euler_multi_quadrants_periodicBC."""
    config = get_base_config(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    config.net = LazyConfig(WellUNetConvNext)(
        dim_in=IN_CHANNELS,
        dim_out=OUT_CHANNELS,
        n_spatial_dims=DATA_DIM,
        spatial_resolution=SPATIAL_RESOLUTION,
        stages=STAGES,
        blocks_per_stage=BLOCKS_PER_STAGE,
        blocks_at_neck=BLOCKS_AT_NECK,
        init_features=INIT_FEATURES,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    return config
