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

"""Phase 1 (784 tokens): pretrain attention-only on fixed-placement spatial recall."""

from examples.hyena2d_parallel_adapter._base_lg import (
    LR_PRETRAIN,
    PRETRAIN_ITERS,
    make_config,
    make_net_cfg,
)
from experiments.default_cfg import ExperimentConfig


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the 784-token pretrain config (attention only, fixed placement)."""
    config = make_config("fixed", PRETRAIN_ITERS, LR_PRETRAIN, seed)
    config.net = make_net_cfg("a")
    config.comment = "lg_pretrain_fixed_784"
    return config
