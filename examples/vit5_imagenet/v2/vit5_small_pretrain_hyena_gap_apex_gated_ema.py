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

"""ViT-5-Small + Hyena GAP gated pretrain with EMA — thin wrapper.

Loads the base GAP gated config and adds LabeledEMAWeightAveraging (decay=0.99996).
Validation metrics are logged as ``val/acc_ema``, ``val/loss_ema``.
"""

from examples.vit5_imagenet.v2.vit5_small_pretrain_hyena_gap_apex_gated import get_config as _base_get_config
from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


def get_config() -> ExperimentConfig:
    """Return base GAP gated pretrain config with EMA weight averaging."""
    config = _base_get_config()
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
    config.trainer.checkpoint_monitor = "val/acc_ema"
    return config
