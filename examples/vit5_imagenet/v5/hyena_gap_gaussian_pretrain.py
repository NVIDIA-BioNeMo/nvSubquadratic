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

"""ViT-5-Small Hyena GAP gated + Gaussian mask — pretraining config.

Same architecture as hyena_gap_pretrain.py but with a learnable
GaussianModulationND mask on the SIREN kernel output instead of Identity.
"""

from examples.vit5_imagenet.v5._base import HIDDEN_DIM
from examples.vit5_imagenet.v5.hyena_gap_pretrain import get_config as _hyena_gap
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import GaussianModulationND


def get_config() -> ExperimentConfig:
    """Build Hyena GAP gated + Gaussian mask pretraining config."""
    config = _hyena_gap()
    config.net.block_cfg.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(
        GaussianModulationND
    )(
        data_dim=2,
        num_channels=HIDDEN_DIM,
        min_attenuation_at_step=0.1,
        max_attenuation_at_limit=0.95,
        init_extent=1.0,
        parametrization="direct",
    )
    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    return config
