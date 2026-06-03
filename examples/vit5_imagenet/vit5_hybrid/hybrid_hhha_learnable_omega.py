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

"""Hybrid Hyena/Attention – 3:1 ratio (HHHA), learnable-ω₀ kernel (scalar).

Layout: H H H A H H H A H H H A
         └─────── 3 groups ─────┘

Same architecture as ``hybrid_hhha`` but every Hyena block uses
``LearnableOmegaSIRENKernelND`` instead of the baseline scalar-ω₀
``SIRENKernelND``.  The ``2π·ω₀`` factor is taken out of the first-layer
weight init and applied as a runtime multiplier, with an additional
learnable per-row ``omega_0_scale`` clamped to ``[1e-2, 2]`` (so the
effective per-row ω₀ ranges from ``0.01·ω₀`` to ``2·ω₀`` and no row's
first-layer sine ever collapses to a constant).

``apply_lr_scale=True`` attaches ``_lr_scale = 1/(2π·ω₀)`` to the
first-layer weight, which ``construct_optimizer`` honours so the per-step
update size matches the standard SIREN init.

The mask is left as the baseline ``GaussianModulationND``.

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from examples.vit5_imagenet.vit5_hybrid._learnable_omega import (
    apply_learnable_omega_overrides,
)
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.callbacks.omega_scale_monitor import OmegaScaleMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


LAYER_PATTERN = "HHHA" * (NUM_BLOCKS // 4)


def get_config() -> ExperimentConfig:
    """Build the HHHA hybrid config with learnable-ω₀ scalar kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    apply_learnable_omega_overrides(config)
    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(OmegaScaleMonitorCallback)(log_every_n_steps=50))
    config.wandb.job_group = "vit5_hybrid_learnable_omega"
    return config
