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

"""Full Hyena – 12 Hyena blocks, learnable-ω₀ kernel (scalar variant).

Layout: H H H H H H H H H H H H

Same architecture as ``full_hyena`` but every Hyena block uses
``LearnableOmegaSIRENKernelND`` instead of the baseline scalar-ω₀
``SIRENKernelND``.  The ``2π·ω₀`` factor is taken out of the first-layer
weight init and applied as a runtime multiplier, with an additional
learnable per-row ``omega_0_scale`` clamped to ``[0, 2]`` (so the
effective per-row ω₀ can grow up to ``2·ω₀``).

``apply_lr_scale=True`` in the kernel attaches ``_lr_scale = 1/(2π·ω₀)``
to the first-layer weight, which ``construct_optimizer`` honours so the
per-step update size matches the standard SIREN init.

The mask is left as the baseline ``GaussianModulationND`` — for the
block-aware mask pair, see ``full_hyena_learnable_omega_blockdiag``.

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


LAYER_PATTERN = "H" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Build the all-Hyena config with learnable-ω₀ scalar kernels in every block."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    apply_learnable_omega_overrides(config)
    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(OmegaScaleMonitorCallback)(log_every_n_steps=50))
    config.wandb.job_group = "vit5_hybrid_learnable_omega"
    return config
