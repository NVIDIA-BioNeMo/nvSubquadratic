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

"""Hyena config for acoustic_scattering_maze (v2) with circular FFT padding.

Identical to ``hyena.py`` but uses circular (periodic) FFT padding instead of
zeros, to compare periodic vs. non-periodic convolution on this dataset.
"""

from examples.well.v2.acoustic_scattering_maze.hyena import get_config as _get_hyena_config
from experiments.default_cfg import ExperimentConfig


def get_config() -> ExperimentConfig:
    """Build Hyena + circular padding config for acoustic_scattering_maze."""
    config = _get_hyena_config()

    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.fft_padding = "circular"

    return config
