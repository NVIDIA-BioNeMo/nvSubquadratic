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

"""4 regs, compress-concat (cr=4), film3_after, register_concat readout (cr=4), RA x3.

Matches v3 pretraining baseline for init (trunc_normal) and gated Hyena (Sigmoid).
Fixes rand_augment (disabled) vs r4_cc4_f3after which had rand-m9 + small_init/wang_init.
"""

from examples.vit5_imagenet.v4._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=4,
        num_film_layers=3,
        film_after_pos_embed=True,
        register_pooling_mode="compress_concat",
        film_compression_ratio=4,
        num_repeats=3,
        readout="register_concat",
        neck_compression_ratio=4,
        # Match v3 pretraining baseline
        init_style="v3_trunc_normal",
        use_gated_hyena=True,
    )
