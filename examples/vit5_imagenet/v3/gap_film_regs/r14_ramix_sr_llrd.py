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

"""A3 baseline + small_random FiLM + LLRD=0.75 (Sweep 6: training recipe)."""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        lr=3e-4,
        wd=0.05,
        drop_path_rate=0.15,
        smoothing=0.1,
        mixup=0.8,
        cutmix=1.0,
        use_three_augment=False,
        rand_augment="rand-m9-mstd0.5-inc1",
        optimizer_type="lamb",
        film_init_type="small_random",
        layer_decay=0.75,
    )
