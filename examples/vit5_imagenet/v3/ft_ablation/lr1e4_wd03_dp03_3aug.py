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

"""FiLM finetuning — lr=1e-4, dp=0.3, wd=0.3, free FiLM + three-augment.

Maximum regularization combo: combines ALL the best strategies found so far.
dp=0.3 (strong structural reg) + three-augment (data reg) + wd=0.3 (weight
reg) + free FiLM. Tests whether stacking all regularizers can push the
overfit-free window past epoch 10.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.3,
        film_wd=True,
        use_three_augment=True,
    )
