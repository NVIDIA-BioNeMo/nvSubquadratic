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

"""FiLM finetuning — relaxed FiLM weight decay (0.01).

Reduces WD on FiLM generator params to allow stronger input-dependent
conditioning while keeping global WD at 0.05.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(film_wd=0.01)
