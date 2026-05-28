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

"""FiLM finetuning — higher drop path (0.20).

dp=0.20 reached 82.02% in v2. Tests whether FiLM benefits from
stronger structural regularization than the attention model.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(drop_path_rate=0.20)
