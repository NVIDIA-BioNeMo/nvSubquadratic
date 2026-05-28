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

"""Shared helpers for detecting channel-first normalization modules."""

import torch.nn as nn


def is_channels_first_norm(module: nn.Module) -> bool:
    """Return True if *module* normalizes over dim=1 and accepts [B, C, *spatial] directly.

    Covers ``nn.GroupNorm``, ``RMSNormChannelFirst``, and any future norm that
    sets the ``channels_first`` class/instance attribute to ``True``.
    """
    if isinstance(module, nn.GroupNorm):
        return True
    return getattr(module, "channels_first", False)
