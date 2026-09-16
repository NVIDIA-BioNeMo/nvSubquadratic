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


"""CIFAR-10 patch-size ablation: hierarchical, patch_size=4.

4 stages: 16×16 → 8×8 → 4×4 → 2×2, dims=[96,192,384,768], depths=[2,2,6,2].
"""

from examples.vit5_imagenet.v6_hierarchical.cifar10._base import build_hier_config


def get_config():
    """Return the hierarchical patch-4 CIFAR-10 experiment config."""
    return build_hier_config(patch_size=4)
