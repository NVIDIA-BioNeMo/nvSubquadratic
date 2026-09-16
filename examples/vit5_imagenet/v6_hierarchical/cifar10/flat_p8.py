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


"""CIFAR-10 patch-size ablation: flat (no merging), patch_size=8, grid=8×8, dim=384."""

from examples.vit5_imagenet.v6_hierarchical.cifar10._base import build_flat_config


def get_config():
    """Return the flat patch-8 CIFAR-10 experiment config."""
    return build_flat_config(patch_size=8)
