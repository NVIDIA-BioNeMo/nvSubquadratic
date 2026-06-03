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

"""WSD finetuning ablation — minimal augmentation, LR=1e-5, WD=0.1.

Closest to pure finetuning: no Mixup, no CutMix, no RandAugment, no
ThreeAugment. Only basic random-resized-crop and horizontal flip.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with minimal augmentation at LR=1e-5."""
    return _base(
        lr=1e-5,
        wd=0.1,
        mixup=0.0,
        cutmix=0.0,
        rand_augment=None,
        use_three_augment=False,
    )
