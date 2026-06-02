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

"""ViT-5-Small attention baseline — best pretraining config.

Reproduces run 44or24g1 (82.22% val/acc_ema, 82.22% test):
- ViT5Attention (6 heads, RoPE, RMSNorm QK-norm), CLS readout, 4 registers
- trunc_normal(std=0.02) init, no biases (qkv, out_proj, MLP)
- SoftTargetCE loss, EMA decay=0.99996
- LAMB lr=4e-3, wd=0.05, cosine schedule, 800 epochs
- 3-Augment + Mixup 0.8 + CutMix 1.0, DropPath 0.05
"""

import torch

from examples.vit5_imagenet.v5._base import (
    FINAL_IMAGE_SIZE,
    HIDDEN_DIM,
    INPUT_CHANNELS,
    LAYER_SCALE_INIT,
    MLP_RATIO,
    NUM_BLOCKS,
    NUM_CLASSES,
    PATCH_SIZE,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory


NUM_HEADS = 6
NUM_REGISTERS = 4
DROP_PATH_RATE = 0.05

_INIT_FN = trunc_normal_init(std=0.02)
_INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


def get_config() -> ExperimentConfig:
    """Build ViT-5-Small attention pretraining config."""
    config = get_base_config()

    config.compile = True
    config.compile_mode = "max-autotune"

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h="${eval:'${net.image_size} // ${net.patch_size}'}",
                num_patches_w="${eval:'${net.image_size} // ${net.patch_size}'}",
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
                out_proj_bias=False,
                init_fn_qkv_proj=_INIT_FN,
                init_fn_out_proj=_INIT_FN,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                bias=False,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=_INIT_FN_FACTORY,
                init_method_out=_INIT_FN_FACTORY,
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )

    return config
