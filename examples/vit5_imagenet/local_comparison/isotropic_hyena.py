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


r"""Isotropic pure-Hyena baseline for local patch-merging comparison.

Architecture: ViT5ClassificationNet — flat (isotropic) 12-block Hyena.
  - patch_size=16, hidden_dim=384, 12 blocks (ViT-S scale, ~22 M params)
  - GAP readout, no CLS token, no APE
  - Scalar-ω₀ SIREN kernel: 3-layer MLP, ω₀=10.0, L_cache=14
  - SiLU 1st gate + Sigmoid 2nd gate, L2 QK-norm, output RMSNorm

Architecturally identical to
``examples/vit5_imagenet/v5/hyena_gap_pretrain.py`` (the 81.50% reference run).
Training schedule patched for local single RTX 3090:
  - batch_size=64, accumulate_grad_steps=32 → effective batch 2048
  - 100 epochs, compile disabled

Run::

    PYTHONPATH=. python experiments/run.py \\
        --config examples/vit5_imagenet/local_comparison/isotropic_hyena.py
"""

import torch

from examples.vit5_imagenet.local_comparison._base_local import get_local_base_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Architecture constants (from v5._base, copied to avoid apex at import time) ─
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
FINAL_IMAGE_SIZE = 224
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4

DROP_PATH_RATE = 0.05

KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0


def get_config() -> ExperimentConfig:
    """Build isotropic pure-Hyena config for local single-GPU training."""
    config = get_local_base_config(epochs=100)

    hyena_mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache="${eval:'${net.image_size} // ${net.patch_size}'}",
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                fft_backend="subq_ops",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=0,
        dropout_rate=0.0,
        readout="gap",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w="${eval:'${net.image_size} // ${net.patch_size}'}",
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                bias=False,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )

    return config
