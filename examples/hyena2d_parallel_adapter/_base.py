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

"""Shared constants and config builders for the parallel Hyena adapter experiment.

Task: MNIST digit placed on a 56×56 canvas, digit patch is 14×14.  The readout
region (bottom-right 14×14 corner) is filled with -1.0.  The model reads out
from the bottom-right patch token (``readout="corner"``).  Patch size 4 yields
a 14×14 = 196 token grid with no padding or CLS token.

Generalisation split:
  - Pretrain: ``placement="fixed"`` (digit always top-left).
  - Fine-tune / eval: ``placement="random"`` (arbitrary non-overlapping placement).

See the experiment brief for full motivation.
"""

import os

import torch

from experiments.datamodules.mnist import MNISTDataModule
from experiments.datamodules.spatial_recall_classification import (
    SpatialRecallClassificationDataModule,
)
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


# ── Architecture constants ──────────────────────────────────────────────────
HIDDEN_DIM = 128
NUM_BLOCKS = 6
NUM_HEADS = 4
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 32
PATCH_SIZE = 4
IMAGE_SIZE = 56
GRID_W = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = GRID_W * GRID_W  # 196
NUM_CLASSES = 10
IN_CHANNELS = 1

# Hyena branch bottleneck rank (arm C/E).
RANK = 32

# ── Dataset constants ───────────────────────────────────────────────────────
TARGET_SIZE = 14
CANVAS_SIZE = 56
READOUT_VALUE = -1.0

# ── Training constants ──────────────────────────────────────────────────────
PRETRAIN_ITERS = 50_000
FINETUNE_ITERS = 30_000
BATCH_SIZE = 128
LR_PRETRAIN = 3e-4
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-2


def _num_workers() -> int:
    n = os.cpu_count() or 4
    return min(n // max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1), 8)


def make_mnist_dataset_cfg(placement: str, seed: int = 0) -> LazyConfig:
    """Build the SpatialRecallClassificationDataModule config.

    Args:
        placement: ``"fixed"`` for pretrain, ``"random"`` for fine-tuning.
        seed: Base seed for placement generators.

    Returns:
        LazyConfig for SpatialRecallClassificationDataModule.
    """
    return LazyConfig(SpatialRecallClassificationDataModule)(
        base_datamodule_cfg=LazyConfig(MNISTDataModule)(
            data_dir=".data/mnist",
            batch_size=BATCH_SIZE,
            data_type="image",
            num_workers=_num_workers(),
            pin_memory=True,
            use_deterministic_worker_init=False,
            seed=seed,
            task="classification",
        ),
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        placement=placement,
        with_mask=False,
        readout_value=READOUT_VALUE,
        batch_size=BATCH_SIZE,
        num_workers=_num_workers(),
        pin_memory=True,
        seed=seed,
    )


def make_attn_cfg(num_registers: int = 0, has_cls: bool = False) -> LazyConfig:
    """Build the ViT5Attention LazyConfig for the toy model.

    Args:
        num_registers: Number of register tokens (0 for this toy).
        has_cls: Whether a CLS token is present (False for ``readout="corner"``).

    Returns:
        LazyConfig for ViT5Attention.
    """
    return LazyConfig(ViT5Attention)(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_patches_h=GRID_W,
        num_patches_w=GRID_W,
        num_registers=num_registers,
        has_cls=has_cls,
        qk_norm=LazyConfig(RMSNorm)(dim=HEAD_DIM, eps=1e-6),
        rope_base=10000.0,
        reg_rope_base=100.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=False,
        out_proj_bias=False,
    )


def make_block_cfg(sequence_mixer_cfg: LazyConfig) -> LazyConfig:
    """Build a ViT5ResidualBlock LazyConfig wrapping the given mixer.

    Args:
        sequence_mixer_cfg: LazyConfig for the sequence mixer.

    Returns:
        LazyConfig for ViT5ResidualBlock.
    """
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=sequence_mixer_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=4.0,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=1e-4,
        drop_path_rate=0.0,
    )


def make_attn_net_cfg(sequence_mixer_cfg: LazyConfig | None = None) -> LazyConfig:
    """Build a ViT5ClassificationNet config (attention backbone).

    Args:
        sequence_mixer_cfg: Override the default ViT5Attention with a custom
            mixer (e.g. a compound parallel adapter).

    Returns:
        LazyConfig for ViT5ClassificationNet.
    """
    if sequence_mixer_cfg is None:
        sequence_mixer_cfg = make_attn_cfg()
    return LazyConfig(ViT5ClassificationNet)(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        readout="corner",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=make_block_cfg(sequence_mixer_cfg),
    )


def make_base_config(
    placement: str,
    num_iters: int,
    lr: float,
    seed: int = 0,
) -> ExperimentConfig:
    """Build the base ExperimentConfig shared across all arms.

    Args:
        placement: Dataset placement mode (``"fixed"`` or ``"random"``).
        num_iters: Total training iterations.
        lr: Peak learning rate.
        seed: Random seed.

    Returns:
        ExperimentConfig with dataset, optimizer, scheduler, and wandb set.
        The caller must set ``config.net`` and may add callbacks.
    """
    config = ExperimentConfig()
    config.seed = seed
    config.debug = False

    config.dataset = make_mnist_dataset_cfg(placement=placement, seed=seed)

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="cross_entropy")

    config.optimizer = LazyConfig(torch.optim.AdamW)(lr=lr, weight_decay=WEIGHT_DECAY)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations=num_iters,
        mode="max",
    )

    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=num_iters,
        grad_clip=1.0,
        precision="32-true",
    )

    config.wandb = WandbConfig(
        entity="clara-discovery",
        job_group="hyena2d_parallel_adapter",
        project="nvsubquadratic",
    )

    return config


def make_hyena2d_inner_mixer_cfg(use_film: bool = False) -> LazyConfig:
    r"""Build the inner QKVSequenceMixer(Hyena) config for the Hyena branch.

    The inner mixer operates at bottleneck width ``RANK``, not ``HIDDEN_DIM``.
    Uses ``fft_padding="zero"`` + ``grid_type="double"`` for non-periodic 2-D
    images (see §4 of brief and TestAssertions in test_ckconv_nd_subq.py).

    Args:
        use_film: When ``True``, attach a
            :class:`~nvsubquadratic.modules.film.KernelFiLMGenerator` so the SIREN
            kernel is input-dependent — i.e. HyenaND as defined in
            arXiv:2607.19378 (``K_i(x) = w(c_i) ⊙ f_θ(c_i; z(x))``).  When
            ``False`` (default, matching the original 196-token run) the kernel is
            **static**, which is a weaker operator and must be reported as such.

    Returns:
        LazyConfig for QKVSequenceMixer wrapping Hyena.
    """
    from nvsubquadratic.modules.film import KernelFiLMGenerator

    film_cfg = (
        LazyConfig(KernelFiLMGenerator)(
            cond_dim=HIDDEN_DIM,
            kernel_hidden_dim=64,
            num_film_layers=3,
            film_hidden_dim=64,
            no_weight_decay=5e-3,
            init_type="identity",
        )
        if use_film
        else None
    )
    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.hyena_nd import Hyena
    from nvsubquadratic.modules.kernels_nd import SIRENKernelND
    from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
    from nvsubquadratic.utils.init import small_init
    from nvsubquadratic.utils.qk_norm import L2Norm

    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=RANK,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=RANK,
                fft_padding="zero",
                grid_type="double",
                is_causal=False,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=RANK,
                    mlp_hidden_dim=64,
                    num_layers=3,
                    embedding_dim=64,
                    omega_0=30.0,
                    L_cache=GRID_W,
                    use_bias=True,
                    hidden_omega_0=1.0,
                    film_cfg=film_cfg,
                    film_after_pos_embed=True,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * RANK,
                out_channels=3 * RANK,
                kernel_size=3,
                groups=3 * RANK,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=RANK),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=RANK),
            qk_norm_cfg=LazyConfig(L2Norm)(),
        ),
        init_method_in=small_init,
    )


def make_parallel_mixer_cfg_c0() -> LazyConfig:
    """Build ViT5ParallelHyenaSequenceMixer config for arm C0 (no inner mixer).

    C0 ablation: same scaffold (W_down → W_up), zero token mixing.
    Tests whether gains in C2 are capacity or inductive bias.

    Returns:
        LazyConfig for ViT5ParallelHyenaSequenceMixer with inner_mixer=Identity.
    """
    from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer

    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=None,  # nn.Identity — no token mixing
    )


def make_parallel_mixer_cfg_c1() -> LazyConfig:
    """Build ViT5ParallelHyenaSequenceMixer config for arm C1 (depthwise conv).

    C1 ablation: local 3×3 depthwise convolutional prior.
    Tests whether gains in C2 require *global* mixing or merely *local* mixing.

    Returns:
        LazyConfig for ViT5ParallelHyenaSequenceMixer with depthwise conv inner.
    """
    from nvsubquadratic.modules.vit5_depthwise_conv2d_mixer import ViT5DepthwiseConv2dMixer
    from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer

    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=LazyConfig(ViT5DepthwiseConv2dMixer)(
            channels=RANK,
            grid_w=GRID_W,
            kernel_size=3,
        ),
    )


def make_parallel_mixer_cfg_c2() -> LazyConfig:
    """Build ViT5ParallelHyenaSequenceMixer config for arm C2 (Hyena2D).

    C2 is the main arm: global implicit filter (translation-equivariant,
    constant along diagonals of the mixing matrix).

    Returns:
        LazyConfig for ViT5ParallelHyenaSequenceMixer with Hyena2D inner.
    """
    from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
    from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer

    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=make_hyena2d_inner_mixer_cfg(),
            grid_w=GRID_W,
        ),
    )


# Keep for backward compat and optional arm L.
make_parallel_hyena_mixer_cfg = make_parallel_mixer_cfg_c2


def make_lora_mixer_cfg(lora_rank: int) -> LazyConfig:
    """Build the ViT5LoRASequenceMixer config (optional arm L).

    Args:
        lora_rank: LoRA rank.

    Returns:
        LazyConfig for ViT5LoRASequenceMixer.
    """
    from examples.hyena2d_parallel_adapter.modules import ViT5LoRASequenceMixer

    return LazyConfig(ViT5LoRASequenceMixer)(
        attn_cfg=make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=lora_rank,
    )


def make_interleaved_net_cfg() -> LazyConfig:
    """Build the from-scratch interleaved Hyena+Attention net config (arm F).

    Uses the existing ``layer_pattern`` / ``layer_types`` mechanism to
    alternate Hyena and Attention blocks: ``"HA" * 3`` for 6 blocks.
    The Hyena blocks here are full-width (hidden_dim=HIDDEN_DIM), not
    bottleneck, so arm F is the from-scratch upper bound on performance.

    Returns:
        LazyConfig for ViT5ClassificationNet with interleaved blocks.
    """
    # Full-width inner mixer for arm F (RANK→HIDDEN_DIM everywhere).
    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.hyena_nd import Hyena
    from nvsubquadratic.modules.kernels_nd import SIRENKernelND
    from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
    from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
    from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
    from nvsubquadratic.utils.qk_norm import L2Norm

    full_inner = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                fft_padding="zero",
                grid_type="double",
                is_causal=False,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=64,
                    num_layers=3,
                    embedding_dim=64,
                    omega_0=30.0,
                    L_cache=GRID_W,
                    use_bias=True,
                    hidden_omega_0=1.0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
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
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM),
            qk_norm_cfg=LazyConfig(L2Norm)(),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )

    hyena_block_cfg = make_block_cfg(
        LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=full_inner,
            grid_w=GRID_W,
        )
    )
    attn_block_cfg = make_block_cfg(make_attn_cfg())

    return LazyConfig(ViT5ClassificationNet)(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        readout="corner",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        layer_pattern="HA" * (NUM_BLOCKS // 2),
        layer_types={"H": hyena_block_cfg, "A": attn_block_cfg},
        padding_types={"H"},
    )
