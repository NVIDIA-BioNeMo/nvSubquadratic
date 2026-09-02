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

"""Large-scale (784-token) variant of the parallel adapter experiment.

Fixes the two confounds identified in the first run
(``reports/hyena2d_parallel_adapter/REPORT.md``):

1. **Grid too small.** The first run used a 14×14 = 196-token grid, where a
   stack of 3×3 convolutions already spans the whole grid within a few layers,
   so global mixing had nothing to buy.  Here ``canvas_size=112`` and
   ``patch_size=4`` give a **28×28 = 784-token** grid — a 3×3 conv needs ~14
   sequential hops to cross it, more than the 6 available blocks.

2. **C2 was not HyenaND.** The first run built the Hyena branch with
   ``film_cfg=None``, i.e. a *static* implicit long convolution.  Wessels et al.
   (arXiv:2607.19378) define the operator as ``K_i(x) = w(c_i) ⊙ f_θ(c_i; z(x))``
   — input dependence via FiLM is constitutive.  Arm ``c2f`` enables it;
   arm ``c2s`` keeps the static kernel as an explicit ablation.

Arms in this variant::

    a    head only                              (lower bound)
    c0   bottleneck, no mixer                   (capacity)
    c1   bottleneck + depthwise 3x3 conv        (local prior)
    c2s  bottleneck + Hyena2D, static kernel    (global, no input dependence)
    c2f  bottleneck + Hyena2D, FiLM kernel      (HyenaND proper)

``c2s`` vs ``c2f`` is a FiLM-on/off ablation at matched everything-else, which
neither the paper nor the rest of this repo currently contains.
"""

import torch

from examples.hyena2d_parallel_adapter._base import (
    BATCH_SIZE,
    HIDDEN_DIM,
    IN_CHANNELS,
    NUM_BLOCKS,
    NUM_CLASSES,
    NUM_HEADS,
    RANK,
    WEIGHT_DECAY,
    _num_workers,
)
from experiments.datamodules.mnist import MNISTDataModule
from experiments.datamodules.spatial_recall_classification import (
    SpatialRecallClassificationDataModule,
)
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_depthwise_conv2d_mixer import ViT5DepthwiseConv2dMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Scale ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 112
CANVAS_SIZE = 112
PATCH_SIZE = 4
GRID_W = IMAGE_SIZE // PATCH_SIZE  # 28
NUM_PATCHES = GRID_W * GRID_W  # 784
TARGET_SIZE = 16  # 4 patches exactly
READOUT_VALUE = -1.0
HEAD_DIM = HIDDEN_DIM // NUM_HEADS

# ── SIREN kernel ────────────────────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 64
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 64
# omega_0 scales with grid resolution (see reports/siren_omega0_dimensional_scaling).
# The 196-token run used 30.0 at grid 14; grid 28 is a 2x resolution step.
KERNEL_OMEGA_0 = 60.0

# ── FiLM (matches the v3 `peeaqdkq` winner: identity init, wd 5e-3) ─────────
FILM_HIDDEN_DIM = 64
FILM_AFTER_POS_EMBED = True
FILM_NUM_LAYERS = KERNEL_NUM_LAYERS  # 2 hidden + 1 on the positional sine
FILM_INIT_TYPE = "identity"
FILM_WEIGHT_DECAY = 5e-3

# ── Training ────────────────────────────────────────────────────────────────
PRETRAIN_ITERS = 40_000
FINETUNE_ITERS = 25_000
LR_PRETRAIN = 3e-4
LR_FINETUNE = 1e-4


def make_dataset_cfg(placement: str, seed: int = 0) -> LazyConfig:
    """Build the 112x112 spatial-recall classification datamodule config.

    Args:
        placement: ``"fixed"`` (pretrain) or ``"random"`` (fine-tune).
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


def make_attn_cfg() -> LazyConfig:
    """Build the ViT5Attention config for the 28x28 grid (no CLS, no registers)."""
    return LazyConfig(ViT5Attention)(
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_patches_h=GRID_W,
        num_patches_w=GRID_W,
        num_registers=0,
        has_cls=False,
        qk_norm=LazyConfig(RMSNorm)(dim=HEAD_DIM, eps=1e-6),
        rope_base=10000.0,
        reg_rope_base=100.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=False,
        out_proj_bias=False,
    )


def make_hyena_inner_cfg(use_film: bool) -> LazyConfig:
    r"""Build the bottleneck-width Hyena2D inner mixer config.

    Uses ``fft_padding="zero"`` + ``grid_type="double"`` (images are not
    periodic) and ``is_causal=False`` (2-D is non-causal by construction).

    Args:
        use_film: When ``True``, attach a
            :class:`~nvsubquadratic.modules.film.KernelFiLMGenerator` to the SIREN
            kernel, making it input-dependent — i.e. HyenaND as defined in
            arXiv:2607.19378.  When ``False`` the kernel is **static**, which is
            a different (weaker) operator and must be reported as such.
            ``cond_dim`` is ``HIDDEN_DIM`` because the adapter pools ``z(x)``
            from the pre-bottleneck token stream.

    Returns:
        LazyConfig for QKVSequenceMixer wrapping Hyena at width ``RANK``.
    """
    film_cfg = (
        LazyConfig(KernelFiLMGenerator)(
            cond_dim=HIDDEN_DIM,
            kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=FILM_NUM_LAYERS,
            film_hidden_dim=FILM_HIDDEN_DIM,
            no_weight_decay=FILM_WEIGHT_DECAY,
            init_type=FILM_INIT_TYPE,
        )
        if use_film
        else None
    )

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
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=GRID_W,
                    use_bias=True,
                    hidden_omega_0=1.0,
                    film_cfg=film_cfg,
                    film_after_pos_embed=FILM_AFTER_POS_EMBED,
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


def make_mixer_cfg(arm: str) -> LazyConfig:
    r"""Build the sequence-mixer config for a given arm.

    Args:
        arm: One of ``"a"`` (plain attention), ``"c0"`` (bottleneck only),
            ``"c1"`` (depthwise conv), ``"c2s"`` (static Hyena), ``"c2f"``
            (FiLM-conditioned Hyena / HyenaND).

    Returns:
        LazyConfig for the block's ``sequence_mixer``.

    Raises:
        ValueError: If ``arm`` is unrecognised.
    """
    if arm == "a":
        return make_attn_cfg()

    if arm == "c0":
        inner, cond = None, "none"
    elif arm == "c1":
        inner = LazyConfig(ViT5DepthwiseConv2dMixer)(channels=RANK, grid_w=GRID_W, kernel_size=3)
        cond = "none"
    elif arm in ("c2s", "c2f"):
        inner = LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=make_hyena_inner_cfg(use_film=(arm == "c2f")),
            grid_w=GRID_W,
        )
        cond = "token_mean" if arm == "c2f" else "none"
    else:
        raise ValueError(f"Unknown arm {arm!r}; expected one of a, c0, c1, c2s, c2f")

    return LazyConfig(ViT5ParallelHyenaSequenceMixer)(
        attn_cfg=make_attn_cfg(),
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        inner_mixer_cfg=inner,
        conditioning_source=cond,
    )


def make_net_cfg(arm: str) -> LazyConfig:
    """Build the full ViT5ClassificationNet config for an arm.

    Args:
        arm: Arm identifier — see :func:`make_mixer_cfg`.

    Returns:
        LazyConfig for ViT5ClassificationNet with ``readout="corner"``.
    """
    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=make_mixer_cfg(arm),
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
        block_cfg=block_cfg,
    )


def make_config(placement: str, num_iters: int, lr: float, seed: int) -> ExperimentConfig:
    """Build the shared ExperimentConfig scaffold (dataset, optimiser, schedule).

    Args:
        placement: ``"fixed"`` or ``"random"``.
        num_iters: Total training iterations.
        lr: Peak learning rate.
        seed: Random seed.

    Returns:
        ExperimentConfig with ``net`` left unset for the caller.
    """
    config = ExperimentConfig()
    config.seed = seed
    config.debug = False
    config.dataset = make_dataset_cfg(placement=placement, seed=seed)
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
        job_group="hyena2d_parallel_adapter_784",
        project="nvsubquadratic",
    )
    return config
