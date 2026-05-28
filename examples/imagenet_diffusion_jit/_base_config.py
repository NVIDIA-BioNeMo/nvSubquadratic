# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared experiment-config builder for the JiT-faithful diffusion configs.

Wraps :mod:`nvsubquadratic.networks.jit` (already a direct port of
``LTH14/JiT``) with the **training recipe taken verbatim from the JiT paper
and reference scripts** (``~/projects/JiT/main_jit.py``,
``~/projects/JiT/util/lr_sched.py``, ``~/projects/JiT/util/misc.py``,
``~/projects/JiT/engine_jit.py``).

Recipe (all defaults below match JiT's 600-epoch ImageNet-1k script):

============================  =====================================================
Item                          Value (source)
============================  =====================================================
Optimizer                     ``AdamW(betas=(0.9, 0.95))``  (main_jit.py L187)
Weight decay                  ``0.0``                       (main_jit.py L47)
Base LR                       ``blr = 5e-5``                (main_jit.py L41)
Absolute LR                   ``lr = blr * eff_bs / 256``   (main_jit.py L175-176)
LR schedule                   linear warmup -> constant     (util/lr_sched.py)
Warmup                        5 epochs                      (main_jit.py L35)
Epochs                        600 (paper script)            (README scripts)
EMA decay                     0.9999 (primary, for sampling)(main_jit.py L49)
EMA warmup steps              0 (starts at step 0)          (denoiser.update_ema)
Mixed precision               bf16                          (engine_jit.py L37,109)
Effective batch size          1024 (= 128/GPU * 8 GPUs)     (README scripts)
P_mean / P_std                -0.8 / 0.8                    (denoiser.py L23-24)
noise_scale                   1.0 (256px), 2.0 (512px)      (README scripts)
t_eps                         5e-2 (clamp on 1-t in v calc) (denoiser.py L25)
Label dropout (cond_dropout)  0.1                           (denoiser.py L22)
Heun sampler steps            50                            (main_jit.py L71)
CFG interval                  [0.1, 1.0]                    (README scripts)
CFG (train)                   2.9 (B), 2.2 (H)              (README scripts)
CFG (eval)                    3.0 (B), 2.4 (L), 2.2 (H)     (README scripts)
============================  =====================================================

Wrapper-level deviations (see this package's ``__init__.py`` for details):

- ``DiffusionWrapper`` keeps **one** EMA at the configured ``ema_decay``,
  not two (JiT tracks 0.9999 + 0.9996; only 0.9999 is used for sampling, so
  a single 0.9999 EMA matches sampling behaviour).
- Wrapper's time MLP has slightly different inner widths than JiT's
  ``TimestepEmbedder``: ``Linear(timestep_dim, hidden*2) -> SiLU ->
  Linear(hidden*2, hidden)`` vs JiT's
  ``Linear(256, hidden) -> SiLU -> Linear(hidden, hidden)``.  Same function
  space, ~1.2M-param difference for B/16.
- Wrapper uses ``target_v = x - eps`` (unclamped) vs JiT's clamped target.
  Differs only when t > 0.95 (~6% of training samples); the gradient
  direction is identical, only the magnitude near t->1 differs slightly.

Bias / norm weight-decay grouping
---------------------------------
JiT's ``util/misc.add_weight_decay`` places ``param.shape == 1``-d
parameters (biases + RMSNorm weights) in a no-decay group.  Our project's
:func:`experiments.lightning_wrappers.base_lightning_wrapper._build_param_groups`
honours a per-parameter ``_no_weight_decay`` attribute and emits a warning
for 1-d params that lack the flag.  The existing JiT model implementation
at :mod:`nvsubquadratic.networks.jit` does **not** tag these parameters;
this is harmless when ``weight_decay = 0`` (the JiT default) because every
param ends up with effective wd=0 regardless of grouping.  Leave the flag
unset for now; if you ever set ``weight_decay > 0``, walk the model after
construction and set ``param._no_weight_decay = True`` for every 1-d
parameter.
"""

import os
from typing import Callable

import torch

from experiments.datamodules._deprecated.ref_imagenet import ImageNetDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig


__all__ = ["get_base_config"]


# ─── Defaults (override per leaf config) ─────────────────────────────────────
WANDB_ENTITY = "dafidofff"
WANDB_PROJECT = "nvsubquadratic"
NUM_CLASSES = 1000

HF_DATASET_DEFAULT = os.environ.get("IMAGENET_HF_DATASET", "imagenet-1k")
HF_CACHE_DEFAULT = os.environ.get("IMAGENET_PATH", "/scratch-shared/dknigge/hf_cache")

# ─── JiT recipe constants ────────────────────────────────────────────────────
IMAGENET_TRAIN_SIZE = 1_281_167
JIT_BLR = 5e-5  # main_jit.py default
JIT_WEIGHT_DECAY = 0.0  # main_jit.py default
JIT_EPOCHS = 600  # README scripts
JIT_WARMUP_EPOCHS = 5  # main_jit.py default
JIT_EMA_DECAY = 0.9999  # main_jit.py ema_decay1 (used for sampling)
# JiT also tracks a SECOND EMA at 0.9996 ("ema_decay2") that is checkpointed
# but never used for sampling in the released eval scripts.  We track it for
# parity / Karras-style post-hoc EMA experiments; set to ``None`` in any leaf
# config to disable (and save ~model-size of extra GPU memory).
JIT_EMA_DECAY_SECONDARY = 0.9996
JIT_BETAS = (0.9, 0.95)  # main_jit.py L187
JIT_PRECISION = "bf16-mixed"  # engine_jit.py L37 / L109

# Continuous-time diffusion / sampling defaults — all match the wrapper's
# JiT-style defaults, but we set them explicitly for clarity.
JIT_P_MEAN = -0.8
JIT_P_STD = 0.8
JIT_T_EPS = 5e-2  # informational; clamp is hardcoded to 0.05 in the wrapper
JIT_NUM_TRAIN_TIMESTEPS = 1000
JIT_NUM_INFERENCE_STEPS = 50  # Heun + last-Euler (handled by the wrapper)
JIT_CONDITION_DROPOUT_PROB = 0.1
JIT_CFG_INTERVAL = (0.1, 1.0)
JIT_NUM_SAMPLES = 16
JIT_LOG_SAMPLES = True


def _absolute_lr(blr: float, eff_batch_size: int) -> float:
    """JiT's LR rule: ``lr = blr * eff_bs / 256`` (``main_jit.py`` L175-176)."""
    return blr * eff_batch_size / 256


def _iters_per_epoch(eff_batch_size: int) -> int:
    """Iterations per epoch on full ImageNet-1k at the given effective batch."""
    return IMAGENET_TRAIN_SIZE // eff_batch_size


def get_base_config(
    *,
    model_factory: Callable,
    image_size: int,
    batch_size: int,
    num_gpus: int = 8,
    accumulate_grad_steps: int = 1,
    num_workers: int | None = None,
    blr: float = JIT_BLR,
    weight_decay: float = JIT_WEIGHT_DECAY,
    epochs: int = JIT_EPOCHS,
    warmup_epochs: int = JIT_WARMUP_EPOCHS,
    ema_decay: float = JIT_EMA_DECAY,
    noise_scale: float = 1.0,
    guidance_scale: float = 2.9,
    proj_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    fid_stats_file: str = "",
    fid_online_jit: bool = False,
    hf_dataset: str = HF_DATASET_DEFAULT,
    hf_cache: str = HF_CACHE_DEFAULT,
    job_group: str = "imagenet_diffusion_jit",
    extra_tags: list | None = None,
) -> DiffusionExperimentConfig:
    """Build a JiT-faithful ``DiffusionExperimentConfig`` ready for training.

    ``model_factory`` should be one of the factories from
    :mod:`nvsubquadratic.networks.jit` (e.g. ``JiT_B_16``, ``JiT_L_16``,
    ``JiT_H_16``).  The factory's signature is ``factory(**kwargs) -> JiT``
    where the kwargs forward to :class:`~nvsubquadratic.networks.jit.JiT`.

    Args:
        model_factory: One of ``JiT_B_16`` / ``JiT_L_16`` / ``JiT_H_16``
            (etc.) from :mod:`nvsubquadratic.networks.jit`.  Wrapped in a
            :class:`LazyConfig` and instantiated with ``input_size``,
            ``num_classes``, ``attn_drop``, ``proj_drop``.  The factory's
            hard-coded ``depth`` / ``hidden_size`` / ``num_heads`` /
            ``bottleneck_dim`` / ``in_context_*`` / ``patch_size`` are kept.
        image_size: Spatial side (256 or 512 for the JiT paper).
        batch_size: Per-GPU batch size.  JiT's reference scripts use 128.
        num_gpus: Number of GPUs the recipe targets.  Used purely to compute
            the effective batch size and from it the absolute LR / iteration
            count; the actual GPU count at training time is read from the
            launcher and can differ from this value (the LR is fixed in
            :meth:`get_base_config` at config-build time).
        accumulate_grad_steps: Optimizer accumulation.  Use to recover the
            JiT effective batch of 1024 when running on fewer GPUs.
        num_workers: Dataloader worker count.  Defaults to ``min(12, ncpu-2)``.
        blr: Base LR before batch-size scaling.  JiT default ``5e-5``.
        weight_decay: AdamW weight decay.  JiT default ``0.0``.
        epochs: Training epochs.  JiT paper / README scripts use ``600``;
            the argparser default is ``200``.
        warmup_epochs: Linear-warmup phase length.  JiT default ``5``.
        ema_decay: EMA decay (single EMA in our wrapper).  JiT uses
            ``0.9999`` for its primary sampling EMA, so default to that.
        noise_scale: Initial noise scale used for both the training z_t mix
            and the sampling initialisation.  JiT uses ``1.0`` for 256px,
            ``2.0`` for 512px.
        guidance_scale: Classifier-free guidance scale.  Per-size JiT
            defaults: 2.9 (B), 2.4 (L, eval), 2.2 (H).
        proj_dropout: Projection dropout inside ``Attention`` and the
            SwiGLU MLP (see :func:`nvsubquadratic.networks.jit._make_swiglu_mlp`).
            JiT default 0.0 (B/L), 0.2 (H).  Applied only to the middle
            half of the block stack (see ``JiT.__init__``).
        attn_dropout: Attention dropout (applied only to middle blocks).
            JiT keeps this at 0.0 for all sizes.
        fid_stats_file: Path to a ``.npz`` of pre-computed FID statistics
            (JiT-style online eval).  Set ``fid_online_jit=True`` to enable.
            JiT's reference stats live under
            ``~/projects/JiT/fid_stats/jit_in{256,512}_stats.npz``; we leave
            the path empty by default.
        fid_online_jit: Whether to run the JiT-style online FID evaluation
            during validation epochs.
        hf_dataset: HuggingFace dataset name (e.g. ``"imagenet-1k"``).
        hf_cache: Local cache directory for the HuggingFace dataset.
        job_group: WandB job group.
        extra_tags: Extra WandB tags appended to the default set.

    Returns:
        A fully-populated ``DiffusionExperimentConfig`` whose ``config.net``
        is the requested JiT factory wrapped in a :class:`LazyConfig`.
    """
    if num_workers is None:
        num_workers = min(12, (os.cpu_count() or 4) - 2)

    eff_batch_size = batch_size * num_gpus * accumulate_grad_steps
    learning_rate = _absolute_lr(blr, eff_batch_size)
    iters_per_epoch = _iters_per_epoch(eff_batch_size)
    total_iterations = epochs * iters_per_epoch
    warmup_iterations_pct = warmup_epochs / epochs

    config = DiffusionExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir=hf_cache,
        hf_dataset_name=hf_dataset,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        seed=42,
        task="generation",  # normalises pixels to [-1, 1] (matches JiT)
        drop_labels=False,  # required for class conditioning + in-context tokens
        # JiT trains on deterministic ADM-style center crops (no scale
        # jitter); ``"random_resized"`` (the legacy default) is a strong
        # classification-style augmentation that JiT does not use.
        train_crop_mode="center",
    )

    # The JiT model factory is wrapped in LazyConfig — it accepts forwarded
    # kwargs that override the factory's optional args.  ``depth``,
    # ``hidden_size``, ``num_heads``, ``bottleneck_dim``, ``in_context_*``,
    # and ``patch_size`` are baked into the factory itself and are NOT
    # passed here so we don't risk drifting from the JiT spec.
    config.net = LazyConfig(model_factory)(
        input_size=image_size,
        num_classes=NUM_CLASSES,
        attn_drop=attn_dropout,
        proj_drop=proj_dropout,
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=JIT_BETAS,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=total_iterations,
        grad_clip=0.0,  # JiT does not clip gradients (no clipping in engine_jit.py)
        accumulate_grad_steps=accumulate_grad_steps,
        precision=JIT_PRECISION,
    )

    config.scheduler = SchedulerConfig(
        name="constant",
        warmup_iterations_percentage=warmup_iterations_pct,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.diffusion = DiffusionConfig(
        num_train_timesteps=JIT_NUM_TRAIN_TIMESTEPS,
        num_inference_steps=JIT_NUM_INFERENCE_STEPS,
        num_samples=JIT_NUM_SAMPLES,
        log_samples=JIT_LOG_SAMPLES,
        ema_enabled=True,
        ema_decay=ema_decay,
        ema_update_every=1,
        ema_warmup_steps=0,  # JiT starts EMA from step 0
        ema_decay_secondary=JIT_EMA_DECAY_SECONDARY,
        noise_scale=noise_scale,
        use_classifier_free_guidance=True,
        guidance_scale=guidance_scale,
        condition_dropout_prob=JIT_CONDITION_DROPOUT_PROB,
        num_classes=NUM_CLASSES,
        p_mean=JIT_P_MEAN,
        p_std=JIT_P_STD,
        cfg_interval_start=JIT_CFG_INTERVAL[0],
        cfg_interval_end=JIT_CFG_INTERVAL[1],
        # JiT-exact knobs (see the wrapper-level deviations note in
        # ``__init__.py``).  Together these flip ``DiffusionWrapper`` from
        # legacy mode to byte-exact JiT replication:
        #   - ``clamp_target_v=True``: match JiT's v-loss target clamping.
        #   - ``t_eps_clamp=JIT_T_EPS``: matches JiT's ``--t_eps 5e-2``.
        #   - ``network_handles_conditioning=True``: bypass wrapper's
        #     time MLP / label embed; JiT's own ``TimestepEmbedder`` /
        #     ``LabelEmbedder`` are used instead (matches JiT shapes,
        #     also brings them into the EMA tracker which the legacy
        #     wrapper-side embedders were not).
        clamp_target_v=True,
        t_eps_clamp=JIT_T_EPS,
        network_handles_conditioning=True,
        fid_online_jit=fid_online_jit,
        fid_stats_file=fid_stats_file,
    )

    tags = ["jit", "diffusion", f"imagenet{image_size}", "exact-jit"]
    if extra_tags:
        tags.extend(extra_tags)
    config.wandb = WandbConfig(
        job_group=job_group,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        tags=tags,
    )

    return config
