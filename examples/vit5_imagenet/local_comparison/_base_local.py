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


"""Local single-GPU training recipe for patch-merging comparison.

Wraps the cluster recipe from v5._base and patches it for single RTX 3090 (24 GB):
  - batch_size=64 per GPU, accumulate_grad_steps=32 → effective batch 2048 (unchanged)
  - ITERS_PER_EPOCH stays at 1_281_167 // 2048 = 625 (same optimizer steps per epoch)
  - compile disabled (easier local debugging; re-enable to measure throughput)
  - check_val_every_n_epoch=5 (saves time vs the cluster default of 4)

All cluster-environment imports (apex, lightning EMA callback) are deferred to
call time so this module can be imported without those dependencies.

Usage::

    from examples.vit5_imagenet.local_comparison._base_local import get_local_base_config
    config = get_local_base_config(epochs=100)
    config.net = ...
"""

from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


# ── Local recipe constants ────────────────────────────────────────────────────

BATCH_SIZE_LOCAL = 64
ACCUMULATE_GRAD_STEPS = 32  # 64 × 32 = 2048 effective (matches cluster)
LOCAL_EPOCHS = 100
IMAGENET_TRAIN_SIZE = 1_281_167
EFFECTIVE_BATCH_SIZE = BATCH_SIZE_LOCAL * ACCUMULATE_GRAD_STEPS  # 2048
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE  # 625 — same as cluster


def get_local_base_config(
    epochs: int = LOCAL_EPOCHS,
    **kwargs,
) -> ExperimentConfig:
    """Return a single-GPU-patched ExperimentConfig.

    Calls v5._base.get_base_config(epochs=epochs, **kwargs) for the full recipe,
    then overrides batch-size / accumulation / compile / val-frequency fields.

    Set IMAGENET_PATH and IMAGENET_FOLDER_PATH env vars before running.
    """
    # Defer cluster-env imports to call time (apex, lightning EMA).
    from examples.vit5_imagenet.v5._base import (
        get_base_config as _cluster_cfg,
    )
    from experiments.callbacks.iteration_speed import IterationSpeedCallback
    from experiments.callbacks.model_ema import LabeledEMAWeightAveraging

    config: ExperimentConfig = _cluster_cfg(epochs=epochs, **kwargs)

    # ── Batch size ─────────────────────────────────────────────────────────────
    # config.dataset is an OmegaConf DictConfig; mutate in place.
    config.dataset.batch_size = BATCH_SIZE_LOCAL

    # ── Gradient accumulation ──────────────────────────────────────────────────
    # config.train is a plain TrainConfig dataclass.
    config.train.accumulate_grad_steps = ACCUMULATE_GRAD_STEPS

    # ── Compile ────────────────────────────────────────────────────────────────
    config.compile = False
    config.compile_mode = None

    # ── Validation frequency ───────────────────────────────────────────────────
    config.trainer.check_val_every_n_epoch = 5

    # ── Callbacks: rebuild with correct local batch size ───────────────────────
    ema_decay = 0.99996  # matches cluster default
    config.callbacks = [
        LazyConfig(LabeledEMAWeightAveraging)(decay=ema_decay),
        LazyConfig(IterationSpeedCallback)(
            log_every_n_steps=50,
            batch_size_per_gpu=BATCH_SIZE_LOCAL,
        ),
    ]

    # ── WandB ─────────────────────────────────────────────────────────────────
    config.wandb.job_group = "local_comparison"

    return config
