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

"""Arm B: fine-tune head + LoRA, random placement (parameter-count control for arm C).

Trainable: head (``out_proj``) + LoRA A/B matrices inside each block's mixer.
Architecture: ViT5LoRASequenceMixer wraps ViT5Attention with a rank-r parallel
correction (A zero-initialised, so bit-exact at step 0).

IMPORTANT — parameter matching:
  Before running, compute the trainable adapter parameter count for arm C::

      model_c = instantiate(arm_c_config.net)
      adapter_params = sum(
          p.numel() for n, p in model_c.named_parameters()
          if "w_down" in n or "w_up" in n or "hyena_adapter" in n
      )

  Set ``LORA_RANK`` so ``2 * HIDDEN_DIM * LORA_RANK ≈ adapter_params``.
  Report both counts explicitly in the results table.

Run::

    PYTHONPATH=. python experiments/run.py \
        config_path=examples/hyena2d_parallel_adapter/arm_b.py \
        start_from_checkpoint.run_path=<entity>/<project>/<run_id> \
        seed=0
"""

from examples.hyena2d_parallel_adapter._base import (
    FINETUNE_ITERS,
    HIDDEN_DIM,
    LR_FINETUNE,
    make_base_config,
    make_block_cfg,
    make_lora_mixer_cfg,
)
from examples.hyena2d_parallel_adapter.modules import RemapPretrainKeys
from experiments.default_cfg import ExperimentConfig, StartFromCheckpointConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


# Tune this so 2 * HIDDEN_DIM * LORA_RANK matches arm C's adapter param count.
# Starting estimate: arm C ~31k adapter params → LORA_RANK ≈ 120.
# Re-measure and adjust before the final run.
LORA_RANK = 120


def get_config(seed: int = 0) -> ExperimentConfig:
    """Return the arm B fine-tune config (head + LoRA).

    Args:
        seed: Random seed.

    Returns:
        ExperimentConfig for arm B.
    """
    from examples.hyena2d_parallel_adapter._base import (
        IMAGE_SIZE,
        IN_CHANNELS,
        NUM_BLOCKS,
        NUM_CLASSES,
        PATCH_SIZE,
    )

    config = make_base_config(
        placement="random",
        num_iters=FINETUNE_ITERS,
        lr=LR_FINETUNE,
        seed=seed,
    )

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=0,
        readout="corner",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=make_block_cfg(make_lora_mixer_cfg(lora_rank=LORA_RANK)),
    )

    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        alias="best",
        strict=False,
        partial_load=True,
        run_path="",  # Set via CLI
        callbacks=[LazyConfig(RemapPretrainKeys)()],
    )

    config.comment = f"arm_b_lora_rank{LORA_RANK}"
    return config
