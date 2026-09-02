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

"""Experiment-local modules for the parallel Hyena adapter study.

Contains:
- ``ViT5LoRASequenceMixer`` — attention + zero-init low-rank parallel branch (arm B control).
- ``remap_pretrain_keys`` — state-dict callable for arms whose architecture adds
  a nesting level (``sequence_mixer.*`` → ``sequence_mixer.attn.*``).
"""

import math

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5LoRASequenceMixer(nn.Module):
    """Attention plus a zero-init rank-r parallel correction (arm B baseline).

    Computes::

        y = Attn(x) + B(A(x))

    where ``A: C → r`` (Kaiming init) and ``B: r → C`` (zero init).
    ``B`` zero-init guarantees bit-exact identity at step 0, matching arm C's
    zero-init guarantee for ``W_up``.

    This is the *trainable parameter count* baseline for arm C: tune ``rank``
    so ``2 * hidden_dim * rank ≈ param_count(arm_C_adapter)``.

    Args:
        attn_cfg: LazyConfig for the attention module.
        hidden_dim: Transformer hidden dimension.
        rank: LoRA rank ``r``.
    """

    def __init__(self, attn_cfg: LazyConfig, hidden_dim: int, rank: int) -> None:
        """Build the attention module and the zero-init LoRA branch (see class docstring)."""
        super().__init__()
        self.attn = instantiate(attn_cfg)
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Return ``Attn(x) + lora_B(lora_A(x))``."""
        return self.attn(x) + self.lora_B(self.lora_A(x))

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """FLOPs = attention + 2 * T * C * rank (both linear projections)."""
        flops = 0
        if hasattr(self.attn, "flop_count"):
            flops += self.attn.flop_count(num_tokens, inference=inference)
        flops += 4 * num_tokens * self.hidden_dim * self.rank
        return flops

    def extra_repr(self) -> str:
        """Show hidden dim and LoRA rank in ``repr(module)``."""
        return f"hidden_dim={self.hidden_dim}, rank={self.rank}"


class RemapPretrainKeys:
    """Callable that remaps pretrain keys for arms that nest ViT5Attention inside a compound mixer.

    Pretrain keys:    ``network.blocks.N.sequence_mixer.<param>``
    Arm C0/C1/C2/E:  ``network.blocks.N.sequence_mixer.attn.<param>``

    Used in ``StartFromCheckpointConfig.callbacks``.  Must be a class (not a
    bare function) so that ``LazyConfig`` can instantiate it before ``run.py``
    calls the instance with the state dict.
    """

    def __call__(self, state_dict: dict, **_kwargs) -> dict:
        """Remap sequence_mixer keys to sequence_mixer.attn keys.

        Args:
            state_dict: Checkpoint state dict.
            **_kwargs: Extra kwargs passed by run.py (model, config, etc.) — ignored.

        Returns:
            Updated state dict with remapped keys.
        """
        new_sd = {}
        for k, v in state_dict.items():
            if ".sequence_mixer." in k and ".sequence_mixer.attn." not in k:
                new_k = k.replace(".sequence_mixer.", ".sequence_mixer.attn.")
                new_sd[new_k] = v
            else:
                new_sd[k] = v
        return new_sd


# Convenience alias kept for any direct call sites.
remap_pretrain_keys = RemapPretrainKeys()
