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

"""Callback that logs ``‖W_up‖_F`` per layer for the parallel Hyena adapter experiment.

:class:`WUpNormMonitorCallback` walks the network at fit-start, discovers all
:class:`~nvsubquadratic.modules.vit5_parallel_hyena_adapter.ViT5ParallelHyenaSequenceMixer`
instances, records their block depths from the module path, and logs the
Frobenius norm of each ``w_up.weight`` every ``log_every_n_steps`` training
steps.

This answers the question: "at which depths does the model actually want global
conv mixing?"  If the norm at a given layer remains near zero throughout
training, that layer preferred to route purely through attention.

The monitoring pattern follows
:mod:`experiments.callbacks.omega_scale_monitor`.

Usage::

    config.callbacks += [
        LazyConfig(WUpNormMonitorCallback)(log_every_n_steps=200),
    ]
"""

import re

import pytorch_lightning as pl
import torch


class WUpNormMonitorCallback(pl.Callback):
    r"""Log Frobenius norm of ``W_up`` for each parallel Hyena adapter layer.

    Discovery happens once in ``on_fit_start``: all modules in
    ``trainer.model.network`` whose class is
    ``ViT5ParallelHyenaSequenceMixer`` are collected and keyed by their
    ``blocks.<i>`` depth extracted from the module path.

    Logging happens every ``log_every_n_steps`` global training steps on the
    global-zero rank.  For each discovered layer, logs::

        wup_norm/block_{i}  —  ‖w_up.weight‖_F

    Args:
        log_every_n_steps: Logging interval in global training steps.
    """

    def __init__(self, log_every_n_steps: int = 200) -> None:
        """Store logging interval."""
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._adapters: dict[int, torch.nn.Module] = {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Discover all ViT5ParallelHyenaSequenceMixer instances in the network.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: The Lightning module wrapping the network.
        """
        # Import here to avoid circular dependency at module load time.
        from nvsubquadratic.modules.vit5_parallel_hyena_adapter import ViT5ParallelHyenaSequenceMixer

        self._adapters = {}
        network = getattr(pl_module, "network", pl_module)
        for name, module in network.named_modules():
            if isinstance(module, ViT5ParallelHyenaSequenceMixer):
                m = re.search(r"blocks\.(\d+)", name)
                block_id = int(m.group(1)) if m else len(self._adapters)
                self._adapters[block_id] = module

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Log ``‖w_up.weight‖_F`` for each adapter layer.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: The Lightning module.
            outputs: Training step outputs (unused).
            batch: Current batch (unused).
            batch_idx: Batch index within the epoch (unused).
        """
        if not self._adapters:
            return
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        for block_id, adapter in self._adapters.items():
            norm = adapter.w_up.weight.data.norm().item()
            pl_module.log(
                f"wup_norm/block_{block_id}",
                norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                rank_zero_only=True,
            )

    def state_dict(self) -> dict:
        """Persist nothing — norms are recomputed from live weights on resume."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """No-op — nothing to restore."""
