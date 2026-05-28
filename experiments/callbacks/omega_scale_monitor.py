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

"""Callback to monitor learnable per-row ω₀ scales during training.

Walks the model to find :class:`LearnableOmegaSIRENPositionalEmbeddingND`
modules (used by ``LearnableOmegaSIRENKernelND`` and
``BlockDiagonalLearnableOmegaSIRENKernelND``) and logs a single wandb
``line_series`` chart per block tracking the per-block effective-ω₀
extrema and mean (``omega_0 · scale``) over training.

History is persisted via :meth:`state_dict` / :meth:`load_state_dict` so
charts survive checkpoint resumes.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
import torch


def _empty_history() -> dict[str, list[float]]:
    return {
        "steps": [],
        "omega_eff_min": [],
        "omega_eff_mean": [],
        "omega_eff_max": [],
    }


class OmegaScaleMonitorCallback(pl.Callback):
    """Logs a single chart per block tracking the effective per-row ω₀.

    For each Hyena block whose kernel uses a
    :class:`LearnableOmegaSIRENPositionalEmbeddingND`, we log
    ``omega_eff_min`` / ``omega_eff_mean`` / ``omega_eff_max``, the running
    per-block stats of ``omega_0 · scale`` (post-clamp).  The raw scale
    series is intentionally omitted because the per-block ω₀ values differ
    substantially (typically by ~24×), which would otherwise compress the
    scale axis on shared charts.

    Args:
        log_every_n_steps: How often to log (in global steps).
    """

    def __init__(self, log_every_n_steps: int = 50):  # noqa: D107
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        # block_id -> positional embedding module (the one that owns omega_0_scale)
        self._embeds: dict[int, torch.nn.Module] = {}
        # Accumulated history per block for line_series charts (persisted in checkpoints)
        self._history: dict[int, dict[str, list[float]]] = defaultdict(_empty_history)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D102
        from nvsubquadratic.modules.kernels_nd import (
            LearnableOmegaSIRENPositionalEmbeddingND,
        )

        network = pl_module.network
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod

        for name, module in network.named_modules():
            if not isinstance(module, LearnableOmegaSIRENPositionalEmbeddingND):
                continue
            block_match = re.search(r"blocks\.(\d+)", name)
            if block_match is None:
                continue
            block_id = int(block_match.group(1))
            self._embeds[block_id] = module

        if trainer.is_global_zero:
            print(f"[OmegaScaleMonitor] Found {len(self._embeds)} learnable-ω₀ positional embeddings.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not self._embeds:
            return
        if not trainer.is_global_zero:
            return

        import wandb

        charts: dict[str, "wandb.plot.line_series"] = {}

        for block_id in sorted(self._embeds):
            module = self._embeds[block_id]
            hist = self._history[block_id]
            hist["steps"].append(trainer.global_step)

            with torch.no_grad():
                # Apply the same clamp the forward pre-hook would apply, so
                # the logged stats reflect what the model actually uses.
                scale = (
                    module.omega_0_scale.detach()
                    .to(torch.float32)
                    .clamp(
                        min=module.omega_0_scale_min,
                        max=module.omega_0_scale_max,
                    )
                )
                omega_0 = float(module.omega_0)
                eff_min = omega_0 * float(scale.min())
                eff_mean = omega_0 * float(scale.mean())
                eff_max = omega_0 * float(scale.max())

            hist["omega_eff_min"].append(eff_min)
            hist["omega_eff_mean"].append(eff_mean)
            hist["omega_eff_max"].append(eff_max)

            charts[f"omega_scale/block_{block_id}"] = wandb.plot.line_series(
                xs=hist["steps"],
                ys=[
                    hist["omega_eff_min"],
                    hist["omega_eff_mean"],
                    hist["omega_eff_max"],
                ],
                keys=[
                    "omega_eff_min",
                    "omega_eff_mean",
                    "omega_eff_max",
                ],
                title=f"ω₀ effective block {block_id} (ω₀={omega_0:.3g})",
                xname="step",
            )
        if charts:
            trainer.logger.experiment.log({**charts, "trainer/global_step": trainer.global_step})

    # ------------------------------------------------------------------
    # Checkpoint persistence (Lightning auto-routes via state_key)
    # ------------------------------------------------------------------

    @property
    def state_key(self) -> str:  # noqa: D102
        return "OmegaScaleMonitorCallback"

    def state_dict(self) -> dict[str, Any]:  # noqa: D102
        return {"history": {int(k): dict(v) for k, v in self._history.items()}}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D102
        history = state_dict.get("history", {}) or {}
        self._history = defaultdict(_empty_history)
        for k, v in history.items():
            entry = _empty_history()
            for series_name in entry:
                entry[series_name] = list(v.get(series_name, []))
            self._history[int(k)] = entry
