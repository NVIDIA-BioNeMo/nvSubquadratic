"""Callback to monitor modulation-mask parameters during training.

Walks the model to find ``GaussianModulationND`` and ``ExponentialModulationND``
modules, then logs a single wandb ``line_series`` chart per block showing
the min and max of the effective parameter over training.

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
    return {"steps": [], "min": [], "max": []}


class MaskMonitorCallback(pl.Callback):
    """Logs a single min/max chart per block for every mask module.

    Args:
        log_every_n_steps: How often to log (in global steps).
    """

    def __init__(self, log_every_n_steps: int = 50):  # noqa: D107
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        # Populated in on_fit_start: block_id -> (module, kind)
        self._masks: dict[int, tuple[torch.nn.Module, str]] = {}
        # Accumulated history per block for line_series charts (persisted in checkpoints)
        self._history: dict[int, dict[str, list[float]]] = defaultdict(_empty_history)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D102
        from nvsubquadratic.modules.masks_nd import (
            ExponentialModulationND,
            GaussianModulationND,
        )

        network = pl_module.network
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod

        for name, module in network.named_modules():
            block_match = re.search(r"blocks\.(\d+)", name)
            if block_match is None:
                continue
            block_id = int(block_match.group(1))

            if isinstance(module, GaussianModulationND):
                self._masks[block_id] = (module, "gaussian")
            elif isinstance(module, ExponentialModulationND):
                self._masks[block_id] = (module, "exponential")

        if trainer.is_global_zero:
            kinds = [k for _, k in self._masks.values()]
            print(f"[MaskMonitor] Found {len(self._masks)} mask modules: {kinds}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not self._masks:
            return
        if not trainer.is_global_zero:
            return

        import wandb

        charts: dict[str, wandb.plot.line_series] = {}

        for block_id in sorted(self._masks):
            module, kind = self._masks[block_id]
            hist = self._history[block_id]
            hist["steps"].append(trainer.global_step)

            with torch.no_grad():
                if kind == "gaussian":
                    std = module._compute_std()  # [data_dim, num_channels]
                    hist["min"].append(std.min().item())
                    hist["max"].append(std.max().item())
                    label = "std"
                else:  # exponential
                    decay = module.weight.float().abs()  # [data_dim, num_channels]
                    hist["min"].append(decay.min().item())
                    hist["max"].append(decay.max().item())
                    label = "decay"

            charts[f"mask/block_{block_id}"] = wandb.plot.line_series(
                xs=hist["steps"],
                ys=[hist["min"], hist["max"]],
                keys=[f"{label}_min", f"{label}_max"],
                title=f"Mask block {block_id}",
                xname="step",
            )
        if charts:
            trainer.logger.experiment.log({**charts, "trainer/global_step": trainer.global_step})

    # ------------------------------------------------------------------
    # Checkpoint persistence (Lightning auto-routes via state_key)
    # ------------------------------------------------------------------

    @property
    def state_key(self) -> str:  # noqa: D102
        return "MaskMonitorCallback"

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
