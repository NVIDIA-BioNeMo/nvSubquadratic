"""Callback to measure and log iteration throughput.

Logs ``perf/iter_per_sec`` and ``perf/samples_per_sec`` to wandb at regular
intervals using a sliding-window average to smooth out variance from
validation steps and checkpointing pauses.
"""

from __future__ import annotations

import time

import pytorch_lightning as pl


class IterationSpeedCallback(pl.Callback):
    """Logs iteration throughput as wandb scalars.

    Args:
        log_every_n_steps: How often to log speed metrics.
        window_size: Number of recent batch times to average over.
        batch_size_per_gpu: Batch size on each GPU (for samples/sec calc).
            If ``None``, attempts to read from ``trainer.datamodule``.
    """

    def __init__(  # noqa: D107
        self,
        log_every_n_steps: int = 10,
        window_size: int | None = None,
        batch_size_per_gpu: int | None = None,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.window_size = window_size if window_size is not None else log_every_n_steps
        self.batch_size_per_gpu = batch_size_per_gpu

        self._batch_times: list[float] = []
        self._last_time: float | None = None
        # Skip the first N batches (compilation warmup)
        self._warmup_done = False
        self._warmup_batches = 5

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # noqa: D102
        self._last_time = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if self._last_time is None:
            return

        elapsed = time.monotonic() - self._last_time

        if not self._warmup_done:
            self._warmup_batches -= 1
            if self._warmup_batches <= 0:
                self._warmup_done = True
            return

        self._batch_times.append(elapsed)
        if len(self._batch_times) > self.window_size:
            self._batch_times = self._batch_times[-self.window_size :]

        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return
        if len(self._batch_times) < 3:
            return

        avg_time = sum(self._batch_times) / len(self._batch_times)
        iter_per_sec = 1.0 / avg_time if avg_time > 0 else 0.0

        bs = self.batch_size_per_gpu
        if bs is None:
            try:
                bs = trainer.datamodule.batch_size
            except (AttributeError, TypeError):
                bs = 0

        world_size = trainer.world_size if trainer.world_size else 1
        samples_per_sec = iter_per_sec * bs * world_size

        pl_module.log_dict(
            {
                "perf/iter_per_sec": iter_per_sec,
                "perf/samples_per_sec": samples_per_sec,
                "perf/batch_time_ms": avg_time * 1000.0,
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

    def on_validation_start(self, trainer, pl_module):  # noqa: D102
        self._last_time = None

    def on_validation_end(self, trainer, pl_module):  # noqa: D102
        self._last_time = None
