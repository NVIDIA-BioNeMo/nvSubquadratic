"""Callback to measure and log iteration throughput.

Logs detailed timing breakdown to wandb:
    ``perf/iter_per_sec``      – wall-clock iterations per second (start-to-start)
    ``perf/samples_per_sec``   – samples per second
    ``perf/total_ms``          – total iteration time (start-to-start)
    ``perf/fwd_ms``            – forward pass (batch_start → before_backward)
    ``perf/bwd_ms``            – backward pass (before_backward → after_backward)
    ``perf/other_ms``          – optimizer + data loading + callbacks + overhead

Uses **start-to-start** timing for the total iteration so that work done by
other callbacks in ``on_train_batch_end`` does not pollute the measurement.

A ``torch.cuda.synchronize()`` is inserted at each timing boundary so that the
fwd/bwd/other breakdown accurately reflects GPU time rather than CPU
kernel-launch time.  This adds ~0.5 ms per step but prevents wildly misleading
numbers with ``torch.compile`` or deep async pipelines.
"""

from __future__ import annotations

import time
from collections import deque

import pytorch_lightning as pl
import torch


class IterationSpeedCallback(pl.Callback):
    """Logs iteration throughput and fwd/bwd breakdown as wandb scalars.

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

        self._iter_times: deque[float] = deque(maxlen=self.window_size)
        self._fwd_times: deque[float] = deque(maxlen=self.window_size)
        self._bwd_times: deque[float] = deque(maxlen=self.window_size)

        self._prev_start: float | None = None
        self._batch_start: float | None = None
        self._bwd_start: float | None = None
        # Skip the first N batches (compilation warmup)
        self._warmup_done = False
        self._warmup_batches = 5

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # noqa: D102
        self._sync()
        now = time.monotonic()

        if self._prev_start is not None:
            elapsed = now - self._prev_start

            if not self._warmup_done:
                self._warmup_batches -= 1
                if self._warmup_batches <= 0:
                    self._warmup_done = True
            else:
                self._iter_times.append(elapsed)

        self._prev_start = now
        self._batch_start = now

    def on_before_backward(self, trainer, pl_module, loss):  # noqa: D102
        self._sync()
        now = time.monotonic()
        if self._warmup_done and self._batch_start is not None:
            self._fwd_times.append(now - self._batch_start)
        self._bwd_start = now

    def on_after_backward(self, trainer, pl_module):  # noqa: D102
        self._sync()
        if self._warmup_done and self._bwd_start is not None:
            self._bwd_times.append(time.monotonic() - self._bwd_start)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D102
        if not self._warmup_done:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return
        if len(self._iter_times) < min(3, self.window_size):
            return

        avg_iter = sum(self._iter_times) / len(self._iter_times)
        avg_fwd = sum(self._fwd_times) / len(self._fwd_times) if self._fwd_times else 0.0
        avg_bwd = sum(self._bwd_times) / len(self._bwd_times) if self._bwd_times else 0.0
        avg_other = max(0.0, avg_iter - avg_fwd - avg_bwd)

        iter_per_sec = 1.0 / avg_iter if avg_iter > 0 else 0.0

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
                "perf/total_ms": avg_iter * 1000.0,
                "perf/fwd_ms": avg_fwd * 1000.0,
                "perf/bwd_ms": avg_bwd * 1000.0,
                "perf/other_ms": avg_other * 1000.0,
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

    def on_validation_start(self, trainer, pl_module):  # noqa: D102
        self._prev_start = None
        self._batch_start = None
        self._bwd_start = None

    def on_validation_end(self, trainer, pl_module):  # noqa: D102
        self._prev_start = None
        self._batch_start = None
        self._bwd_start = None
