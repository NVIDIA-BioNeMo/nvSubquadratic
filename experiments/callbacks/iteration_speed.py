"""Callback to measure and log iteration throughput.

Logs to wandb:

**Reliable wall-clock metrics** (cumulative, never lie):
    ``perf/wc_iter_per_sec``     – training steps / training wall-seconds
    ``perf/wc_samples_per_sec``  – samples / training wall-seconds
    ``perf/wc_total_sec``        – total training wall-time so far (excl. validation)

**Windowed breakdown** (best-effort, can be noisy with ``torch.compile``):
    ``perf/iter_per_sec``        – 1 / avg_iter over recent window
    ``perf/total_ms``            – avg iteration time (start-to-start)
    ``perf/fwd_ms``              – forward pass (batch_start → before_backward)
    ``perf/bwd_ms``              – backward pass (before_backward → after_backward)
    ``perf/other_ms``            – everything else (data loading, optimizer, callbacks)

**GPU memory**:
    ``perf/peak_gpu_mb``         – peak allocated GPU memory (torch.cuda)
    ``perf/current_gpu_mb``      – current allocated GPU memory

The wall-clock metrics accumulate *only* training time — the timer pauses during
validation and resumes when training starts again, so they reflect true training
throughput regardless of how often or how long validation runs.

``torch.cuda.synchronize()`` is used at timing boundaries for the windowed
breakdown.  This adds ~0.5 ms per step but is necessary for meaningful numbers
when the GPU pipeline is deep.
"""

from __future__ import annotations

import time
from collections import deque

import pytorch_lightning as pl
import torch


class IterationSpeedCallback(pl.Callback):
    """Logs iteration throughput, fwd/bwd breakdown, and GPU memory to wandb.

    Provides two families of metrics:

    * **Wall-clock** (``perf/wc_*``): cumulative counters that track true
      training throughput.  The timer *pauses* during validation and
      *resumes* when training continues, so these are immune to
      variable-frequency validation skewing the numbers.

    * **Windowed** (``perf/iter_per_sec``, ``perf/fwd_ms``, etc.): rolling
      averages over the last ``window_size`` batches.  More responsive to
      local changes but can be noisy, especially during ``torch.compile``
      warmup (the first ``warmup_batches`` steps are excluded).

    GPU memory is sampled from ``torch.cuda.max_memory_allocated`` /
    ``memory_allocated`` and logged as ``perf/peak_gpu_mb`` /
    ``perf/current_gpu_mb``.

    Args:
        log_every_n_steps: How often to log speed metrics.
        window_size: Number of recent batch times to average over.
            Defaults to ``log_every_n_steps``.
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

        # Windowed breakdown (rolling deques for recent batch timings)
        self._iter_times: deque[float] = deque(maxlen=self.window_size)
        self._fwd_times: deque[float] = deque(maxlen=self.window_size)
        self._bwd_times: deque[float] = deque(maxlen=self.window_size)

        self._prev_start: float | None = None  # start of *previous* batch (for iter delta)
        self._batch_start: float | None = None  # start of current batch (for fwd timing)
        self._bwd_start: float | None = None  # start of backward pass
        # Exclude the first N batches from windowed stats (torch.compile JIT warmup)
        self._warmup_done = False
        self._warmup_batches = 5

        # Cumulative wall-clock training time (pauses during validation)
        self._wc_train_seconds: float = 0.0
        self._wc_train_steps: int = 0
        self._wc_epoch_start: float | None = None

    def _sync(self) -> None:
        """Synchronize CUDA stream so host-side timestamps reflect actual GPU work."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    #  Training hooks                                                      #
    # ------------------------------------------------------------------ #

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
        # wall-clock: mark where this batch started
        self._wc_epoch_start = now

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
        # Always accumulate wall-clock (even during warmup — warmup is real time)
        if self._wc_epoch_start is not None:
            self._wc_train_seconds += time.monotonic() - self._wc_epoch_start
            self._wc_train_steps += 1
            self._wc_epoch_start = None

        if not self._warmup_done:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return
        if len(self._iter_times) < min(3, self.window_size):
            return

        # --- windowed metrics ---
        avg_iter = sum(self._iter_times) / len(self._iter_times)
        avg_fwd = sum(self._fwd_times) / len(self._fwd_times) if self._fwd_times else 0.0
        avg_bwd = sum(self._bwd_times) / len(self._bwd_times) if self._bwd_times else 0.0
        avg_other = max(0.0, avg_iter - avg_fwd - avg_bwd)
        iter_per_sec = 1.0 / avg_iter if avg_iter > 0 else 0.0

        # --- batch size ---
        bs = self.batch_size_per_gpu
        if bs is None:
            try:
                bs = trainer.datamodule.batch_size
            except (AttributeError, TypeError):
                bs = 0
        world_size = trainer.world_size if trainer.world_size else 1

        # --- wall-clock metrics ---
        wc_ips = self._wc_train_steps / self._wc_train_seconds if self._wc_train_seconds > 0 else 0.0
        wc_sps = wc_ips * bs * world_size

        # --- GPU memory ---
        peak_gpu_mb = 0.0
        current_gpu_mb = 0.0
        if torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2)
            current_gpu_mb = torch.cuda.memory_allocated() / (1024**2)

        pl_module.log_dict(
            {
                # Reliable wall-clock throughput
                "perf/wc_iter_per_sec": wc_ips,
                "perf/wc_samples_per_sec": wc_sps,
                "perf/wc_total_sec": self._wc_train_seconds,
                # Windowed breakdown (best-effort)
                "perf/iter_per_sec": iter_per_sec,
                "perf/samples_per_sec": iter_per_sec * bs * world_size,
                "perf/total_ms": avg_iter * 1000.0,
                "perf/fwd_ms": avg_fwd * 1000.0,
                "perf/bwd_ms": avg_bwd * 1000.0,
                "perf/other_ms": avg_other * 1000.0,
                # GPU memory
                "perf/peak_gpu_mb": peak_gpu_mb,
                "perf/current_gpu_mb": current_gpu_mb,
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

    # ------------------------------------------------------------------ #
    #  Validation hooks — pause wall-clock timer                           #
    # ------------------------------------------------------------------ #

    def on_validation_start(self, trainer, pl_module):  # noqa: D102
        # Flush any partial batch time (shouldn't happen, but be safe)
        if self._wc_epoch_start is not None:
            self._wc_train_seconds += time.monotonic() - self._wc_epoch_start
            self._wc_epoch_start = None
        self._prev_start = None
        self._batch_start = None
        self._bwd_start = None

    def on_validation_end(self, trainer, pl_module):  # noqa: D102
        self._prev_start = None
        self._batch_start = None
        self._bwd_start = None
