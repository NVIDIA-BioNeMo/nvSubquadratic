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

"""Callback to periodically run `wandb artifact cache cleanup ${X}GB` to cap local cache size."""

import subprocess
import threading
from typing import Mapping, Optional

import pytorch_lightning as pl


class WandbCacheCleanupCallback(pl.callbacks.Callback):
    """Periodically run `wandb artifact cache cleanup` to cap local cache size.

    Args:
        max_cache_size: Size cap passed to the W&B CLI, e.g., "10GB", "5GB".
        every_n_epochs: Run cleanup when (current_epoch + 1) % N == 0.
        run_on_fit_start: If True, also run once at fit start.
        only_on_global_rank_zero: If True, only run on rank 0 in DDP.
        executable: CLI executable name/path for wandb (default: "wandb").
        extra_env: Optional environment overrides for the subprocess.
        background: If True, run non-blocking via background thread + Popen.
        timeout: Optional timeout (seconds) for blocking mode.
    """

    def __init__(
        self,
        max_cache_size: str = "5GB",
        every_n_epochs: int = 1,
        run_on_fit_start: bool = False,
        only_on_global_rank_zero: bool = True,
        executable: str = "wandb",
        extra_env: Optional[Mapping[str, str]] = None,
        background: bool = True,
        timeout: Optional[int] = None,
    ) -> None:
        """Initializes the WandbCacheCleanupCallback."""
        super().__init__()
        self.max_cache_size = str(max_cache_size)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.run_on_fit_start = bool(run_on_fit_start)
        self.only_on_global_rank_zero = bool(only_on_global_rank_zero)
        self.executable = executable
        self.extra_env = dict(extra_env) if extra_env is not None else None
        self.background = bool(background)
        self.timeout = timeout
        self._background_thread: Optional[threading.Thread] = None

    def _should_run(self, trainer: pl.Trainer) -> bool:
        if self.only_on_global_rank_zero and hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)
        return True

    def _cleanup_task(self) -> None:
        try:
            subprocess.Popen(
                [self.executable, "artifact", "cache", "cleanup", self.max_cache_size],
                env=self.extra_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass

    def _run_cleanup(self) -> None:
        if not self.background:
            try:
                subprocess.run(
                    [self.executable, "artifact", "cache", "cleanup", self.max_cache_size],
                    check=False,
                    env=self.extra_env,
                    timeout=self.timeout,
                )
            except Exception:
                pass
            return

        if self._background_thread is None or not self._background_thread.is_alive():
            self._background_thread = threading.Thread(target=self._cleanup_task, daemon=True)
            self._background_thread.start()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[name-defined]
        """Runs cleanup at the start of fitting."""
        if self.run_on_fit_start and self._should_run(trainer):
            self._run_cleanup()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[name-defined]
        """Runs cleanup at the end of each training epoch."""
        if not self._should_run(trainer):
            return
        epoch_idx = int(getattr(trainer, "current_epoch", 0)) + 1
        if epoch_idx % self.every_n_epochs == 0:
            self._run_cleanup()
