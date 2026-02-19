"""Exponential Moving Average (EMA) callback for PyTorch Lightning.

Maintains a deep-copied shadow model whose parameters track the training model
with exponential decay. At validation/test time the shadow model is used
directly — no weight swapping needed.

This mirrors the EMA approach used in the diffusion wrapper, unified as a
reusable callback for any task (classification, diffusion, etc.).
"""

import copy

import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """Maintains an EMA shadow model and uses it for validation/test evaluation.

    The shadow model is a full ``deepcopy`` of the network kept on the same
    device.  During training the shadow parameters are updated in-place with
    exponential moving average.  At validation/test boundaries the wrapper's
    ``network`` attribute is temporarily replaced with the shadow model so that
    all downstream code (metrics, logging, etc.) evaluates EMA weights without
    any explicit weight-swap overhead.
    """

    def __init__(self, decay: float = 0.9999, update_every: int = 1, warmup_steps: int = 5000):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self._ema_model: torch.nn.Module | None = None
        self._training_model: torch.nn.Module | None = None
        self._has_been_updated = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create the EMA shadow model and move it to the training device."""
        if self._ema_model is None:
            self._ema_model = copy.deepcopy(pl_module.network)
            for p in self._ema_model.parameters():
                p.detach_()
                p.requires_grad_(False)
        self._ema_model.to(pl_module.device)
        self._ema_model.eval()

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx
    ) -> None:
        """Update EMA parameters at the configured interval."""
        if self._ema_model is None:
            return
        if trainer.global_step < self.warmup_steps:
            return
        if trainer.global_step % self.update_every != 0:
            return

        decay = self.decay
        with torch.no_grad():
            for ema_param, param in zip(self._ema_model.parameters(), pl_module.network.parameters()):
                ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
            for ema_buf, buf in zip(self._ema_model.buffers(), pl_module.network.buffers()):
                if ema_buf.shape != buf.shape:
                    ema_buf.resize_as_(buf)
                ema_buf.copy_(buf)
            self._has_been_updated = True

    # ------------------------------------------------------------------
    # Validation / test: swap network pointer (no weight copies)
    # ------------------------------------------------------------------

    def _use_ema(self, pl_module: pl.LightningModule) -> None:
        """Replace the wrapper's network with the EMA shadow model."""
        if self._ema_model is None or not self._has_been_updated:
            return
        self._training_model = pl_module.network
        pl_module.network = self._ema_model

    def _restore_training(self, pl_module: pl.LightningModule) -> None:
        """Restore the original training network."""
        if self._training_model is None:
            return
        pl_module.network = self._training_model
        self._training_model = None

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._use_ema(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restore_training(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._use_ema(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._restore_training(pl_module)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Save EMA model state for checkpointing."""
        return {
            "ema_model_state_dict": self._ema_model.state_dict() if self._ema_model is not None else None,
            "has_been_updated": self._has_been_updated,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA model state from checkpoint."""
        ema_sd = state_dict.get("ema_model_state_dict")
        if ema_sd is not None and self._ema_model is not None:
            self._ema_model.load_state_dict(ema_sd)
        self._has_been_updated = state_dict.get("has_been_updated", False)
