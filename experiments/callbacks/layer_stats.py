# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer statistics callback for debugging training dynamics.

This callback tracks activation and gradient statistics at the residual block level
to help diagnose training instabilities like gradient explosions.
"""

from functools import partial
from typing import Optional

import pytorch_lightning as pl
import torch


class LayerStatsCallback(pl.callbacks.Callback):
    """Callback to log layer-wise activation and gradient statistics.

    Uses forward hooks to collect statistics without modifying module code.
    Logs statistics periodically to WandB for debugging training dynamics.

    By default, tracks ResidualBlock outputs which gives a good overview of
    how activations/gradients flow through the network.

    Args:
        log_every_n_steps: How often to log statistics (in training steps).
        log_activations: Whether to log activation (output) statistics.
        log_gradients: Whether to log gradient statistics.
        track_residual_blocks: Whether to track ResidualBlock outputs.
        track_ckconv_layers: Whether to track CKConv layer outputs and kernels.
        layer_name_filters: Optional list of substrings to filter which layers to track.
            If None, uses default tracking based on track_* flags.
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        log_activations: bool = True,
        log_gradients: bool = True,
        track_residual_blocks: bool = True,
        track_ckconv_layers: bool = True,
        layer_name_filters: Optional[list[str]] = None,
    ) -> None:
        """Initialize the callback."""
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_activations = log_activations
        self.log_gradients = log_gradients
        self.track_residual_blocks = track_residual_blocks
        self.track_ckconv_layers = track_ckconv_layers
        self.layer_name_filters = layer_name_filters

        # Storage for collected statistics
        self._activation_stats: dict[str, float] = {}
        self._gradient_stats: dict[str, float] = {}
        self._hooks: list = []
        self._last_logged_step = -1

    def _should_track_layer(self, name: str, module: torch.nn.Module) -> bool:
        """Determine if a layer should be tracked."""
        module_type_name = type(module).__name__

        # Check explicit name filters first
        if self.layer_name_filters is not None:
            name_lower = name.lower()
            return any(f.lower() in name_lower for f in self.layer_name_filters)

        # Track ResidualBlock
        if self.track_residual_blocks and module_type_name == "ResidualBlock":
            return True

        # Track CKConv layers
        if self.track_ckconv_layers and "CKConv" in module_type_name:
            return True

        return False

    def _compute_tensor_stats(self, tensor: torch.Tensor, prefix: str) -> dict[str, float]:
        """Compute statistics for a tensor."""
        with torch.no_grad():
            stats = {
                f"{prefix}/norm": tensor.norm().item(),
                f"{prefix}/max_abs": tensor.abs().max().item(),
            }
        return stats

    def _forward_hook(
        self,
        module: torch.nn.Module,
        input: tuple,
        output: torch.Tensor,
        name: str,
    ) -> None:
        """Forward hook to collect activation statistics."""
        if not self.log_activations:
            return

        # Handle tuple outputs (some layers return tuples)
        if isinstance(output, tuple):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            return

        # Compute and store stats
        stats = self._compute_tensor_stats(output, f"activations/{name}")
        self._activation_stats.update(stats)

        # Special handling for CKConv layers - check for cached kernel stats
        if hasattr(module, "_debug_stats"):
            for key, value in module._debug_stats.items():
                self._activation_stats[f"kernel/{name}/{key}"] = value

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: tuple,
        grad_output: tuple,
        name: str,
    ) -> None:
        """Backward hook to collect gradient statistics."""
        if not self.log_gradients:
            return

        # grad_output is a tuple, get first element
        if grad_output and grad_output[0] is not None:
            grad = grad_output[0]
            if isinstance(grad, torch.Tensor):
                stats = self._compute_tensor_stats(grad, f"gradients/{name}")
                self._gradient_stats.update(stats)

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Register hooks on model layers."""
        if stage != "fit":
            return

        for name, module in pl_module.named_modules():
            if not self._should_track_layer(name, module):
                continue

            # Enable debug stats caching for CKConv layers
            if "CKConv" in type(module).__name__:
                module._cache_debug_stats = True

            # Register forward hook
            handle = module.register_forward_hook(partial(self._forward_hook, name=name))
            self._hooks.append(handle)

            # Register backward hook for gradient stats
            if self.log_gradients:
                handle = module.register_full_backward_hook(partial(self._backward_hook, name=name))
                self._hooks.append(handle)

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Remove hooks when training ends."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Log collected statistics after each batch."""
        global_step = trainer.global_step

        if global_step == 0:
            return
        if global_step % self.log_every_n_steps != 0:
            # Clear stats but don't log
            self._activation_stats.clear()
            self._gradient_stats.clear()
            return
        if global_step == self._last_logged_step:
            return

        self._last_logged_step = global_step

        # Combine all stats
        all_stats = {}
        all_stats.update(self._activation_stats)
        all_stats.update(self._gradient_stats)

        # Log to WandB if available
        if all_stats and hasattr(trainer.logger, "experiment"):
            trainer.logger.experiment.log(all_stats, step=global_step)

        # Clear for next collection
        self._activation_stats.clear()
        self._gradient_stats.clear()
