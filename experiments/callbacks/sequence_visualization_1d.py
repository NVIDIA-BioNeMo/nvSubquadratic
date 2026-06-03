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

# David W. Romero, 2025-01-19

"""1D sequence visualization callback for PyTorch Lightning.

For spatial recall 1D tasks where:
- Input: 1D canvas [B, L, C] (after on_before_batch_transfer)
- Output: Flattened image [B, segment_length, C] that should be reshaped to 2D

Visualization shows:
- Canvas as 1D line plot (optionally with mask as separate plot)
- Prediction reshaped to 2D image
- Label reshaped to 2D image
"""

import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb


class Sequence1DVisualizationCallback(pl.callbacks.Callback):
    """1D sequence visualization callback for spatial recall tasks.

    Visualizes:
    - Input canvas as 1D line plot
    - Prediction and label as 2D images (reshaped from segment)

    Can be triggered every N epochs or every N training iterations.

    Args:
        num_samples: Number of samples to visualize.
        target_size: Original 2D image size (segment_length = target_size²).
        every_n_epochs: How often to visualize (in epochs). Set to None to disable.
        every_n_train_steps: How often to visualize (in training steps). Set to None to disable.
        key: Key to use for the visualization in the logger.
        show_input: Whether to show the input canvas alongside prediction and label.
        show_mask_separately: If True and input has 2 channels, display the
                canvas and mask as separate line plots.
                Grid becomes: [canvas, mask, prediction, label] per row.
        denormalize: Whether to denormalize the images.
        mean: Mean of the dataset (for denormalization).
        std: Standard deviation of the dataset (for denormalization).
        readout_value: Value used for readout region (for visualization reference line).
    """

    def __init__(
        self,
        num_samples: int = 4,
        target_size: int = 16,
        every_n_epochs: Optional[int] = 1,
        every_n_train_steps: Optional[int] = None,
        key: str = "val/sequence_1d_grid",
        show_input: bool = True,
        show_mask_separately: bool = False,
        denormalize: bool = True,
        mean: float = 0.1307,
        std: float = 0.3081,
        readout_value: float = 0.0,
    ) -> None:
        """Initialize the callback."""
        super().__init__()
        self.num_samples = num_samples
        self.target_size = target_size
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
        self.key = key
        self.show_input = show_input
        self.show_mask_separately = show_mask_separately
        self.denormalize = denormalize
        self.mean = mean
        self.std = std
        self.readout_value = readout_value
        self._last_logged_step = -1

    def _maybe_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor from normalized space to [0, 1] range."""
        if not self.denormalize:
            return x
        return x * self.std + self.mean

    def _plot_1d_canvas(
        self,
        ax: plt.Axes,
        canvas: np.ndarray,
        segment_length: int,
        title: str = "Canvas (1D)",
        color: str = "steelblue",
        show_readout_shading: bool = True,
    ) -> None:
        """Plot a 1D canvas as a line plot with optional readout region shading.

        Args:
            ax: Matplotlib axes to plot on.
            canvas: 1D numpy array of canvas values (single channel).
            segment_length: Length of the segment (for readout region).
            title: Title for the plot.
            color: Color for the line plot.
            show_readout_shading: Whether to show the readout region shading.
        """
        canvas_denorm = self._maybe_denorm(torch.tensor(canvas)).numpy()
        ax.plot(canvas_denorm, linewidth=0.5, color=color)

        # Adjust y-axis to show readout_value if present
        y_min = min(-0.1, canvas_denorm.min() - 0.1)
        y_max = max(1.1, canvas_denorm.max() + 0.1)
        ax.set_ylim(y_min, y_max)

        # Add readout reference line if not zero
        if self.readout_value != 0:
            ax.axhline(y=self.readout_value, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

        # Mark readout region with shading
        if show_readout_shading:
            canvas_length = canvas.shape[0]
            readout_start = canvas_length - segment_length
            ax.axvspan(readout_start, canvas_length, alpha=0.1, color="green", label="readout")

        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")

    def _plot_1d_canvas_rgb(
        self,
        ax: plt.Axes,
        canvas_rgb: np.ndarray,
        segment_length: int,
        title: str = "Canvas (1D RGB)",
        show_readout_shading: bool = True,
    ) -> None:
        """Plot a multi-channel 1D canvas with one line per RGB channel.

        Args:
            ax: Matplotlib axes to plot on.
            canvas_rgb: Array of shape [L, 3] (R, G, B channels).
            segment_length: Length of the segment (for readout region).
            title: Title for the plot.
            show_readout_shading: Whether to show the readout region shading.
        """
        for ch, ch_color in enumerate(["red", "green", "blue"]):
            ch_data = self._maybe_denorm(torch.tensor(canvas_rgb[:, ch])).numpy()
            ax.plot(ch_data, linewidth=0.4, color=ch_color, alpha=0.8)

        y_min = min(-0.1, canvas_rgb.min() - 0.1)
        y_max = max(1.1, canvas_rgb.max() + 0.1)
        ax.set_ylim(y_min, y_max)

        if self.readout_value != 0:
            ax.axhline(y=self.readout_value, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        if show_readout_shading:
            canvas_length = canvas_rgb.shape[0]
            readout_start = canvas_length - segment_length
            ax.axvspan(readout_start, canvas_length, alpha=0.1, color="green", label="readout")

        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Value")

    def _plot_2d_image(
        self,
        ax: plt.Axes,
        flat_data: torch.Tensor,
        title: str = "Image",
        num_channels: int = 1,
    ) -> None:
        """Plot flattened data as a 2D image (grayscale or RGB).

        Args:
            ax: Matplotlib axes to plot on.
            flat_data: Tensor of shape [segment_length] (grayscale) or
                [segment_length, C] (multi-channel).
            title: Title for the plot.
            num_channels: Number of output channels (1=grayscale, 3=RGB).
        """
        if num_channels == 1:
            data_2d = self._maybe_denorm(flat_data).reshape(self.target_size, self.target_size).numpy()
            ax.imshow(data_2d, cmap="gray", vmin=0, vmax=1)
        else:
            # [segment_length, C] -> [H, W, C]
            data_2d = self._maybe_denorm(flat_data).reshape(self.target_size, self.target_size, num_channels).numpy()
            ax.imshow(np.clip(data_2d, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    @torch.no_grad()
    def _log_visualization(self, trainer: pl.Trainer, pl_module: pl.LightningModule, event_idx: int) -> None:
        """Generate and log 1D sequence visualization.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
            event_idx: Index of the logging event (used to select different batches).
        """
        # Get a validation batch
        val_loaders = getattr(trainer, "val_dataloaders", None)
        val_loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders

        # Select a different batch each logging event
        num_batches = len(val_loader)
        batch_idx = event_idx % num_batches
        batch_iter = itertools.islice(iter(val_loader), batch_idx, None)
        batch = next(batch_iter)

        # Apply datamodule's batch transfer if defined
        if hasattr(trainer.datamodule, "on_before_batch_transfer"):
            batch = trainer.datamodule.on_before_batch_transfer(batch, 0)

        if isinstance(batch, dict):
            x = batch["input"]
            y = batch["label"]
            condition = batch.get("condition")
        else:
            x, y = batch
            condition = None

        device = pl_module.device
        x = x.to(device)
        y = y.to(device)
        if condition is not None:
            condition = condition.to(device)

        # Forward pass
        pl_module.eval()
        preds = pl_module({"input": x, "condition": condition})["logits"]

        # x: [B, L, C_in], y: [B, segment_length, C_out], preds: [B, segment_length, C_out]
        n = min(self.num_samples, x.shape[0])
        num_in_channels = x.shape[2]
        num_out_channels = y.shape[2]
        segment_length = self.target_size * self.target_size
        is_rgb_input = num_in_channels == 3
        is_rgb_output = num_out_channels == 3

        # Determine number of columns based on options
        # Base: prediction + label = 2 columns
        # With show_input: + canvas = 3 columns
        # With show_mask_separately and 2 channels: + mask = 4 columns
        input_has_mask = num_in_channels == 2 and self.show_mask_separately

        num_cols = 2  # prediction, label
        if self.show_input:
            num_cols += 1  # canvas
            if input_has_mask:
                num_cols += 1  # mask

        # Create figure
        fig, axes = plt.subplots(n, num_cols, figsize=(4 * num_cols, 2.5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i in range(n):
            col_idx = 0

            if self.show_input:
                if is_rgb_input:
                    # RGB canvas: plot all 3 channels
                    canvas_rgb = x[i].cpu().numpy()  # [L, 3]
                    self._plot_1d_canvas_rgb(
                        axes[i, col_idx],
                        canvas_rgb,
                        segment_length,
                        title="Canvas (1D RGB)" if i == 0 else "",
                    )
                else:
                    # Grayscale canvas: 1D line plot [L, C] -> [L] for channel 0
                    canvas = x[i, :, 0].cpu().numpy()  # [L]
                    self._plot_1d_canvas(
                        axes[i, col_idx],
                        canvas,
                        segment_length,
                        title="Canvas (1D)" if i == 0 else "",
                        color="steelblue",
                    )
                col_idx += 1

                # Mask: separate line plot if 2-channel input
                if input_has_mask:
                    mask = x[i, :, 1].cpu().numpy()  # [L]
                    ax_mask = axes[i, col_idx]
                    ax_mask.plot(mask, linewidth=0.5, color="orange")
                    ax_mask.set_ylim(-0.1, 1.1)
                    if i == 0:
                        ax_mask.set_title("Mask (1D)")
                    ax_mask.set_xlabel("Position")
                    ax_mask.set_ylabel("Value")
                    col_idx += 1

            # Prediction: reshape to 2D image
            if is_rgb_output:
                pred_data = preds[i].cpu()  # [segment_length, 3]
            else:
                pred_data = preds[i, :, 0].cpu()  # [segment_length]
            self._plot_2d_image(
                axes[i, col_idx],
                pred_data,
                title="Prediction" if i == 0 else "",
                num_channels=num_out_channels,
            )
            col_idx += 1

            # Label: reshape to 2D image
            if is_rgb_output:
                label_data = y[i].cpu()  # [segment_length, 3]
            else:
                label_data = y[i, :, 0].cpu()  # [segment_length]
            self._plot_2d_image(
                axes[i, col_idx],
                label_data,
                title="Label" if i == 0 else "",
                num_channels=num_out_channels,
            )

        fig.tight_layout()

        # Convert figure to image (compatible with newer matplotlib)
        fig.canvas.draw()
        img_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
        plt.close(fig)

        # Log with available logger
        if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log({self.key: [wandb.Image(img_array)]}, step=trainer.global_step)
        elif hasattr(trainer.logger, "log_image"):
            # Convert to tensor for Lightning logger API
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            trainer.logger.log_image(key=self.key, images=[img_tensor])

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        """Visualize sequences during training every N steps."""
        if self.every_n_train_steps is None:
            return

        global_step = trainer.global_step
        if global_step == 0:
            return
        if global_step % self.every_n_train_steps != 0:
            return
        if global_step == self._last_logged_step:
            return

        self._last_logged_step = global_step
        event_idx = global_step // self.every_n_train_steps
        self._log_visualization(trainer, pl_module, event_idx)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Visualize at the end of the validation epoch."""
        if self.every_n_epochs is None:
            return

        epoch = trainer.current_epoch
        if self.every_n_epochs > 1 and (epoch % self.every_n_epochs != 0):
            return

        event_idx = epoch // max(self.every_n_epochs, 1)
        self._log_visualization(trainer, pl_module, event_idx)
