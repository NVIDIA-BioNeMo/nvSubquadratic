# David W. Romero, 2025-09-01

"""Validation image grid callback for PyTorch Lightning."""

import itertools
from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from torchvision.utils import make_grid

import wandb


class ValidationImageGridCallback(pl.callbacks.Callback):
    """Validation image grid callback for PyTorch Lightning.

    Visualizes input, prediction, and label images in a grid during validation.
    Can be triggered every N epochs or every N training iterations.

    Args:
        num_samples: Number of samples to visualize.
        every_n_epochs: How often to visualize (in epochs). Set to None to disable.
        every_n_train_steps: How often to visualize (in training steps). Set to None to disable.
        key: Key to use for the visualization in the logger.
        show_input: Whether to show the input image alongside prediction and label.
        denormalize: Whether to denormalize the images.
        mean: Mean of the dataset (for denormalization).
        std: Standard deviation of the dataset (for denormalization).
        flattened_image_shape: Optional (H, W) to reshape flattened tensors of shape
            [B, H*W, C] into images. If not provided, will try to auto-infer a square shape.
    """

    def __init__(
        self,
        num_samples: int = 4,
        every_n_epochs: Optional[int] = 1,
        every_n_train_steps: Optional[int] = None,
        key: str = "val/image_grid",
        show_input: bool = True,
        denormalize: bool = True,
        mean: float = 0.1307,
        std: float = 0.3081,
        flattened_image_shape: tuple | None = None,
    ) -> None:
        """Initialize the callback."""
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
        self.key = key
        self.show_input = show_input
        self.denormalize = denormalize
        self.mean = mean
        self.std = std
        self.flattened_image_shape = flattened_image_shape
        self._last_logged_step = -1

    def _maybe_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor from normalized space to [0, 1] range."""
        if not self.denormalize:
            return x
        return x * self.std + self.mean

    def _as_nchw_images(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a batch tensor into NCHW image format.

        Supports:
        - BCHW (returned as-is)
        - BHWC (reordered to BCHW)
        - B(H*W)C where H*W is flattened (reshaped using `flattened_image_shape` or inferred square)
        """
        if tensor.ndim == 4:
            # Either BCHW or BHWC
            if tensor.shape[-1] in (1, 3):
                return rearrange(tensor, "b h w c -> b c h w")
            return tensor

        if tensor.ndim == 3:
            # Flattened features per sample: [B, H*W, C]
            flattened_features = tensor.shape[1]
            if self.flattened_image_shape is not None:
                h, w = self.flattened_image_shape
            else:
                # Assume square shape
                h = w = int(flattened_features**0.5)
                assert h * w == flattened_features, "Flattened features must be a square"
            tensor = tensor.view(tensor.shape[0], h, w, tensor.shape[2])
            # To BCHW
            return rearrange(tensor, "b h w c -> b c h w")

        raise ValueError(
            f"Unsupported tensor shape for visualization: {tensor.shape}. Expected BCHW, BHWC, or B(H*W)C."
        )

    @torch.no_grad()
    def _log_image_grid(self, trainer: pl.Trainer, pl_module: pl.LightningModule, event_idx: int) -> None:
        """Generate and log an image grid comparing input, prediction, and label.

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

        # If the datamodule defines on_before_batch_transfer, use it to match model input format
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

        # Convert to NCHW images, supporting flattened inputs.
        x_nchw = self._as_nchw_images(x)
        preds_nchw = self._as_nchw_images(preds)
        y_nchw = self._as_nchw_images(y)

        n = min(self.num_samples, preds_nchw.shape[0])
        x_nchw = x_nchw[:n]
        preds_nchw = preds_nchw[:n]
        y_nchw = y_nchw[:n]

        # Denormalize for visualization
        x_nchw = self._maybe_denorm(x_nchw)
        preds_nchw = self._maybe_denorm(preds_nchw)
        y_nchw = self._maybe_denorm(y_nchw)

        # Resize all images to the largest spatial size for consistent grid display
        # This handles cases where input is larger than prediction/label (e.g., spatial recall)
        max_h = max(x_nchw.shape[2], preds_nchw.shape[2], y_nchw.shape[2])
        max_w = max(x_nchw.shape[3], preds_nchw.shape[3], y_nchw.shape[3])
        max_c = max(x_nchw.shape[1], preds_nchw.shape[1], y_nchw.shape[1])

        def resize_if_needed(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
                return torch.nn.functional.interpolate(tensor, size=(target_h, target_w), mode="nearest")
            return tensor

        def expand_channels_if_needed(tensor: torch.Tensor, target_c: int) -> torch.Tensor:
            """Expand grayscale (1-channel) to RGB (3-channel) if needed."""
            if tensor.shape[1] == target_c:
                return tensor
            if tensor.shape[1] == 1 and target_c == 3:
                return tensor.repeat(1, 3, 1, 1)
            return tensor

        x_nchw = resize_if_needed(x_nchw, max_h, max_w)
        preds_nchw = resize_if_needed(preds_nchw, max_h, max_w)
        y_nchw = resize_if_needed(y_nchw, max_h, max_w)

        x_nchw = expand_channels_if_needed(x_nchw, max_c)
        preds_nchw = expand_channels_if_needed(preds_nchw, max_c)
        y_nchw = expand_channels_if_needed(y_nchw, max_c)

        # Build image grid: [input0, pred0, label0, input1, pred1, label1, ...] or [pred0, label0, ...]
        images = []
        for i in range(n):
            if self.show_input:
                images.append(x_nchw[i])
            images.append(preds_nchw[i])
            images.append(y_nchw[i])

        imgs = torch.stack(images, dim=0).detach().cpu().clamp(0.0, 1.0)
        nrow = 3 if self.show_input else 2
        grid = make_grid(imgs, nrow=nrow, padding=2)

        # Log with available logger
        if hasattr(trainer.logger, "log_image"):
            # Preferred Lightning logger API (logs a single image)
            trainer.logger.log_image(key=self.key, images=[grid])
        elif hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
            grid_np = grid.permute(1, 2, 0).numpy()
            trainer.logger.experiment.log({self.key: [wandb.Image(grid_np)]}, step=trainer.global_step)
        else:
            raise ValueError(f"Unsupported logger: {trainer.logger}")

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Visualize the validation images at the end of the epoch."""
        if self.every_n_epochs is None:
            return

        epoch = trainer.current_epoch
        if self.every_n_epochs > 1 and (epoch % self.every_n_epochs != 0):
            return

        event_idx = epoch // max(self.every_n_epochs, 1)
        self._log_image_grid(trainer, pl_module, event_idx)

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        """Visualize images during training every N steps."""
        if self.every_n_train_steps is None:
            return

        global_step = trainer.global_step
        # if global_step == 0:
        #     return
        if global_step % self.every_n_train_steps != 0:
            return
        if global_step == self._last_logged_step:
            return

        self._last_logged_step = global_step
        event_idx = global_step // self.every_n_train_steps
        self._log_image_grid(trainer, pl_module, event_idx)
