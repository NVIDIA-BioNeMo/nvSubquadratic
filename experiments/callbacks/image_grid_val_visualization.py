# David W. Romero, 2025-09-01

"""Validation image grid callback for PyTorch Lightning."""

import itertools
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from einops import rearrange
from torchvision.utils import make_grid


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
        show_mask_separately: If True and input has 2 channels, display the
                grayscale canvas and mask as separate side-by-side images in the grid.
                Grid becomes: [canvas, mask, prediction, label] per row.
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
        show_mask_separately: bool = False,
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
        self.show_mask_separately = show_mask_separately
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
            # Either BCHW or BHWC - detect by checking which dim is small (channels)
            # BHWC: last dim is small (1, 2, or 3 channels), and second dim is large (H)
            # BCHW: second dim is small (1, 2, or 3 channels)
            if tensor.shape[-1] in (1, 2, 3) and tensor.shape[1] > 3:
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
        # max_c = max(x_nchw.shape[1], preds_nchw.shape[1], y_nchw.shape[1])

        def resize_if_needed(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
                return torch.nn.functional.interpolate(tensor, size=(target_h, target_w), mode="nearest")
            return tensor

        def to_grayscale_rgb(tensor: torch.Tensor) -> torch.Tensor:
            """Convert 1-channel grayscale to 3-channel RGB by repeating."""
            if tensor.shape[1] == 1:
                return tensor.repeat(1, 3, 1, 1)
            if tensor.shape[1] == 3:
                return tensor
            # For 2 channels without mask separation, just take first channel
            return tensor[:, 0:1].repeat(1, 3, 1, 1)

        # Check if we should split mask for separate side-by-side display
        input_has_mask = x_nchw.shape[1] == 2 and self.show_mask_separately
        mask_nchw = None
        if input_has_mask:
            # Split input: channel 0 = grayscale canvas, channel 1 = binary mask
            mask_nchw = x_nchw[:, 1:2]  # [B, 1, H, W]
            x_nchw = x_nchw[:, 0:1]  # [B, 1, H, W]

        # Resize all to same spatial size
        x_nchw = resize_if_needed(x_nchw, max_h, max_w)
        preds_nchw = resize_if_needed(preds_nchw, max_h, max_w)
        y_nchw = resize_if_needed(y_nchw, max_h, max_w)
        if mask_nchw is not None:
            mask_nchw = resize_if_needed(mask_nchw, max_h, max_w)

        # Convert all to RGB for consistent grid display
        x_nchw = to_grayscale_rgb(x_nchw)
        preds_nchw = to_grayscale_rgb(preds_nchw)
        y_nchw = to_grayscale_rgb(y_nchw)
        if mask_nchw is not None:
            mask_nchw = to_grayscale_rgb(mask_nchw)

        # Build image grid
        # With mask separation: [canvas, mask, pred, label] per row (nrow=4)
        # Without: [input, pred, label] or [pred, label] per row
        images = []
        for i in range(n):
            if self.show_input:
                images.append(x_nchw[i])
            if mask_nchw is not None:
                images.append(mask_nchw[i])
            images.append(preds_nchw[i])
            images.append(y_nchw[i])

        imgs = torch.stack(images, dim=0).detach().cpu().clamp(0.0, 1.0)
        if mask_nchw is not None:
            nrow = 4  # canvas | mask | prediction | label
        elif self.show_input:
            nrow = 3  # input | prediction | label
        else:
            nrow = 2  # prediction | label
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
        if global_step == 0:
            return
        if global_step % self.every_n_train_steps != 0:
            return
        if global_step == self._last_logged_step:
            return

        self._last_logged_step = global_step
        event_idx = global_step // self.every_n_train_steps
        self._log_image_grid(trainer, pl_module, event_idx)


class ValidationVolumeGridCallback(pl.callbacks.Callback):
    """Validation volume grid callback for 3D spatial recall tasks.

    Visualizes 3D input volumes as perspective 3D scatter plots with depth on X-axis,
    alongside 2D prediction and label images. Similar to the dataset visualization.

    Args:
        num_samples: Number of samples to visualize.
        every_n_epochs: How often to visualize (in epochs). Set to None to disable.
        every_n_train_steps: How often to visualize (in training steps). Set to None to disable.
        key: Key to use for the visualization in the logger.
        show_mask_separately: If True and input has mask channel, display mask in 3D view.
        target_size: Size of the target/prediction image (for readout region display).
        denormalize: Whether to denormalize the images.
        mean: Mean of the dataset (for denormalization).
        std: Standard deviation of the dataset (for denormalization).
    """

    def __init__(
        self,
        num_samples: int = 4,
        every_n_epochs: Optional[int] = 1,
        every_n_train_steps: Optional[int] = None,
        key: str = "val/volume_grid",
        show_mask_separately: bool = False,
        target_size: int = 16,
        denormalize: bool = True,
        mean: float = 0.1307,
        std: float = 0.3081,
    ) -> None:
        """Initialize the callback."""
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
        self.key = key
        self.show_mask_separately = show_mask_separately
        self.target_size = target_size
        self.denormalize = denormalize
        self.mean = mean
        self.std = std
        self._last_logged_step = -1

    def _maybe_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor from normalized space to [0, 1] range."""
        if not self.denormalize:
            return x
        return x * self.std + self.mean

    def _draw_3d_volume(
        self,
        ax,
        vol: "np.ndarray",
        vol_rgb: "np.ndarray | None",
        D: int,
        H: int,
        W: int,
        t: int,
        title: str,
    ) -> None:
        """Draw a 3D perspective view of a volume using scatter plot.

        Args:
            ax: Matplotlib 3D axis.
            vol: Volume array [D, H, W] (grayscale/luminance).
            vol_rgb: Optional RGB volume [3, D, H, W].
            D: Depth dimension.
            H: Height dimension.
            W: Width dimension.
            t: Target size (for readout region).
            title: Title for the subplot.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import ndimage

        # Scale depth for better visualization
        depth_scale = max(1.0, H / D / 1.5)

        # Find non-zero voxels and draw coordinate indicator lines
        threshold = 0.05

        # Find individual items by detecting connected components per depth slice
        items = []  # List of (depth, h_center, w_center, h_min, w_min)
        for d in range(D):
            slice_2d = np.abs(vol[d]) > threshold
            if slice_2d.any():
                labeled, num_features = ndimage.label(slice_2d)
                for label_id in range(1, num_features + 1):
                    component = labeled == label_id
                    nz = np.where(component)
                    if len(nz[0]) > 10:  # Filter small noise
                        h_min, h_max = nz[0].min(), nz[0].max()
                        w_min, w_max = nz[1].min(), nz[1].max()
                        h_center = (h_min + h_max) / 2
                        w_center = (w_min + w_max) / 2
                        items.append((d, h_center, w_center, h_min, w_min))

        # Draw all non-zero voxels
        nz_coords = np.where(np.abs(vol) > threshold)
        if len(nz_coords[0]) > 0:
            d_coords = nz_coords[0] * depth_scale  # Depth on X
            h_coords = H - nz_coords[1]  # Height on Z (inverted)
            w_coords = nz_coords[2]  # Width on Y

            if vol_rgb is not None:
                # Use actual RGB colors from the volume
                colors = np.zeros((len(nz_coords[0]), 4))
                colors[:, 0] = np.clip(vol_rgb[0][nz_coords], 0, 1)  # R
                colors[:, 1] = np.clip(vol_rgb[1][nz_coords], 0, 1)  # G
                colors[:, 2] = np.clip(vol_rgb[2][nz_coords], 0, 1)  # B
                intensities = np.clip(vol[nz_coords], 0, 1)
                colors[:, 3] = np.clip(intensities * 0.9 + 0.1, 0, 1)  # Alpha
            else:
                # Grayscale
                intensities = np.clip(vol[nz_coords], 0, 1)
                colors = np.zeros((len(intensities), 4))
                colors[:, 0] = intensities  # R
                colors[:, 1] = intensities  # G
                colors[:, 2] = intensities  # B
                colors[:, 3] = np.clip(intensities * 0.9 + 0.1, 0, 1)  # Alpha

            ax.scatter(d_coords, w_coords, h_coords, c=colors, s=8, marker="s", depthshade=False)

        # Draw coordinate indicator lines for each item
        item_colors = plt.cm.tab10(np.linspace(0, 1, max(len(items), 1)))
        d_max_vis = (D - 0.5) * depth_scale

        for idx, (d, h_center, w_center, h_min, w_min) in enumerate(items):
            d_pos = d * depth_scale
            z_top = H - h_min  # Top of item (inverted)
            y_left = w_min  # Left of item
            color = item_colors[idx % len(item_colors)]

            # Draw marker at the item's top-left corner
            ax.scatter(
                [d_pos], [y_left], [z_top], c=[color], s=40, marker="o", edgecolors="black", linewidths=0.5, zorder=10
            )

            # Line down to floor (z=0)
            ax.plot([d_pos, d_pos], [y_left, y_left], [z_top, 0], color=color, linewidth=1.5, linestyle=":", alpha=0.7)

            # Line to back wall (d=max)
            ax.plot(
                [d_pos, d_max_vis], [y_left, y_left], [0, 0], color=color, linewidth=1.0, linestyle="--", alpha=0.5
            )

            # Line to side wall (y=0)
            ax.plot([d_pos, d_pos], [y_left, 0], [0, 0], color=color, linewidth=1.0, linestyle="--", alpha=0.5)

            # Small marker on floor showing projection
            ax.scatter([d_pos], [y_left], [0], c=[color], s=20, marker="x", alpha=0.7)

            # Add coordinate label
            ax.text(d_pos + 1, y_left + 2, -2, f"d={d}", fontsize=7, color=color, fontweight="bold")

        # Draw canvas wireframe (light)
        def draw_box_edges(ax, x0, x1, y0, y1, z0, z1, color="gray", linewidth=0.5, linestyle="-"):
            edges = [
                ([x0, x1], [y0, y0], [z0, z0]),
                ([x0, x1], [y1, y1], [z0, z0]),
                ([x0, x1], [y0, y0], [z1, z1]),
                ([x0, x1], [y1, y1], [z1, z1]),
                ([x0, x0], [y0, y1], [z0, z0]),
                ([x1, x1], [y0, y1], [z0, z0]),
                ([x0, x0], [y0, y1], [z1, z1]),
                ([x1, x1], [y0, y1], [z1, z1]),
                ([x0, x0], [y0, y0], [z0, z1]),
                ([x1, x1], [y0, y0], [z0, z1]),
                ([x0, x0], [y1, y1], [z0, z1]),
                ([x1, x1], [y1, y1], [z0, z1]),
            ]
            for e in edges:
                ax.plot(e[0], e[1], e[2], color=color, linewidth=linewidth, linestyle=linestyle)

        d_max = (D - 0.5) * depth_scale
        draw_box_edges(ax, 0, d_max, 0, W, 0, H, color="lightgray", linewidth=0.5, linestyle="-")

        # Draw readout region as red dashed rectangle on back slice
        d_back = (D - 1) * depth_scale
        readout_y = [W - t, W, W, W - t, W - t]
        readout_z = [0, 0, t, t, 0]
        ax.plot([d_back] * 5, readout_y, readout_z, color="red", linewidth=2, linestyle="--", alpha=0.8)
        ax.text(d_back + 1, W - t / 2, t / 2, "readout", fontsize=7, color="red")

        # Set labels
        ax.set_xlabel("Depth")
        ax.set_ylabel("Width")
        ax.set_zlabel("Height")
        ax.set_xlim(-1, D * depth_scale)
        ax.set_ylim(0, W)
        ax.set_zlim(0, H)

        # Fix depth axis ticks
        depth_ticks = np.arange(D) * depth_scale
        ax.set_xticks(depth_ticks)
        ax.set_xticklabels([str(d) for d in range(D)])

        # Set viewing angle and aspect ratio (matching dataset visualization)
        ax.view_init(elev=20, azim=-50)
        ax.set_box_aspect([D * depth_scale / W, 1, H / W])

        ax.set_title(title, fontsize=10)

    @torch.no_grad()
    def _log_volume_grid(self, trainer: pl.Trainer, pl_module: pl.LightningModule, event_idx: int) -> None:
        """Generate and log a 3D perspective visualization of volumes with predictions.

        Creates a figure with 3D scatter plot views of input volumes alongside
        2D prediction and label images, similar to the dataset visualization.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
            event_idx: Index of the logging event (used to select different batches).
        """
        import io

        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        # Get a validation batch
        val_loaders = getattr(trainer, "val_dataloaders", None)
        val_loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders

        # Select a different batch each logging event
        num_batches = len(val_loader)
        batch_idx = event_idx % num_batches
        batch_iter = itertools.islice(iter(val_loader), batch_idx, None)
        batch = next(batch_iter)

        # If the datamodule defines on_before_batch_transfer, use it
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

        # Limit samples
        n = min(self.num_samples, x.shape[0])
        x = x[:n]
        preds = preds[:n]
        y = y[:n]

        # Note: For colored frames (RGB), the colors are raw values [0,1], so we don't denormalize.
        # For grayscale data, denormalization may be needed but clipping to [0,1] handles it.
        # The dataset visualization also doesn't denormalize - it just clips values.

        # Move to CPU and numpy (no denormalization - just clip later)
        x_np = x.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Get volume dimensions
        # x is [B, D, H, W, C] in volume format
        if x_np.ndim == 5:
            _B, D, H, W, C = x_np.shape
        else:
            raise ValueError(f"Expected 5D volume tensor, got shape {x_np.shape}")

        t = self.target_size

        # Determine if RGB (3 channels) or grayscale+mask (1 or 2 channels)
        is_rgb = C == 3
        has_mask = C == 2 and self.show_mask_separately

        # Number of columns: 3D input, (mask), prediction, label
        num_cols = 4 if has_mask else 3

        # Create figure (matching dataset visualization size)
        fig = plt.figure(figsize=(8 * num_cols, 7 * n))

        for i in range(n):
            vol_data = x_np[i]  # [D, H, W, C]

            # Rearrange to [C, D, H, W] for easier handling
            vol_cdhw = np.transpose(vol_data, (3, 0, 1, 2))  # [C, D, H, W]

            # Get volume for visualization
            if is_rgb:
                # RGB: use luminance for item detection
                vol = 0.299 * vol_cdhw[0] + 0.587 * vol_cdhw[1] + 0.114 * vol_cdhw[2]
                vol_rgb = vol_cdhw  # [3, D, H, W]
            elif has_mask:
                vol = vol_cdhw[0]  # [D, H, W]
                vol_rgb = None
                mask_vol = vol_cdhw[1]  # [D, H, W]
            else:
                vol = vol_cdhw[0]  # [D, H, W]
                vol_rgb = None

            # === 3D Input View ===
            ax3d = fig.add_subplot(n, num_cols, i * num_cols + 1, projection="3d")
            self._draw_3d_volume(ax3d, vol, vol_rgb, D, H, W, t, f"3D Input ({D}×{H}×{W})")

            col_idx = 2

            # === Mask View (if applicable) ===
            if has_mask:
                ax_mask = fig.add_subplot(n, num_cols, i * num_cols + col_idx, projection="3d")
                self._draw_3d_volume(ax_mask, mask_vol, None, D, H, W, t, f"Mask ({D}×{H}×{W})")
                col_idx += 1

            # === Prediction ===
            ax_pred = fig.add_subplot(n, num_cols, i * num_cols + col_idx)
            pred_img = preds_np[i]  # [H, W, C]
            if pred_img.shape[-1] == 1:
                ax_pred.imshow(pred_img[:, :, 0], cmap="gray", vmin=0, vmax=1)
            else:
                ax_pred.imshow(np.clip(pred_img, 0, 1))
            ax_pred.set_title(f"Prediction ({t}×{t})", fontsize=10)
            ax_pred.axis("off")
            col_idx += 1

            # === Label ===
            ax_label = fig.add_subplot(n, num_cols, i * num_cols + col_idx)
            label_img = y_np[i]  # [H, W, C]
            if label_img.shape[-1] == 1:
                ax_label.imshow(label_img[:, :, 0], cmap="gray", vmin=0, vmax=1)
            else:
                ax_label.imshow(np.clip(label_img, 0, 1))
            ax_label.set_title(f"Label ({t}×{t})", fontsize=10)
            ax_label.axis("off")

        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.15, hspace=0.2)

        # Convert figure to image tensor for logging
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close(fig)

        # Convert to tensor [C, H, W] for make_grid compatibility
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]  # Remove alpha
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        # Log with available logger
        if hasattr(trainer.logger, "log_image"):
            trainer.logger.log_image(key=self.key, images=[img_tensor])
        elif hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
            trainer.logger.experiment.log({self.key: [wandb.Image(img_array)]}, step=trainer.global_step)
        else:
            raise ValueError(f"Unsupported logger: {trainer.logger}")

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Visualize the validation volumes at the end of the epoch."""
        if self.every_n_epochs is None:
            return

        epoch = trainer.current_epoch
        if self.every_n_epochs > 1 and (epoch % self.every_n_epochs != 0):
            return

        event_idx = epoch // max(self.every_n_epochs, 1)
        self._log_volume_grid(trainer, pl_module, event_idx)

    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        """Visualize volumes during training every N steps."""
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
        self._log_volume_grid(trainer, pl_module, event_idx)
