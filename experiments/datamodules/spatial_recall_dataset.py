"""Spatial Recall Dataset and DataModule for PyTorch Lightning.

This module wraps base datamodules (e.g., MNISTDataModule, EMNISTDataModule) to create
spatial recall tasks where images are placed on a larger canvas and the model must
recall the target at a designated readout location (bottom-right corner).

Supports:
    - 2D Spatial Recall: Images placed as 2D patches on 2D canvas
    - 1D Spatial Recall: Images flattened first, then placed as contiguous segments in 1D canvas
    - Fixed placement: Target always at start position
    - Random placement: Target at random valid positions (non-overlapping with readout)
    - Optional mask channel to indicate target location
    - Colored frames mode: RGB canvas with colored bounding boxes around items
    - Multiple items (distractors) on the canvas

Usage:
    # 2D mode
    PYTHONPATH=. python experiments/datamodules/spatial_recall_dataset.py

    # 1D mode
    PYTHONPATH=. python experiments/datamodules/spatial_recall_dataset.py --mode 1d
"""

from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nvsubq_paper.lazy_config import LazyConfig, instantiate


class SpatialRecallDataset(Dataset):
    """Spatial recall dataset wrapper.

    Places images from a base dataset onto a larger canvas. The target must be recalled
    at the bottom-right corner of the canvas (readout region).

    Args:
        base_dataset: Base dataset providing (image, label) pairs. Images should be
            tensors of shape [C, H, W] (typically [1, 28, 28] for MNIST/EMNIST).
        target_size: Size to resize images to before placing on canvas.
        canvas_size: Size of the output canvas (square).
        generator: Random generator for reproducibility.
        placement: Placement mode - "fixed" (top-left) or "random".
        with_mask: If True, add a binary mask channel indicating target location.
        readout_value: Value to fill the readout region with (default 0.0). Use e.g. -1.0 to
            explicitly mark the readout region so the model knows where to output.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_size: int,
        canvas_size: int,
        generator: torch.Generator,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        readout_value: float = 0.0,
    ) -> None:
        """Initialize the SpatialRecallDataset."""
        super().__init__()

        assert canvas_size >= target_size, (
            f"canvas_size must be >= target_size. Got canvas_size={canvas_size}, target_size={target_size}"
        )
        if placement == "random":
            assert canvas_size >= 2 * target_size, (
                f"Random placement requires canvas_size >= 2 * target_size to avoid overlap with readout region. "
                f"Got canvas_size={canvas_size}, target_size={target_size}"
            )

        self.base_dataset = base_dataset
        self.target_size = target_size
        self.canvas_size = canvas_size
        self.generator = generator
        self.placement = placement
        self.with_mask = with_mask
        self.readout_value = readout_value

        # Precompute valid positions for random placement
        if placement == "random":
            self._precompute_valid_positions()

    def _precompute_valid_positions(self) -> None:
        """Precompute grid of valid top-left positions that don't overlap the readout region."""
        C = self.canvas_size
        t = self.target_size
        S = C - t  # Max valid start position
        invalid_start = C - 2 * t  # Positions beyond this overlap with readout

        ys = torch.arange(0, S + 1, dtype=torch.long)
        xs = torch.arange(0, S + 1, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Valid positions: not in the bottom-right quadrant that overlaps readout
        mask_valid = ~((grid_y > invalid_start) & (grid_x > invalid_start))
        self.valid_positions = torch.stack([grid_y[mask_valid], grid_x[mask_valid]], dim=1)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return canvas and target label for the given index."""
        img, _ = self.base_dataset[idx]
        # img: [C, H, W] from base dataset

        # Resize to target size
        target_img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Create canvas
        num_channels = target_img.shape[0]
        canvas = torch.zeros(
            (num_channels, self.canvas_size, self.canvas_size),
            dtype=target_img.dtype,
            device=target_img.device,
        )

        h, w = self.target_size, self.target_size

        # Determine placement position
        if self.placement == "fixed":
            y0, x0 = 0, 0
        else:  # random
            num_pos = self.valid_positions.shape[0]
            idx_pos = int(torch.randint(low=0, high=num_pos, size=(1,), generator=self.generator).item())
            y0, x0 = self.valid_positions[idx_pos].tolist()

        # Place image on canvas
        canvas[:, y0 : y0 + h, x0 : x0 + w] = target_img

        # Fill readout region (bottom-right corner) with readout_value
        # This marks where the model should output the recalled image
        if self.readout_value != 0.0:
            readout_y0 = self.canvas_size - self.target_size
            readout_x0 = self.canvas_size - self.target_size
            canvas[:, readout_y0:, readout_x0:] = self.readout_value

        # Add mask channel if requested
        if self.with_mask:
            mask = torch.zeros(
                (1, self.canvas_size, self.canvas_size),
                dtype=target_img.dtype,
                device=target_img.device,
            )
            mask[:, y0 : y0 + h, x0 : x0 + w] = 1.0
            canvas = torch.cat([canvas, mask], dim=0)

        # Label is the target image (to be recalled at readout location)
        label = target_img

        return canvas, label


class SpatialRecallDataModule(pl.LightningDataModule):
    """Spatial Recall DataModule for PyTorch Lightning.

    Wraps a base datamodule (MNIST, EMNIST, etc.) to create spatial recall tasks where
    images are placed on a canvas and must be recalled at the readout location.

    Args:
        base_datamodule_cfg: A LazyConfig/DictConfig for the base datamodule. The base
            datamodule must have train_dataset, val_dataset, and optionally test_dataset
            attributes after setup().
        target_size: Size to resize images to.
        canvas_size: Size of the output canvas.
        data_type: Output format - "image" ([B, H, W, C]) or "sequence" ([B, L, C]).
        placement: Placement mode - "fixed" or "random".
        with_mask: Add mask channel indicating target location.
        use_colored_frames: Use RGB canvas with colored bounding boxes.
        num_items: Number of items to place (1 = target only, >1 = target + distractors).
        readout_value: Value to fill the readout region with (default 0.0). Use e.g. -1.0 to
            explicitly mark the readout region so the model knows where to output.
            Note: When use_colored_frames=True, the colored border is preserved.
        colored_label: If True and use_colored_frames=True, the label will be RGB with the
            digit colored using the same color as its frame. This creates a "color conditioning"
            task where the model must output the digit in the correct color.
    """

    # Fixed RGB palette for colored frames (8 high-contrast colors)
    PALETTE = torch.tensor(
        [
            [1.00, 0.00, 0.00],  # Red
            [0.00, 0.60, 0.20],  # Green
            [0.00, 0.00, 1.00],  # Blue
            [1.00, 1.00, 0.00],  # Yellow
            [0.00, 0.75, 1.00],  # Cyan
            [1.00, 0.00, 1.00],  # Magenta
            [1.00, 0.50, 0.00],  # Orange
            [0.58, 0.00, 0.83],  # Violet
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        base_datamodule_cfg: LazyConfig,  # LazyConfig[pl.LightningDataModule]
        target_size: int,
        canvas_size: int,
        data_type: Literal["sequence", "image"] = "image",
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        use_colored_frames: bool = False,
        num_items: int = 1,
        readout_value: float = 0.0,
        colored_label: bool = False,
    ) -> None:
        """Initialize the SpatialRecallDataModule."""
        super().__init__()

        # Validate arguments
        assert data_type in ("sequence", "image"), f"data_type must be 'sequence' or 'image', got {data_type}"
        assert placement in ("fixed", "random"), f"placement must be 'fixed' or 'random', got {placement}"
        assert not (with_mask and use_colored_frames), "with_mask and use_colored_frames cannot both be True"
        if colored_label:
            assert use_colored_frames, "colored_label=True requires use_colored_frames=True"
        if num_items > 1:
            assert placement == "random", "num_items > 1 requires placement='random'"
            assert with_mask or use_colored_frames, (
                "num_items > 1 requires with_mask=True or use_colored_frames=True to identify target"
            )
            assert num_items <= len(self.PALETTE), (
                f"num_items must be <= {len(self.PALETTE)} (palette size). Got {num_items}"
            )

        # Store base datamodule config (will be instantiated in setup)
        self._base_datamodule_cfg = base_datamodule_cfg
        self._base_datamodule: Optional[pl.LightningDataModule] = None

        self.target_size = target_size
        self.canvas_size = canvas_size
        self.data_type = data_type
        self.placement = placement
        self.with_mask = with_mask
        self.use_colored_frames = use_colored_frames
        self.num_items = num_items
        self.readout_value = readout_value
        self.colored_label = colored_label

        # These will be set from base datamodule after instantiation
        self._batch_size: Optional[int] = None
        self._num_workers: Optional[int] = None
        self._pin_memory: Optional[bool] = None
        self._seed: Optional[int] = None

        # Create generators (will be re-seeded after we get the seed from base)
        self._generator: Optional[torch.Generator] = None
        self._train_generator: Optional[torch.Generator] = None
        self._val_generator: Optional[torch.Generator] = None
        self._test_generator: Optional[torch.Generator] = None

        # Determine input/output channels
        if use_colored_frames:
            self.input_channels = 3  # RGB
        elif with_mask:
            self.input_channels = 2  # Grayscale + mask
        else:
            self.input_channels = 1  # Grayscale

        # Output channels: 3 if colored_label, otherwise 1 (grayscale)
        self.output_channels = 3 if colored_label else 1

        # Placeholders
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _instantiate_base_datamodule(self) -> pl.LightningDataModule:
        """Instantiate the base datamodule from LazyConfig."""
        self._base_datamodule = instantiate(self._base_datamodule_cfg)
        return self._base_datamodule

    def _extract_base_properties(self) -> None:
        """Extract properties from the base datamodule."""
        base_datamodule = self._base_datamodule
        self._batch_size = base_datamodule.batch_size
        self._num_workers = base_datamodule.num_workers
        self._pin_memory = base_datamodule.pin_memory
        self._seed = base_datamodule.seed

        # Initialize generators with the seed from base
        self._generator = torch.Generator().manual_seed(self._seed)
        self._train_generator = torch.Generator().manual_seed(self._seed + 1000)
        self._val_generator = torch.Generator().manual_seed(self._seed + 2000)
        self._test_generator = torch.Generator().manual_seed(self._seed + 3000)

    @property
    def batch_size(self) -> int:
        """Batch size from base datamodule."""
        if self._batch_size is None:
            raise RuntimeError("Call setup() before accessing batch_size.")
        return self._batch_size

    @property
    def num_workers(self) -> int:
        """Number of workers from base datamodule."""
        if self._num_workers is None:
            raise RuntimeError("Call setup() before accessing num_workers.")
        return self._num_workers

    @property
    def pin_memory(self) -> bool:
        """Pin memory setting from base datamodule."""
        if self._pin_memory is None:
            raise RuntimeError("Call setup() before accessing pin_memory.")
        return self._pin_memory

    @property
    def seed(self) -> int:
        """Seed from base datamodule."""
        if self._seed is None:
            raise RuntimeError("Call setup() before accessing seed.")
        return self._seed

    def prepare_data(self) -> None:
        """Prepare data by calling base datamodule's prepare_data."""
        base = self._instantiate_base_datamodule()
        base.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for the given stage."""
        # Instantiate and setup the base datamodule
        base_datamodule = self._instantiate_base_datamodule()
        base_datamodule.setup(stage)
        self._extract_base_properties()

        # For multi-item or colored frames, we use the base dataset directly
        # and apply transformations in the collate function
        use_simple_dataset = self.num_items == 1 and not self.use_colored_frames

        if stage in ("fit", None):
            base_train_datamodule = base_datamodule.train_dataset
            base_val_datamodule = base_datamodule.val_dataset

            if use_simple_dataset:
                self.train_dataset = SpatialRecallDataset(
                    base_train_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._train_generator,
                    self.placement,
                    self.with_mask,
                    self.readout_value,
                )
                self.val_dataset = SpatialRecallDataset(
                    base_val_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._val_generator,
                    self.placement,
                    self.with_mask,
                    self.readout_value,
                )
            else:
                # For multi-item mode, wrap with simple dataset (no mask) and apply in collate
                # For colored frames, the collate function creates RGB canvas and handles readout_value
                self.train_dataset = SpatialRecallDataset(
                    base_train_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._train_generator,
                    self.placement,
                    with_mask=self.with_mask and not self.use_colored_frames,
                    readout_value=self.readout_value,
                )
                self.val_dataset = SpatialRecallDataset(
                    base_val_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._val_generator,
                    self.placement,
                    with_mask=self.with_mask and not self.use_colored_frames,
                    readout_value=self.readout_value,
                )

        if stage in ("test", None):
            base_test_datamodule = base_datamodule.test_dataset
            if use_simple_dataset:
                self.test_dataset = SpatialRecallDataset(
                    base_test_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._test_generator,
                    self.placement,
                    self.with_mask,
                    self.readout_value,
                )
            else:
                self.test_dataset = SpatialRecallDataset(
                    base_test_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._test_generator,
                    self.placement,
                    with_mask=self.with_mask and not self.use_colored_frames,
                    readout_value=self.readout_value,
                )

    def _multi_item_collate(self, batch: list) -> Tuple[Tensor, Tensor]:
        """Collate function for multi-item mode with mask channel.

        Adds distractor items from the same batch to each canvas.
        """
        xs, ys = zip(*batch)
        xs = [x.clone() for x in xs]
        ys = list(ys)

        batch_size = len(xs)
        t = self.target_size
        C = self.canvas_size

        # Valid positions (precomputed structure)
        S = C - t
        invalid_start = C - 2 * t

        ys_grid = torch.arange(0, S + 1, dtype=torch.long)
        xs_grid = torch.arange(0, S + 1, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(ys_grid, xs_grid, indexing="ij")
        mask_valid = ~((grid_y > invalid_start) & (grid_x > invalid_start))
        valid_positions = torch.stack([grid_y[mask_valid], grid_x[mask_valid]], dim=1)

        g = self._generator

        for i in range(batch_size):
            canvas_i = xs[i]
            occupied = []

            # Find target location from mask channel (channel 1)
            if canvas_i.shape[0] >= 2:
                mask = canvas_i[1]
                nz = (mask > 0).nonzero(as_tuple=False)
                if nz.numel() > 0:
                    y0 = int(nz[:, 0].min().item())
                    x0 = int(nz[:, 1].min().item())
                    occupied.append((y0, y0 + t, x0, x0 + t))

            def overlaps_any(y0: int, x0: int) -> bool:
                y1, x1 = y0 + t, x0 + t
                for oy0, oy1, ox0, ox1 in occupied:
                    if not (y1 <= oy0 or oy1 <= y0 or x1 <= ox0 or ox1 <= x0):
                        return True
                return False

            # Get distractor indices
            max_distractors = max(0, self.num_items - 1)
            all_indices = torch.arange(batch_size, dtype=torch.long)
            other_indices = all_indices[all_indices != i]
            perm_idx = torch.randperm(other_indices.numel(), generator=g)
            distractor_indices = other_indices[perm_idx][:max_distractors]

            # Place distractors
            num_positions = valid_positions.shape[0]
            perm_pos = torch.randperm(num_positions, generator=g)
            pos_cursor = 0

            for j in distractor_indices.tolist():
                placed = False
                attempts = 0
                while attempts < num_positions and pos_cursor < num_positions:
                    y0, x0 = valid_positions[perm_pos[pos_cursor]].tolist()
                    pos_cursor += 1
                    attempts += 1
                    if overlaps_any(y0, x0):
                        continue
                    # Place distractor (intensity only, no mask)
                    canvas_i[0, y0 : y0 + t, x0 : x0 + t] = ys[j][0]
                    occupied.append((y0, y0 + t, x0, x0 + t))
                    placed = True
                    break
                if not placed:
                    break

            xs[i] = canvas_i

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _colored_frames_collate(self, batch: list) -> Tuple[Tensor, Tensor]:
        """Collate function for colored frames mode.

        Creates RGB canvas with colored bounding boxes around each item.
        The readout region gets the same color as the target item.
        """
        xs, ys = zip(*batch)
        xs = list(xs)
        ys = list(ys)

        batch_size = len(xs)
        t = self.target_size
        C = self.canvas_size

        # Valid positions
        S = C - t
        invalid_start = C - 2 * t

        ys_grid = torch.arange(0, S + 1, dtype=torch.long)
        xs_grid = torch.arange(0, S + 1, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(ys_grid, xs_grid, indexing="ij")
        mask_valid = ~((grid_y > invalid_start) & (grid_x > invalid_start))
        valid_positions = torch.stack([grid_y[mask_valid], grid_x[mask_valid]], dim=1)

        palette = self.PALETTE.to(dtype=xs[0].dtype, device=xs[0].device)

        def draw_outline_rgb(canvas_rgb: Tensor, y0: int, x0: int, size: int, color: Tensor) -> None:
            y1 = y0 + size - 1
            x1 = x0 + size - 1
            canvas_rgb[:, y0, x0 : x0 + size] = color.view(3, 1)
            canvas_rgb[:, y1, x0 : x0 + size] = color.view(3, 1)
            canvas_rgb[:, y0 : y0 + size, x0] = color.view(3, 1)
            canvas_rgb[:, y0 : y0 + size, x1] = color.view(3, 1)

        g = self._generator
        x_rgb_list = []
        y_sel_list = []

        for i in range(batch_size):
            canvas_rgb = torch.zeros((3, C, C), dtype=xs[0].dtype, device=xs[0].device)

            # Get items to place
            max_distractors = max(0, self.num_items - 1)
            all_indices = torch.arange(batch_size, dtype=torch.long)
            other_indices = all_indices[all_indices != i]
            if other_indices.numel() > 0 and max_distractors > 0:
                perm_idx = torch.randperm(other_indices.numel(), generator=g)
                distractor_indices = other_indices[perm_idx][:max_distractors]
            else:
                distractor_indices = torch.empty(0, dtype=torch.long)

            indices_to_place = [i] + distractor_indices.tolist()
            color_order = torch.randperm(len(palette), generator=g)[: len(indices_to_place)]

            # Place items
            num_positions = valid_positions.shape[0]
            perm_pos = torch.randperm(num_positions, generator=g)
            pos_cursor = 0
            occupied = []
            placed_meta = []

            def overlaps_any(y0: int, x0: int) -> bool:
                y1, x1 = y0 + t, x0 + t
                for oy0, oy1, ox0, ox1 in occupied:
                    if not (y1 <= oy0 or oy1 <= y0 or x1 <= ox0 or ox1 <= x0):
                        return True
                return False

            for k_idx, j in enumerate(indices_to_place):
                placed = False
                attempts = 0
                while attempts < num_positions and pos_cursor < num_positions:
                    y0, x0 = valid_positions[perm_pos[pos_cursor]].tolist()
                    pos_cursor += 1
                    attempts += 1
                    if overlaps_any(y0, x0):
                        continue

                    # Place grayscale digit as RGB
                    patch = ys[j][0]
                    canvas_rgb[:, y0 : y0 + t, x0 : x0 + t] = patch.unsqueeze(0).repeat(3, 1, 1)

                    # Draw colored bbox
                    cidx = int(color_order[k_idx].item())
                    color = palette[cidx]
                    draw_outline_rgb(canvas_rgb, y0, x0, t, color)

                    occupied.append((y0, y0 + t, x0, x0 + t))
                    placed_meta.append((y0, x0, cidx, j))
                    placed = True
                    break

                if not placed:
                    break

            # Draw readout box with target's color
            target_meta = None
            for py0, px0, pcidx, pj in placed_meta:
                if pj == i:
                    target_meta = (py0, px0, pcidx, pj)
                    break

            if target_meta is not None:
                _, _, sel_cidx, _ = target_meta
                color = palette[sel_cidx]
                y0_readout, x0_readout = C - t, C - t

                # Fill readout region interior with readout_value (if not 0.0)
                # This marks where the model should output the recalled image
                # Interior is the region inside the 1-pixel border
                if self.readout_value != 0.0 and t > 2:
                    # Interior region: skip the 1-pixel border on all sides
                    canvas_rgb[:, y0_readout + 1 : C - 1, x0_readout + 1 : C - 1] = self.readout_value

                draw_outline_rgb(canvas_rgb, y0_readout, x0_readout, t, color)
                x_rgb_list.append(canvas_rgb)

                # Build label: colored (RGB) or grayscale
                if self.colored_label:
                    # Create RGB label: digit intensity * color
                    # ys[i] is [1, H, W], color is [3]
                    # Result: [3, H, W] where each channel = intensity * color_channel
                    label_rgb = ys[i] * color.view(3, 1, 1)
                    y_sel_list.append(label_rgb)
                else:
                    y_sel_list.append(ys[i])
            else:
                # Fallback
                x_rgb_list.append(xs[i].repeat(3, 1, 1) if xs[i].shape[0] == 1 else xs[i][:3])
                if self.colored_label:
                    y_sel_list.append(ys[i].repeat(3, 1, 1))  # Fallback: grayscale as RGB
                else:
                    y_sel_list.append(ys[i])

        return torch.stack(x_rgb_list, dim=0), torch.stack(y_sel_list, dim=0)

    def _get_collate_fn(self):
        """Get the appropriate collate function based on configuration."""
        if self.num_items > 1:
            if self.use_colored_frames:
                return self._colored_frames_collate
            else:
                return self._multi_item_collate
        return None

    def _build_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        """Build a DataLoader with the appropriate collate function."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=self._generator,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._get_collate_fn(),
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train dataloader.")
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val dataloader.")
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test dataloader.")
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(self, batch, dataloader_idx) -> dict:
        """Rearrange batch tensors to expected format.

        For image: [B, C, H, W] -> [B, H, W, C]
        For sequence: [B, C, H, W] -> [B, H*W, C]

        Returns:
            dict: A dictionary with keys "input", "label", and "condition".
        """
        x, y = batch

        if self.data_type == "image":
            x = rearrange(x, "b c h w -> b h w c")
            y = rearrange(y, "b c h w -> b h w c")
        elif self.data_type == "sequence":
            x = rearrange(x, "b c h w -> b (h w) c")
            y = rearrange(y, "b c h w -> b (h w) c")
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

        return {"input": x, "label": y, "condition": None}


# =============================================================================
# 1D Spatial Recall Dataset and DataModule
# =============================================================================
class SpatialRecall1DDataset(Dataset):
    """1D Spatial Recall Dataset.

    Creates a truly 1D spatial recall task where:
    1. Images are resized to target_size × target_size
    2. Images are flattened to a 1D sequence of length target_size²
    3. The flattened image is placed as a contiguous segment in a 1D canvas
    4. The model must recall the flattened image at the readout region (end of canvas)

    This is fundamentally different from flattening a 2D canvas because:
    - In 2D→flatten: 2D spatial locality is partially preserved (row-major order)
    - In true 1D: The image is an unstructured blob identified only by position

    Args:
        base_dataset: Base dataset providing (image, label) pairs.
        target_size: Size to resize images to (target_size × target_size → target_size² elements).
        canvas_length: Length of the 1D canvas. Must be >= 2 * target_size² for random placement.
        generator: Random generator for reproducibility.
        placement: Placement mode - "fixed" (start) or "random".
        with_mask: If True, add a binary mask channel indicating target location.
        readout_value: Value to fill the readout region with (default 0.0). Use e.g. -1.0 to
            explicitly mark the readout region so the model knows where to output.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_size: int,
        canvas_length: int,
        generator: torch.Generator,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        readout_value: float = 0.0,
    ) -> None:
        """Initialize the SpatialRecall1DDataset."""
        super().__init__()

        self.segment_length = target_size * target_size  # Flattened image length

        assert canvas_length >= self.segment_length, (
            f"canvas_length must be >= target_size². "
            f"Got canvas_length={canvas_length}, target_size²={self.segment_length}"
        )
        if placement == "random":
            assert canvas_length >= 2 * self.segment_length, (
                f"Random placement requires canvas_length >= 2 * target_size² to avoid overlap with readout. "
                f"Got canvas_length={canvas_length}, target_size²={self.segment_length}"
            )

        self.base_dataset = base_dataset
        self.target_size = target_size
        self.canvas_length = canvas_length
        self.generator = generator
        self.placement = placement
        self.with_mask = with_mask
        self.readout_value = readout_value

        # Precompute valid positions for random placement
        # Readout region is at the END of the sequence (last segment_length elements)
        # So valid start positions are 0 to (canvas_length - 2 * segment_length)
        if placement == "random":
            max_start = canvas_length - 2 * self.segment_length
            self.valid_positions = torch.arange(0, max_start + 1, dtype=torch.long)
        else:
            self.valid_positions = None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return 1D canvas and flattened target for the given index."""
        img, _ = self.base_dataset[idx]
        # img: [C, H, W] from base dataset

        # Resize to target size
        target_img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # [C, target_size, target_size]

        # Flatten to 1D: [C, target_size, target_size] -> [C, target_size²]
        num_channels = target_img.shape[0]
        target_flat = target_img.view(num_channels, -1)  # [C, segment_length]

        # Create 1D canvas: [C, canvas_length]
        canvas = torch.zeros(
            (num_channels, self.canvas_length),
            dtype=target_flat.dtype,
            device=target_flat.device,
        )

        # Determine placement position
        if self.placement == "fixed":
            pos = 0
        else:  # random
            num_pos = self.valid_positions.shape[0]
            idx_pos = int(torch.randint(low=0, high=num_pos, size=(1,), generator=self.generator).item())
            pos = int(self.valid_positions[idx_pos].item())

        # Place flattened image in canvas
        canvas[:, pos : pos + self.segment_length] = target_flat

        # Fill readout region (last segment_length elements) with readout_value
        # This marks where the model should output the recalled image
        readout_start = self.canvas_length - self.segment_length
        if self.readout_value != 0.0:
            canvas[:, readout_start:] = self.readout_value

        # Add mask channel if requested
        if self.with_mask:
            mask = torch.zeros(
                (1, self.canvas_length),
                dtype=target_flat.dtype,
                device=target_flat.device,
            )
            mask[:, pos : pos + self.segment_length] = 1.0
            canvas = torch.cat([canvas, mask], dim=0)

        # Label is the flattened target image
        label = target_flat

        return canvas, label


class SpatialRecall1DDataModule(pl.LightningDataModule):
    """1D Spatial Recall DataModule for PyTorch Lightning.

    Wraps a base datamodule to create 1D spatial recall tasks where flattened images
    are placed in a 1D canvas and must be recalled.

    Args:
        base_datamodule_cfg: A LazyConfig for the base datamodule.
        target_size: Size to resize images to (becomes target_size² length segment).
        canvas_size: Size of the canvas per dimension (canvas_length = canvas_size²).
        placement: Placement mode - "fixed" or "random".
        with_mask: Add mask channel indicating target location.
        num_items: Number of items to place (1 = target only, >1 = target + distractors).
        readout_value: Value to fill the readout region with (default 0.0). Use e.g. -1.0 to
            explicitly mark the readout region so the model knows where to output.
    """

    def __init__(
        self,
        base_datamodule_cfg: LazyConfig,
        target_size: int,
        canvas_size: int,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        num_items: int = 1,
        readout_value: float = 0.0,
    ) -> None:
        """Initialize the SpatialRecall1DDataModule."""
        super().__init__()

        assert placement in ("fixed", "random"), f"placement must be 'fixed' or 'random', got {placement}"
        if num_items > 1:
            assert placement == "random", "num_items > 1 requires placement='random'"
            assert with_mask, "num_items > 1 requires with_mask=True to identify target"

        self._base_datamodule_cfg = base_datamodule_cfg
        self._base_datamodule: Optional[pl.LightningDataModule] = None

        self.target_size = target_size
        self.canvas_size = canvas_size
        self.canvas_length = canvas_size * canvas_size  # Computed from canvas_size
        self.segment_length = target_size * target_size
        self.placement = placement
        self.with_mask = with_mask
        self.num_items = num_items
        self.readout_value = readout_value

        # Properties from base datamodule
        self._batch_size: Optional[int] = None
        self._num_workers: Optional[int] = None
        self._pin_memory: Optional[bool] = None
        self._seed: Optional[int] = None

        # Generators
        self._generator: Optional[torch.Generator] = None
        self._train_generator: Optional[torch.Generator] = None
        self._val_generator: Optional[torch.Generator] = None
        self._test_generator: Optional[torch.Generator] = None

        # Input/output channels
        if with_mask:
            self.input_channels = 2  # Grayscale + mask
        else:
            self.input_channels = 1  # Grayscale
        self.output_channels = 1

        # Datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _instantiate_base_datamodule(self) -> pl.LightningDataModule:
        """Instantiate the base datamodule from LazyConfig."""
        self._base_datamodule = instantiate(self._base_datamodule_cfg)
        return self._base_datamodule

    def _extract_base_properties(self) -> None:
        """Extract properties from the base datamodule."""
        base = self._base_datamodule
        self._batch_size = base.batch_size
        self._num_workers = base.num_workers
        self._pin_memory = base.pin_memory
        self._seed = base.seed

        self._generator = torch.Generator().manual_seed(self._seed)
        self._train_generator = torch.Generator().manual_seed(self._seed + 1000)
        self._val_generator = torch.Generator().manual_seed(self._seed + 2000)
        self._test_generator = torch.Generator().manual_seed(self._seed + 3000)

    @property
    def batch_size(self) -> int:
        """Batch size from base datamodule."""
        if self._batch_size is None:
            raise RuntimeError("Call setup() before accessing batch_size.")
        return self._batch_size

    @property
    def num_workers(self) -> int:
        """Number of workers from base datamodule."""
        if self._num_workers is None:
            raise RuntimeError("Call setup() before accessing num_workers.")
        return self._num_workers

    @property
    def pin_memory(self) -> bool:
        """Pin memory setting from base datamodule."""
        if self._pin_memory is None:
            raise RuntimeError("Call setup() before accessing pin_memory.")
        return self._pin_memory

    @property
    def seed(self) -> int:
        """Seed from base datamodule."""
        if self._seed is None:
            raise RuntimeError("Call setup() before accessing seed.")
        return self._seed

    def prepare_data(self) -> None:
        """Prepare data by calling base datamodule's prepare_data."""
        base = self._instantiate_base_datamodule()
        base.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for the given stage."""
        base = self._instantiate_base_datamodule()
        base.setup(stage)
        self._extract_base_properties()

        if stage in ("fit", None):
            self.train_dataset = SpatialRecall1DDataset(
                base.train_dataset,
                self.target_size,
                self.canvas_length,
                self._train_generator,
                self.placement,
                self.with_mask,
                self.readout_value,
            )
            self.val_dataset = SpatialRecall1DDataset(
                base.val_dataset,
                self.target_size,
                self.canvas_length,
                self._val_generator,
                self.placement,
                self.with_mask,
                self.readout_value,
            )

        if stage in ("test", None):
            self.test_dataset = SpatialRecall1DDataset(
                base.test_dataset,
                self.target_size,
                self.canvas_length,
                self._test_generator,
                self.placement,
                self.with_mask,
                self.readout_value,
            )

    def _multi_item_collate(self, batch: list) -> Tuple[Tensor, Tensor]:
        """Collate function for multi-item mode with mask channel."""
        xs, ys = zip(*batch)
        xs = [x.clone() for x in xs]
        ys = list(ys)

        batch_size = len(xs)
        seg_len = self.segment_length
        L = self.canvas_length

        # Valid positions (readout at end, so valid start is 0 to L - 2*seg_len)
        max_start = L - 2 * seg_len
        valid_positions = torch.arange(0, max_start + 1, dtype=torch.long)

        g = self._generator

        for i in range(batch_size):
            canvas_i = xs[i]  # [C, L] where C=2 (intensity + mask)
            occupied = []

            # Find target location from mask channel
            if canvas_i.shape[0] >= 2:
                mask = canvas_i[1]  # [L]
                nz = (mask > 0).nonzero(as_tuple=False)
                if nz.numel() > 0:
                    start = int(nz.min().item())
                    end = int(nz.max().item()) + 1
                    occupied.append((start, end))

            def overlaps_any(pos: int) -> bool:
                p_end = pos + seg_len
                for o_start, o_end in occupied:
                    if not (p_end <= o_start or o_end <= pos):
                        return True
                return False

            # Get distractor indices
            max_distractors = max(0, self.num_items - 1)
            all_indices = torch.arange(batch_size, dtype=torch.long)
            other_indices = all_indices[all_indices != i]
            perm_idx = torch.randperm(other_indices.numel(), generator=g)
            distractor_indices = other_indices[perm_idx][:max_distractors]

            # Place distractors
            num_positions = valid_positions.shape[0]
            perm_pos = torch.randperm(num_positions, generator=g)
            pos_cursor = 0

            for j in distractor_indices.tolist():
                placed = False
                attempts = 0
                while attempts < num_positions and pos_cursor < num_positions:
                    pos = int(valid_positions[perm_pos[pos_cursor]].item())
                    pos_cursor += 1
                    attempts += 1
                    if overlaps_any(pos):
                        continue
                    # Place distractor intensity only (no mask)
                    canvas_i[0, pos : pos + seg_len] = ys[j][0]
                    occupied.append((pos, pos + seg_len))
                    placed = True
                    break
                if not placed:
                    break

            xs[i] = canvas_i

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _get_collate_fn(self):
        """Get the appropriate collate function based on configuration."""
        if self.num_items > 1:
            return self._multi_item_collate
        return None

    def _build_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        """Build a DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=self._generator,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._get_collate_fn(),
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train dataloader.")
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val dataloader.")
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test dataloader.")
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(self, batch, dataloader_idx) -> dict:
        """Rearrange batch tensors to expected format.

        Input: [B, C, L] -> [B, L, C]
        Label: [B, C, segment_length] -> [B, segment_length, C]

        Returns:
            dict: A dictionary with keys "input", "label", and "condition".
        """
        x, y = batch

        # [B, C, L] -> [B, L, C]
        x = rearrange(x, "b c l -> b l c")
        # [B, C, segment_length] -> [B, segment_length, C]
        y = rearrange(y, "b c l -> b l c")

        return {"input": x, "label": y, "condition": None}


# =============================================================================
# 3D Spatial Recall Dataset and DataModule
# =============================================================================
class SpatialRecall3DDataset(Dataset):
    """3D Spatial Recall Dataset.

    Creates a 3D spatial recall task where:
    1. Images are resized to target_size × target_size (2D)
    2. The 2D image is placed on a depth slice of a 3D canvas
    3. The model must recall the image at the readout region (back-bottom-right corner)

    The 3D canvas has shape [C, D, H, W] where:
    - D = canvas_depth (depth dimension)
    - H = W = canvas_size (spatial dimensions)

    The readout region is at the last depth slice (back plane), bottom-right corner.

    Args:
        base_dataset: Base dataset providing (image, label) pairs.
        target_size: Size to resize images to (target_size × target_size).
        canvas_size: Size of the canvas in H and W dimensions.
        canvas_depth: Size of the canvas in D dimension.
        generator: Random generator for reproducibility.
        placement: Placement mode - "fixed" (front-top-left) or "random".
        with_mask: If True, add a binary mask channel indicating target location.
        readout_value: Value to fill the readout region with (default 0.0).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_size: int,
        canvas_size: int,
        canvas_depth: int,
        generator: torch.Generator,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        readout_value: float = 0.0,
    ) -> None:
        """Initialize the SpatialRecall3DDataset."""
        super().__init__()

        assert canvas_size >= target_size, (
            f"canvas_size must be >= target_size. Got canvas_size={canvas_size}, target_size={target_size}"
        )
        assert canvas_depth >= 1, f"canvas_depth must be >= 1. Got canvas_depth={canvas_depth}"

        if placement == "random":
            # For random placement, we need space to place image without overlapping readout
            # Readout is at depth D-1, bottom-right corner
            # Valid positions: any depth slice where the image doesn't overlap readout
            assert canvas_size >= 2 * target_size, (
                f"Random placement requires canvas_size >= 2 * target_size. "
                f"Got canvas_size={canvas_size}, target_size={target_size}"
            )
            # With depth >= 2, we can place on any slice except the last one freely
            # With depth == 1, we need spatial separation (handled by canvas_size constraint)

        self.base_dataset = base_dataset
        self.target_size = target_size
        self.canvas_size = canvas_size
        self.canvas_depth = canvas_depth
        self.generator = generator
        self.placement = placement
        self.with_mask = with_mask
        self.readout_value = readout_value

        # Precompute valid positions for random placement
        if placement == "random":
            self._precompute_valid_positions()

    def _precompute_valid_positions(self) -> None:
        """Precompute grid of valid (d, y, x) positions that don't overlap the readout region.

        Readout region is at: depth=D-1, y=[H-t:H], x=[W-t:W]
        Valid positions are those where the placed image doesn't overlap with readout.
        """
        D = self.canvas_depth
        C = self.canvas_size
        t = self.target_size

        # For each depth slice, compute valid (y, x) positions
        # On the last depth slice (d = D-1), we need to avoid bottom-right corner
        # On other slices, all positions are valid

        valid_list = []

        S = C - t  # Max valid start position in y or x
        invalid_start = C - 2 * t  # Positions beyond this overlap with readout (on last slice)

        for d in range(D):
            ys = torch.arange(0, S + 1, dtype=torch.long)
            xs = torch.arange(0, S + 1, dtype=torch.long)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

            if d == D - 1:
                # Last depth slice: avoid bottom-right quadrant
                mask_valid = ~((grid_y > invalid_start) & (grid_x > invalid_start))
            else:
                # Other slices: all positions valid
                mask_valid = torch.ones_like(grid_y, dtype=torch.bool)

            valid_y = grid_y[mask_valid]
            valid_x = grid_x[mask_valid]
            valid_d = torch.full_like(valid_y, d)

            valid_list.append(torch.stack([valid_d, valid_y, valid_x], dim=1))

        self.valid_positions = torch.cat(valid_list, dim=0)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return 3D canvas and target label for the given index."""
        img, _ = self.base_dataset[idx]
        # img: [C, H, W] from base dataset

        # Resize to target size
        target_img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # [C, target_size, target_size]

        # Create 3D canvas: [C, D, H, W]
        num_channels = target_img.shape[0]
        canvas = torch.zeros(
            (num_channels, self.canvas_depth, self.canvas_size, self.canvas_size),
            dtype=target_img.dtype,
            device=target_img.device,
        )

        h, w = self.target_size, self.target_size

        # Determine placement position (d, y, x)
        if self.placement == "fixed":
            d0, y0, x0 = 0, 0, 0  # Front-top-left corner
        else:  # random
            num_pos = self.valid_positions.shape[0]
            idx_pos = int(torch.randint(low=0, high=num_pos, size=(1,), generator=self.generator).item())
            d0, y0, x0 = self.valid_positions[idx_pos].tolist()

        # Place image on canvas (2D image on a depth slice)
        canvas[:, d0, y0 : y0 + h, x0 : x0 + w] = target_img

        # Fill readout region (back-bottom-right corner) with readout_value
        # Readout is at: depth=D-1, y=[H-t:H], x=[W-t:W]
        if self.readout_value != 0.0:
            readout_d = self.canvas_depth - 1
            readout_y0 = self.canvas_size - self.target_size
            readout_x0 = self.canvas_size - self.target_size
            canvas[:, readout_d, readout_y0:, readout_x0:] = self.readout_value

        # Add mask channel if requested
        if self.with_mask:
            mask = torch.zeros(
                (1, self.canvas_depth, self.canvas_size, self.canvas_size),
                dtype=target_img.dtype,
                device=target_img.device,
            )
            mask[:, d0, y0 : y0 + h, x0 : x0 + w] = 1.0
            canvas = torch.cat([canvas, mask], dim=0)

        # Label is the target image (to be recalled at readout location)
        label = target_img

        return canvas, label


class SpatialRecall3DDataModule(pl.LightningDataModule):
    """3D Spatial Recall DataModule for PyTorch Lightning.

    Wraps a base datamodule to create 3D spatial recall tasks where 2D images
    are placed on depth slices of a 3D canvas and must be recalled.

    The 3D canvas has shape [C, D, H, W] where:
    - D = canvas_depth (depth dimension, handled separately)
    - H = W = canvas_size (spatial dimensions)

    When converted to sequence format, uses depth-first ordering:
    [C, D, H, W] -> [D*H*W, C]

    Args:
        base_datamodule_cfg: A LazyConfig for the base datamodule.
        target_size: Size to resize images to (becomes target_size × target_size).
        canvas_size: Size of the canvas in H and W dimensions.
        canvas_depth: Size of the canvas in D dimension.
        data_type: Output format - "volume" ([B, D, H, W, C]) or "sequence" ([B, D*H*W, C]).
        placement: Placement mode - "fixed" or "random".
        with_mask: Add mask channel indicating target location.
        num_items: Number of items to place (1 = target only, >1 = target + distractors).
        readout_value: Value to fill the readout region with (default 0.0).
    """

    # Use the same palette as 2D
    PALETTE = SpatialRecallDataModule.PALETTE

    def __init__(
        self,
        base_datamodule_cfg: LazyConfig,
        target_size: int,
        canvas_size: int,
        canvas_depth: int,
        data_type: Literal["sequence", "volume"] = "volume",
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        use_colored_frames: bool = False,
        num_items: int = 1,
        readout_value: float = 0.0,
        colored_label: bool = False,
    ) -> None:
        """Initialize the SpatialRecall3DDataModule."""
        super().__init__()

        assert data_type in ("sequence", "volume"), f"data_type must be 'sequence' or 'volume', got {data_type}"
        assert placement in ("fixed", "random"), f"placement must be 'fixed' or 'random', got {placement}"
        assert not (with_mask and use_colored_frames), "with_mask and use_colored_frames cannot both be True"
        if colored_label:
            assert use_colored_frames, "colored_label=True requires use_colored_frames=True"
        if num_items > 1:
            assert placement == "random", "num_items > 1 requires placement='random'"
            assert with_mask or use_colored_frames, (
                "num_items > 1 requires with_mask=True or use_colored_frames=True to identify target"
            )
            assert num_items <= len(self.PALETTE), (
                f"num_items must be <= {len(self.PALETTE)} (palette size). Got {num_items}"
            )

        self._base_datamodule_cfg = base_datamodule_cfg
        self._base_datamodule: Optional[pl.LightningDataModule] = None

        self.target_size = target_size
        self.canvas_size = canvas_size
        self.canvas_depth = canvas_depth
        self.data_type = data_type
        self.placement = placement
        self.with_mask = with_mask
        self.use_colored_frames = use_colored_frames
        self.num_items = num_items
        self.readout_value = readout_value
        self.colored_label = colored_label

        # Computed properties
        self.canvas_volume = canvas_depth * canvas_size * canvas_size

        # Properties from base datamodule
        self._batch_size: Optional[int] = None
        self._num_workers: Optional[int] = None
        self._pin_memory: Optional[bool] = None
        self._seed: Optional[int] = None

        # Generators
        self._generator: Optional[torch.Generator] = None
        self._train_generator: Optional[torch.Generator] = None
        self._val_generator: Optional[torch.Generator] = None
        self._test_generator: Optional[torch.Generator] = None

        # Input/output channels
        if use_colored_frames:
            self.input_channels = 3  # RGB
        elif with_mask:
            self.input_channels = 2  # Grayscale + mask
        else:
            self.input_channels = 1  # Grayscale

        # Output channels: 3 if colored_label, otherwise 1 (grayscale)
        self.output_channels = 3 if colored_label else 1

        # Datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _instantiate_base_datamodule(self) -> pl.LightningDataModule:
        """Instantiate the base datamodule from LazyConfig."""
        self._base_datamodule = instantiate(self._base_datamodule_cfg)
        return self._base_datamodule

    def _extract_base_properties(self) -> None:
        """Extract properties from the base datamodule."""
        base = self._base_datamodule
        self._batch_size = base.batch_size
        self._num_workers = base.num_workers
        self._pin_memory = base.pin_memory
        self._seed = base.seed

        self._generator = torch.Generator().manual_seed(self._seed)
        self._train_generator = torch.Generator().manual_seed(self._seed + 1000)
        self._val_generator = torch.Generator().manual_seed(self._seed + 2000)
        self._test_generator = torch.Generator().manual_seed(self._seed + 3000)

    @property
    def batch_size(self) -> int:
        """Batch size from base datamodule."""
        if self._batch_size is None:
            raise RuntimeError("Call setup() before accessing batch_size.")
        return self._batch_size

    @property
    def num_workers(self) -> int:
        """Number of workers from base datamodule."""
        if self._num_workers is None:
            raise RuntimeError("Call setup() before accessing num_workers.")
        return self._num_workers

    @property
    def pin_memory(self) -> bool:
        """Pin memory setting from base datamodule."""
        if self._pin_memory is None:
            raise RuntimeError("Call setup() before accessing pin_memory.")
        return self._pin_memory

    @property
    def seed(self) -> int:
        """Seed from base datamodule."""
        if self._seed is None:
            raise RuntimeError("Call setup() before accessing seed.")
        return self._seed

    def prepare_data(self) -> None:
        """Prepare data by calling base datamodule's prepare_data."""
        base = self._instantiate_base_datamodule()
        base.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for the given stage."""
        base = self._instantiate_base_datamodule()
        base.setup(stage)
        self._extract_base_properties()

        if stage in ("fit", None):
            self.train_dataset = SpatialRecall3DDataset(
                base.train_dataset,
                self.target_size,
                self.canvas_size,
                self.canvas_depth,
                self._train_generator,
                self.placement,
                self.with_mask,
                self.readout_value,
            )
            self.val_dataset = SpatialRecall3DDataset(
                base.val_dataset,
                self.target_size,
                self.canvas_size,
                self.canvas_depth,
                self._val_generator,
                self.placement,
                self.with_mask,
                self.readout_value,
            )

        if stage in ("test", None):
            self.test_dataset = SpatialRecall3DDataset(
                base.test_dataset,
                self.target_size,
                self.canvas_size,
                self.canvas_depth,
                self._test_generator,
                self.placement,
                self.with_mask,
                self.readout_value,
            )

    def _multi_item_collate(self, batch: list) -> Tuple[Tensor, Tensor]:
        """Collate function for multi-item mode with mask channel."""
        xs, ys = zip(*batch)
        xs = [x.clone() for x in xs]
        ys = list(ys)

        batch_size = len(xs)
        t = self.target_size
        C = self.canvas_size
        D = self.canvas_depth

        # Precompute valid positions
        S = C - t
        invalid_start = C - 2 * t

        valid_list = []
        for d in range(D):
            ys_grid = torch.arange(0, S + 1, dtype=torch.long)
            xs_grid = torch.arange(0, S + 1, dtype=torch.long)
            grid_y, grid_x = torch.meshgrid(ys_grid, xs_grid, indexing="ij")

            if d == D - 1:
                mask_valid = ~((grid_y > invalid_start) & (grid_x > invalid_start))
            else:
                mask_valid = torch.ones_like(grid_y, dtype=torch.bool)

            valid_y = grid_y[mask_valid]
            valid_x = grid_x[mask_valid]
            valid_d = torch.full_like(valid_y, d)
            valid_list.append(torch.stack([valid_d, valid_y, valid_x], dim=1))

        valid_positions = torch.cat(valid_list, dim=0)

        g = self._generator

        for i in range(batch_size):
            canvas_i = xs[i]  # [C, D, H, W] where C includes mask channel
            occupied = []

            # Find target location from mask channel
            if canvas_i.shape[0] >= 2:
                mask = canvas_i[1]  # [D, H, W]
                nz = (mask > 0).nonzero(as_tuple=False)
                if nz.numel() > 0:
                    d0 = int(nz[:, 0].min().item())
                    y0 = int(nz[:, 1].min().item())
                    x0 = int(nz[:, 2].min().item())
                    occupied.append((d0, y0, y0 + t, x0, x0 + t))

            def overlaps_any(d: int, y0: int, x0: int) -> bool:
                y1, x1 = y0 + t, x0 + t
                for od, oy0, oy1, ox0, ox1 in occupied:
                    if d == od:  # Same depth slice
                        if not (y1 <= oy0 or oy1 <= y0 or x1 <= ox0 or ox1 <= x0):
                            return True
                return False

            # Get distractor indices
            max_distractors = max(0, self.num_items - 1)
            all_indices = torch.arange(batch_size, dtype=torch.long)
            other_indices = all_indices[all_indices != i]
            perm_idx = torch.randperm(other_indices.numel(), generator=g)
            distractor_indices = other_indices[perm_idx][:max_distractors]

            # Place distractors
            num_positions = valid_positions.shape[0]
            perm_pos = torch.randperm(num_positions, generator=g)
            pos_cursor = 0

            for j in distractor_indices.tolist():
                placed = False
                attempts = 0
                while attempts < num_positions and pos_cursor < num_positions:
                    d0, y0, x0 = valid_positions[perm_pos[pos_cursor]].tolist()
                    pos_cursor += 1
                    attempts += 1
                    if overlaps_any(d0, y0, x0):
                        continue
                    # Place distractor (intensity only, no mask)
                    canvas_i[0, d0, y0 : y0 + t, x0 : x0 + t] = ys[j][0]
                    occupied.append((d0, y0, y0 + t, x0, x0 + t))
                    placed = True
                    break
                if not placed:
                    break

            xs[i] = canvas_i

        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def _colored_frames_collate(self, batch: list) -> Tuple[Tensor, Tensor]:
        """Collate function for colored frames mode in 3D.

        Creates RGB 3D canvas with colored bounding boxes around each item.
        The readout region gets the same color as the target item.
        """
        xs, ys = zip(*batch)
        xs = list(xs)
        ys = list(ys)

        batch_size = len(xs)
        t = self.target_size
        C = self.canvas_size
        D = self.canvas_depth

        # Valid positions
        S = C - t
        invalid_start = C - 2 * t

        valid_list = []
        for d in range(D):
            ys_grid = torch.arange(0, S + 1, dtype=torch.long)
            xs_grid = torch.arange(0, S + 1, dtype=torch.long)
            grid_y, grid_x = torch.meshgrid(ys_grid, xs_grid, indexing="ij")

            if d == D - 1:
                mask_valid = ~((grid_y > invalid_start) & (grid_x > invalid_start))
            else:
                mask_valid = torch.ones_like(grid_y, dtype=torch.bool)

            valid_y = grid_y[mask_valid]
            valid_x = grid_x[mask_valid]
            valid_d = torch.full_like(valid_y, d)
            valid_list.append(torch.stack([valid_d, valid_y, valid_x], dim=1))

        valid_positions = torch.cat(valid_list, dim=0)

        palette = self.PALETTE.to(dtype=xs[0].dtype, device=xs[0].device)

        def draw_outline_rgb_3d(canvas_rgb: Tensor, d: int, y0: int, x0: int, size: int, color: Tensor) -> None:
            """Draw colored outline on a depth slice."""
            y1 = y0 + size - 1
            x1 = x0 + size - 1
            canvas_rgb[:, d, y0, x0 : x0 + size] = color.view(3, 1)
            canvas_rgb[:, d, y1, x0 : x0 + size] = color.view(3, 1)
            canvas_rgb[:, d, y0 : y0 + size, x0] = color.view(3, 1)
            canvas_rgb[:, d, y0 : y0 + size, x1] = color.view(3, 1)

        g = self._generator
        x_rgb_list = []
        y_sel_list = []

        for i in range(batch_size):
            canvas_rgb = torch.zeros((3, D, C, C), dtype=xs[0].dtype, device=xs[0].device)

            # Get items to place
            max_distractors = max(0, self.num_items - 1)
            all_indices = torch.arange(batch_size, dtype=torch.long)
            other_indices = all_indices[all_indices != i]
            if other_indices.numel() > 0 and max_distractors > 0:
                perm_idx = torch.randperm(other_indices.numel(), generator=g)
                distractor_indices = other_indices[perm_idx][:max_distractors]
            else:
                distractor_indices = torch.empty(0, dtype=torch.long)

            indices_to_place = [i] + distractor_indices.tolist()
            color_order = torch.randperm(len(palette), generator=g)[: len(indices_to_place)]

            # Place items
            num_positions = valid_positions.shape[0]
            perm_pos = torch.randperm(num_positions, generator=g)
            pos_cursor = 0
            occupied = []
            placed_meta = []

            def overlaps_any(d: int, y0: int, x0: int) -> bool:
                y1, x1 = y0 + t, x0 + t
                for od, oy0, oy1, ox0, ox1 in occupied:
                    if d == od:
                        if not (y1 <= oy0 or oy1 <= y0 or x1 <= ox0 or ox1 <= x0):
                            return True
                return False

            for k_idx, j in enumerate(indices_to_place):
                placed = False
                attempts = 0
                while attempts < num_positions and pos_cursor < num_positions:
                    d0, y0, x0 = valid_positions[perm_pos[pos_cursor]].tolist()
                    pos_cursor += 1
                    attempts += 1
                    if overlaps_any(d0, y0, x0):
                        continue

                    # Place grayscale digit as RGB on depth slice
                    patch = ys[j][0]  # [H, W]
                    canvas_rgb[:, d0, y0 : y0 + t, x0 : x0 + t] = patch.unsqueeze(0).repeat(3, 1, 1)

                    # Draw colored bbox
                    cidx = int(color_order[k_idx].item())
                    color = palette[cidx]
                    draw_outline_rgb_3d(canvas_rgb, d0, y0, x0, t, color)

                    occupied.append((d0, y0, y0 + t, x0, x0 + t))
                    placed_meta.append((d0, y0, x0, cidx, j))
                    placed = True
                    break

                if not placed:
                    break

            # Draw readout box on back slice with target's color
            target_meta = None
            for pd, py0, px0, pcidx, pj in placed_meta:
                if pj == i:
                    target_meta = (pd, py0, px0, pcidx, pj)
                    break

            if target_meta is not None:
                _, _, _, sel_cidx, _ = target_meta
                color = palette[sel_cidx]
                y0_readout, x0_readout = C - t, C - t
                d_readout = D - 1

                # Fill readout region interior with readout_value
                if self.readout_value != 0.0 and t > 2:
                    canvas_rgb[:, d_readout, y0_readout + 1 : C - 1, x0_readout + 1 : C - 1] = self.readout_value

                draw_outline_rgb_3d(canvas_rgb, d_readout, y0_readout, x0_readout, t, color)
                x_rgb_list.append(canvas_rgb)

                # Build label: colored (RGB) or grayscale
                if self.colored_label:
                    label_rgb = ys[i] * color.view(3, 1, 1)
                    y_sel_list.append(label_rgb)
                else:
                    y_sel_list.append(ys[i])
            else:
                # Fallback
                x_rgb_list.append(canvas_rgb)
                if self.colored_label:
                    y_sel_list.append(ys[i].repeat(3, 1, 1))
                else:
                    y_sel_list.append(ys[i])

        return torch.stack(x_rgb_list, dim=0), torch.stack(y_sel_list, dim=0)

    def _get_collate_fn(self):
        """Get the appropriate collate function based on configuration."""
        if self.num_items > 1:
            if self.use_colored_frames:
                return self._colored_frames_collate
            else:
                return self._multi_item_collate
        return None

    def _build_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        """Build a DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=self._generator,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._get_collate_fn(),
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train dataloader.")
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val dataloader.")
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test dataloader.")
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(self, batch, dataloader_idx) -> dict:
        """Rearrange batch tensors to expected format.

        For volume: [B, C, D, H, W] -> [B, D, H, W, C]
        For sequence: [B, C, D, H, W] -> [B, D*H*W, C] (depth-first ordering)

        Returns:
            dict: A dictionary with keys "input", "label", and "condition".
        """
        x, y = batch

        if self.data_type == "volume":
            # [B, C, D, H, W] -> [B, D, H, W, C]
            x = rearrange(x, "b c d h w -> b d h w c")
            # Label stays as [B, C, H, W] -> [B, H, W, C]
            y = rearrange(y, "b c h w -> b h w c")
        elif self.data_type == "sequence":
            # [B, C, D, H, W] -> [B, D*H*W, C] (depth-first)
            x = rearrange(x, "b c d h w -> b (d h w) c")
            # Label: [B, C, H, W] -> [B, H*W, C]
            y = rearrange(y, "b c h w -> b (h w) c")
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

        return {"input": x, "label": y, "condition": None}


if __name__ == "__main__":
    import argparse
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.datamodules.emnist import EMNISTDataModule
    from experiments.datamodules.mnist import MNISTDataModule
    from nvsubq_paper.lazy_config import LazyConfig

    parser = argparse.ArgumentParser(description="Visualize Spatial Recall samples")
    parser.add_argument(
        "--mode", type=str, default="2d", choices=["1d", "2d", "3d"], help="1D, 2D or 3D spatial recall"
    )
    parser.add_argument("--placement", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--with-mask", action="store_true", help="Add mask channel")
    parser.add_argument("--colored-frames", action="store_true", help="Use colored frames (RGB, 2D only)")
    parser.add_argument("--num-items", type=int, default=1, help="Number of items on canvas")
    parser.add_argument("--target-size", type=int, default=16, help="Target image size")
    parser.add_argument("--canvas-size", type=int, default=64, help="Canvas size (H, W dimensions)")
    parser.add_argument("--canvas-depth", type=int, default=8, help="Canvas depth (3D only)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="./.data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="_tmp", help="Output directory")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "emnist"])
    parser.add_argument(
        "--emnist-split", type=str, default="digits", choices=["digits", "letters", "balanced", "bymerge", "byclass"]
    )
    parser.add_argument("--readout-value", type=float, default=0.0, help="Value to fill readout region (default 0.0)")
    parser.add_argument(
        "--colored-label",
        action="store_true",
        help="Output RGB label colored with frame color (requires --colored-frames)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.mode == "1d" and args.colored_frames:
        print("Warning: --colored-frames is only supported in 2D and 3D modes. Ignoring.")
        args.colored_frames = False
    if args.colored_label and not args.colored_frames:
        print("Warning: --colored-label requires --colored-frames. Enabling --colored-frames.")
        args.colored_frames = True

    torch.manual_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create base datamodule config using LazyConfig
    if args.dataset == "mnist":
        base_datamodule_cfg = LazyConfig(MNISTDataModule)(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            data_type="image",
            num_workers=0,
            pin_memory=True,
            use_deterministic_worker_init=True,
            seed=42,
            task="classification",
        )
    else:
        base_datamodule_cfg = LazyConfig(EMNISTDataModule)(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            data_type="image",
            num_workers=0,
            pin_memory=False,
            permuted=False,
            seed=42,
            normalize_input=True,
            split=args.emnist_split,
        )

    # Create spatial recall datamodule wrapping the base
    if args.mode == "2d":
        dm = SpatialRecallDataModule(
            base_datamodule_cfg=base_datamodule_cfg,
            target_size=args.target_size,
            canvas_size=args.canvas_size,
            data_type="image",
            placement=args.placement,
            with_mask=args.with_mask,
            use_colored_frames=args.colored_frames,
            num_items=args.num_items,
            readout_value=args.readout_value,
            colored_label=args.colored_label,
        )
    elif args.mode == "3d":
        dm = SpatialRecall3DDataModule(
            base_datamodule_cfg=base_datamodule_cfg,
            target_size=args.target_size,
            canvas_size=args.canvas_size,
            canvas_depth=args.canvas_depth,
            data_type="volume",
            placement=args.placement,
            with_mask=args.with_mask,
            use_colored_frames=args.colored_frames,
            num_items=args.num_items,
            readout_value=args.readout_value,
            colored_label=args.colored_label,
        )
    else:  # 1D mode
        dm = SpatialRecall1DDataModule(
            base_datamodule_cfg=base_datamodule_cfg,
            target_size=args.target_size,
            canvas_size=args.canvas_size,  # DataModule computes canvas_length = canvas_size²
            placement=args.placement,
            with_mask=args.with_mask,
            num_items=args.num_items,
            readout_value=args.readout_value,
        )

    dm.prepare_data()
    dm.setup("fit")

    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "emnist":
        print(f"EMNIST split: {args.emnist_split}")
    print(f"Placement: {args.placement}")
    print(f"With mask: {args.with_mask}")
    if args.mode in ("2d", "3d"):
        print(f"Colored frames: {args.colored_frames}")
        print(f"Colored label: {args.colored_label}")
    if args.readout_value != 0:
        print(f"Readout value: {args.readout_value}")
    print(f"Num items: {args.num_items}")
    print(f"Target size: {args.target_size}")
    if args.mode == "2d":
        print(f"Canvas size: {args.canvas_size}×{args.canvas_size}")
    elif args.mode == "3d":
        print(f"Canvas size: {args.canvas_depth}×{args.canvas_size}×{args.canvas_size} (D×H×W)")
        print(f"Canvas volume: {dm.canvas_volume}")
    else:  # 1D
        print(f"Canvas length: {args.canvas_size * args.canvas_size}")
        print(f"Segment length: {args.target_size * args.target_size}")
    print(f"Input channels: {dm.input_channels}")
    print(f"Output channels: {dm.output_channels}")
    print(f"Batch size: {dm.batch_size}")
    print(f"Train samples: {len(dm.train_dataset)}")
    print(f"Val samples: {len(dm.val_dataset)}")

    loader = dm.train_dataloader()
    x, y = next(iter(loader))
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")

    # Visualize batch
    B = min(args.batch_size, 8)

    if args.mode == "3d":
        # 3D mode: Perspective visualization with depth on X-axis
        # Zero values are transparent, content is visible via scatter plot
        # x shape is [B, C, D, H, W], y shape is [B, C, H, W]
        import numpy as np

        D = args.canvas_depth
        H = args.canvas_size
        W = args.canvas_size
        t = args.target_size

        # Show fewer samples for 3D
        B = min(B, 4)

        # Determine number of columns based on mask
        num_cols = 3 if args.with_mask else 2  # 3D view, (mask view), label

        # Check if we have RGB data (colored frames)
        is_rgb = x.shape[1] == 3

        fig = plt.figure(figsize=(8 * num_cols, 7 * B))

        for i in range(B):
            # For RGB, use luminance for finding items; for grayscale, use channel 0
            if is_rgb:
                vol_rgb = x[i].cpu().numpy()  # [3, D, H, W]
                # Compute luminance for item detection
                vol = 0.299 * vol_rgb[0] + 0.587 * vol_rgb[1] + 0.114 * vol_rgb[2]
            else:
                vol = x[i, 0].cpu().numpy()  # [D, H, W]
                vol_rgb = None

            # Label handling
            if y.shape[1] == 3:  # RGB label (colored_label)
                label_rgb = y[i].cpu().numpy()  # [3, H, W]
                label = None
            else:
                label = y[i, 0].cpu().numpy()  # [H, W]
                label_rgb = None

            # === 3D perspective view using scatter for non-zero voxels ===
            ax3d = fig.add_subplot(B, num_cols, i * num_cols + 1, projection="3d")

            # Scale depth for better visualization
            depth_scale = max(1.0, H / D / 1.5)

            # Find non-zero voxels and draw coordinate indicator lines
            threshold = 0.05

            # Find individual items by detecting connected components per depth slice
            from scipy import ndimage

            items = []  # List of (depth, h_center, w_center, h_min, w_min)
            for d in range(D):
                slice_2d = np.abs(vol[d]) > threshold
                if slice_2d.any():
                    # Label connected components
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

                if is_rgb and vol_rgb is not None:
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

                # Scatter plot - each voxel as a point
                ax3d.scatter(d_coords, w_coords, h_coords, c=colors, s=8, marker="s", depthshade=False)

            # Draw coordinate indicator lines for each item
            item_colors = plt.cm.tab10(np.linspace(0, 1, max(len(items), 1)))
            d_max_vis = (D - 0.5) * depth_scale

            for idx, (d, h_center, w_center, h_min, w_min) in enumerate(items):
                d_pos = d * depth_scale
                z_top = H - h_min  # Top of item (inverted)
                y_left = w_min  # Left of item
                color = item_colors[idx % len(item_colors)]

                # Draw marker at the item's top-left corner
                ax3d.scatter(
                    [d_pos],
                    [y_left],
                    [z_top],
                    c=[color],
                    s=40,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=10,
                )

                # Line down to floor (z=0) - shows depth and y position
                ax3d.plot(
                    [d_pos, d_pos], [y_left, y_left], [z_top, 0], color=color, linewidth=1.5, linestyle=":", alpha=0.7
                )

                # Line to back wall (d=max) - shows x (depth) position
                ax3d.plot(
                    [d_pos, d_max_vis], [y_left, y_left], [0, 0], color=color, linewidth=1.0, linestyle="--", alpha=0.5
                )

                # Line to side wall (y=0) - shows y (width) position
                ax3d.plot([d_pos, d_pos], [y_left, 0], [0, 0], color=color, linewidth=1.0, linestyle="--", alpha=0.5)

                # Small marker on floor showing projection
                ax3d.scatter([d_pos], [y_left], [0], c=[color], s=20, marker="x", alpha=0.7)

                # Add coordinate label
                ax3d.text(d_pos + 1, y_left + 2, -2, f"d={d}", fontsize=7, color=color, fontweight="bold")

            # Draw canvas wireframe (light)
            def draw_box_edges(ax, x0, x1, y0, y1, z0, z1, color="gray", linewidth=0.5, linestyle="-"):
                """Draw edges of a box."""
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
            draw_box_edges(ax3d, 0, d_max, 0, W, 0, H, color="lightgray", linewidth=0.5, linestyle="-")

            # Draw readout region as red dashed rectangle on back slice
            d_back = (D - 1) * depth_scale
            readout_y = [W - t, W, W, W - t, W - t]
            readout_z = [0, 0, t, t, 0]
            ax3d.plot([d_back] * 5, readout_y, readout_z, color="red", linewidth=2, linestyle="--", alpha=0.8)
            # Add label for readout
            ax3d.text(d_back + 1, W - t / 2, t / 2, "readout", fontsize=7, color="red")

            # Set labels: Depth on X, Width on Y, Height on Z
            ax3d.set_xlabel("Depth")
            ax3d.set_ylabel("Width")
            ax3d.set_zlabel("Height")
            ax3d.set_xlim(-1, D * depth_scale)
            ax3d.set_ylim(0, W)
            ax3d.set_zlim(0, H)

            # Fix depth axis ticks to show actual depth values (0 to D-1)
            depth_ticks = np.arange(D) * depth_scale
            ax3d.set_xticks(depth_ticks)
            ax3d.set_xticklabels([str(d) for d in range(D)])

            # Set viewing angle
            ax3d.view_init(elev=20, azim=-50)
            ax3d.set_box_aspect([D * depth_scale / W, 1, H / W])

            if i == 0:
                ax3d.set_title(f"3D Canvas ({D}×{H}×{W})", fontsize=10)

            # === Mask view (if applicable) ===
            col_idx = 2
            if args.with_mask and x.shape[1] >= 2:
                ax_mask = fig.add_subplot(B, num_cols, i * num_cols + col_idx, projection="3d")
                mask_vol = x[i, 1].cpu().numpy()  # [D, H, W]

                # Find mask items using connected components
                mask_items = []
                for d in range(D):
                    mask_2d = mask_vol[d] > 0.5
                    if mask_2d.any():
                        labeled, num_features = ndimage.label(mask_2d)
                        for label_id in range(1, num_features + 1):
                            component = labeled == label_id
                            nz = np.where(component)
                            if len(nz[0]) > 10:
                                h_min = nz[0].min()
                                w_min = nz[1].min()
                                mask_items.append((d, h_min, w_min))

                # Draw mask voxels
                nz_mask = np.where(mask_vol > 0.5)
                if len(nz_mask[0]) > 0:
                    d_m = nz_mask[0] * depth_scale
                    h_m = H - nz_mask[1]
                    w_m = nz_mask[2]
                    ax_mask.scatter(d_m, w_m, h_m, c="orange", s=12, marker="s", alpha=0.8, depthshade=False)

                # Draw coordinate indicators for mask items
                for idx, (d, h_min, w_min) in enumerate(mask_items):
                    d_pos = d * depth_scale
                    z_top = H - h_min
                    y_left = w_min
                    color = "darkorange"

                    # Marker at corner
                    ax_mask.scatter(
                        [d_pos],
                        [y_left],
                        [z_top],
                        c=[color],
                        s=40,
                        marker="o",
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=10,
                    )

                    # Line down to floor
                    ax_mask.plot(
                        [d_pos, d_pos],
                        [y_left, y_left],
                        [z_top, 0],
                        color=color,
                        linewidth=1.5,
                        linestyle=":",
                        alpha=0.7,
                    )

                    # Lines on floor
                    ax_mask.plot(
                        [d_pos, d_max_vis],
                        [y_left, y_left],
                        [0, 0],
                        color=color,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.5,
                    )
                    ax_mask.plot(
                        [d_pos, d_pos], [y_left, 0], [0, 0], color=color, linewidth=1.0, linestyle="--", alpha=0.5
                    )

                    # Floor marker
                    ax_mask.scatter([d_pos], [y_left], [0], c=[color], s=20, marker="x", alpha=0.7)

                    # Label
                    ax_mask.text(d_pos + 1, y_left + 2, -2, f"d={d}", fontsize=7, color=color, fontweight="bold")

                # Draw canvas wireframe
                draw_box_edges(ax_mask, 0, d_max_vis, 0, W, 0, H, color="lightgray", linewidth=0.5, linestyle="-")

                ax_mask.set_xlabel("Depth")
                ax_mask.set_ylabel("Width")
                ax_mask.set_zlabel("Height")
                ax_mask.set_xlim(-1, D * depth_scale)
                ax_mask.set_ylim(0, W)
                ax_mask.set_zlim(0, H)

                # Fix depth axis ticks to show actual depth values
                ax_mask.set_xticks(depth_ticks)
                ax_mask.set_xticklabels([str(d) for d in range(D)])

                ax_mask.view_init(elev=20, azim=-50)
                ax_mask.set_box_aspect([D * depth_scale / W, 1, H / W])
                if i == 0:
                    ax_mask.set_title(f"Mask ({D}×{H}×{W})", fontsize=10)
                col_idx += 1

            # === Target label ===
            ax_label = fig.add_subplot(B, num_cols, i * num_cols + col_idx)
            if label_rgb is not None:
                # RGB label - transpose from [C, H, W] to [H, W, C]
                label_display = np.transpose(label_rgb, (1, 2, 0))
                label_display = np.clip(label_display, 0, 1)
                ax_label.imshow(label_display)
                title_suffix = " (RGB)"
            else:
                ax_label.imshow(label, cmap="gray", vmin=0, vmax=1)
                title_suffix = ""
            if i == 0:
                ax_label.set_title(f"Target ({t}×{t}){title_suffix}", fontsize=10)
            ax_label.axis("off")

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)

    elif args.mode == "1d":
        # 1D mode: show canvas as 1D line plot and label as 2D image
        # Note: x shape is [B, C, L], y shape is [B, C, segment_length] (raw from dataloader)
        if args.with_mask:
            num_cols = 3
            fig, axes = plt.subplots(B, num_cols, figsize=(12, 2 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            for i in range(B):
                # Canvas intensity as line plot [C, L] -> [L] for channel 0
                canvas_data = x[i, 0, :].cpu().numpy()
                axes[i, 0].plot(canvas_data, linewidth=0.5)
                # Adjust ylim based on data range (to show readout_value like -1)
                y_min = min(-0.1, canvas_data.min() - 0.1)
                y_max = max(1.1, canvas_data.max() + 0.1)
                axes[i, 0].set_ylim(y_min, y_max)
                # Add horizontal line at readout_value if it's different from 0
                if args.readout_value != 0:
                    axes[i, 0].axhline(
                        y=args.readout_value,
                        color="red",
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.7,
                        label=f"readout={args.readout_value}",
                    )
                    if i == 0:
                        axes[i, 0].legend(loc="upper right", fontsize=6)
                if i == 0:
                    axes[i, 0].set_title("Canvas (1D)")
                axes[i, 0].set_xlabel("Position")

                # Mask as line plot [C, L] -> [L] for channel 1
                axes[i, 1].plot(x[i, 1, :].cpu().numpy(), linewidth=0.5, color="orange")
                axes[i, 1].set_ylim(-0.1, 1.1)
                if i == 0:
                    axes[i, 1].set_title("Mask (1D)")
                axes[i, 1].set_xlabel("Position")

                # Label reshaped back to 2D [C, seg_len] -> [seg_len] -> [H, W]
                label_2d = y[i, 0, :].cpu().reshape(args.target_size, args.target_size)
                axes[i, 2].imshow(label_2d, cmap="gray")
                if i == 0:
                    axes[i, 2].set_title(f"Label ({args.target_size}×{args.target_size})")
                axes[i, 2].axis("off")
        else:
            num_cols = 2
            fig, axes = plt.subplots(B, num_cols, figsize=(10, 2 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            for i in range(B):
                # Canvas as line plot [C, L] -> [L] for channel 0
                canvas_data = x[i, 0, :].cpu().numpy()
                axes[i, 0].plot(canvas_data, linewidth=0.5)
                # Adjust ylim based on data range (to show readout_value like -1)
                y_min = min(-0.1, canvas_data.min() - 0.1)
                y_max = max(1.1, canvas_data.max() + 0.1)
                axes[i, 0].set_ylim(y_min, y_max)
                # Add horizontal line at readout_value if it's different from 0
                if args.readout_value != 0:
                    axes[i, 0].axhline(
                        y=args.readout_value,
                        color="red",
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.7,
                        label=f"readout={args.readout_value}",
                    )
                    if i == 0:
                        axes[i, 0].legend(loc="upper right", fontsize=6)
                if i == 0:
                    axes[i, 0].set_title("Canvas (1D)")
                axes[i, 0].set_xlabel("Position")

                # Label reshaped back to 2D [C, seg_len] -> [seg_len] -> [H, W]
                label_2d = y[i, 0, :].cpu().reshape(args.target_size, args.target_size)
                axes[i, 1].imshow(label_2d, cmap="gray")
                if i == 0:
                    axes[i, 1].set_title(f"Label ({args.target_size}×{args.target_size})")
                axes[i, 1].axis("off")
    elif args.colored_frames:
        # RGB mode: show canvas and label
        num_cols = 2
        fig, axes = plt.subplots(B, num_cols, figsize=(4 * num_cols, 2.5 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        for i in range(B):
            # Canvas RGB (permute from [C, H, W] to [H, W, C])
            # Normalize to [0, 1] range accounting for possible negative readout_value
            canvas_rgb = x[i].permute(1, 2, 0).cpu()
            if args.readout_value < 0:
                # Shift and scale: map [readout_value, 1] to [0, 1]
                canvas_rgb = (canvas_rgb - args.readout_value) / (1.0 - args.readout_value)
            canvas_rgb = canvas_rgb.clip(0, 1)
            axes[i, 0].imshow(canvas_rgb)
            if i == 0:
                title = "Canvas (RGB)"
                if args.readout_value != 0:
                    title += f" [readout={args.readout_value}]"
                axes[i, 0].set_title(title)
            axes[i, 0].axis("off")
            # Label: RGB if colored_label, grayscale otherwise
            if args.colored_label:
                # y is [3, H, W] -> [H, W, 3]
                label_rgb = y[i].permute(1, 2, 0).cpu().clip(0, 1)
                axes[i, 1].imshow(label_rgb)
                if i == 0:
                    axes[i, 1].set_title("Label (RGB)")
            else:
                axes[i, 1].imshow(y[i, 0].cpu(), cmap="gray")
                if i == 0:
                    axes[i, 1].set_title("Label")
            axes[i, 1].axis("off")
    elif args.with_mask:
        # Mask mode: show intensity, mask, label
        num_cols = 3
        fig, axes = plt.subplots(B, num_cols, figsize=(4 * num_cols, 2.5 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        # Set vmin/vmax to show readout_value properly
        vmin = min(0, args.readout_value)
        vmax = 1
        for i in range(B):
            axes[i, 0].imshow(x[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            if i == 0:
                title = "Canvas"
                if args.readout_value != 0:
                    title += f" [readout={args.readout_value}]"
                axes[i, 0].set_title(title)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(x[i, 1].cpu(), cmap="gray")
            if i == 0:
                axes[i, 1].set_title("Mask")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(y[i, 0].cpu(), cmap="gray")
            if i == 0:
                axes[i, 2].set_title("Label")
            axes[i, 2].axis("off")
    else:
        # Simple mode: canvas and label
        num_cols = 2
        fig, axes = plt.subplots(B, num_cols, figsize=(4 * num_cols, 2.5 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        # Set vmin/vmax to show readout_value properly
        vmin = min(0, args.readout_value)
        vmax = 1
        for i in range(B):
            axes[i, 0].imshow(x[i, 0].cpu(), cmap="gray", vmin=vmin, vmax=vmax)
            if i == 0:
                title = "Canvas"
                if args.readout_value != 0:
                    title += f" [readout={args.readout_value}]"
                axes[i, 0].set_title(title)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(y[i, 0].cpu(), cmap="gray")
            if i == 0:
                axes[i, 1].set_title("Label")
            axes[i, 1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.1, hspace=0.2)

    # Build filename
    mode_str = f"{args.mode}_{args.placement}"
    if args.with_mask:
        mode_str += "_mask"
    if args.colored_frames:
        mode_str += "_colored"
    if args.colored_label:
        mode_str += "_coloredlabel"
    if args.num_items > 1:
        mode_str += f"_{args.num_items}items"

    out_path = os.path.join(args.output_dir, f"spatial_recall_{args.dataset}_{mode_str}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
