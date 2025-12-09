"""Spatial Recall Dataset and DataModule for PyTorch Lightning.

This module wraps base datamodules (e.g., MNISTDataModule, EMNISTDataModule) to create
spatial recall tasks where images are placed on a larger canvas and the model must
recall the target at a designated readout location (bottom-right corner).

Supports:
    - Fixed placement: Target always at top-left corner
    - Random placement: Target at random valid positions (non-overlapping with readout)
    - Optional mask channel to indicate target location
    - Colored frames mode: RGB canvas with colored bounding boxes around items
    - Multiple items (distractors) on the canvas

Usage:
    PYTHONPATH=. python experiments/datamodules/spatial_recall_dataset.py
"""

from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nvsubquadratic.lazy_config import LazyConfig, instantiate


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
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_size: int,
        canvas_size: int,
        generator: torch.Generator,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
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
    ) -> None:
        """Initialize the SpatialRecallDataModule."""
        super().__init__()

        # Validate arguments
        assert data_type in ("sequence", "image"), f"data_type must be 'sequence' or 'image', got {data_type}"
        assert placement in ("fixed", "random"), f"placement must be 'fixed' or 'random', got {placement}"
        assert not (with_mask and use_colored_frames), "with_mask and use_colored_frames cannot both be True"
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

        self.output_channels = 1  # Always grayscale target

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
                )
                self.val_dataset = SpatialRecallDataset(
                    base_val_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._val_generator,
                    self.placement,
                    self.with_mask,
                )
            else:
                # For multi-item mode, wrap with simple dataset (no mask) and apply in collate
                self.train_dataset = SpatialRecallDataset(
                    base_train_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._train_generator,
                    self.placement,
                    with_mask=self.with_mask and not self.use_colored_frames,
                )
                self.val_dataset = SpatialRecallDataset(
                    base_val_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._val_generator,
                    self.placement,
                    with_mask=self.with_mask and not self.use_colored_frames,
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
                )
            else:
                self.test_dataset = SpatialRecallDataset(
                    base_test_datamodule,
                    self.target_size,
                    self.canvas_size,
                    self._test_generator,
                    self.placement,
                    with_mask=self.with_mask and not self.use_colored_frames,
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
                draw_outline_rgb(canvas_rgb, y0_readout, x0_readout, t, color)
                x_rgb_list.append(canvas_rgb)
                y_sel_list.append(ys[i])
            else:
                # Fallback
                x_rgb_list.append(xs[i].repeat(3, 1, 1) if xs[i].shape[0] == 1 else xs[i][:3])
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


if __name__ == "__main__":
    import argparse
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.datamodules.emnist import EMNISTDataModule
    from experiments.datamodules.mnist import MNISTDataModule
    from nvsubquadratic.lazy_config import LazyConfig

    parser = argparse.ArgumentParser(description="Visualize Spatial Recall samples")
    parser.add_argument("--placement", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--with-mask", action="store_true", help="Add mask channel")
    parser.add_argument("--colored-frames", action="store_true", help="Use colored frames (RGB)")
    parser.add_argument("--num-items", type=int, default=1, help="Number of items on canvas")
    parser.add_argument("--target-size", type=int, default=16, help="Target image size")
    parser.add_argument("--canvas-size", type=int, default=64, help="Canvas size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--data-dir", type=str, default="./.data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="_tmp", help="Output directory")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "emnist"])
    parser.add_argument(
        "--emnist-split", type=str, default="digits", choices=["digits", "letters", "balanced", "bymerge", "byclass"]
    )
    args = parser.parse_args()

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
    dm = SpatialRecallDataModule(
        base_datamodule_cfg=base_datamodule_cfg,
        target_size=args.target_size,
        canvas_size=args.canvas_size,
        data_type="image",
        placement=args.placement,
        with_mask=args.with_mask,
        use_colored_frames=args.colored_frames,
        num_items=args.num_items,
    )
    dm.prepare_data()
    dm.setup("fit")

    print(f"Dataset: {args.dataset}")
    if args.dataset == "emnist":
        print(f"EMNIST split: {args.emnist_split}")
    print(f"Placement: {args.placement}")
    print(f"With mask: {args.with_mask}")
    print(f"Colored frames: {args.colored_frames}")
    print(f"Num items: {args.num_items}")
    print(f"Target size: {args.target_size}")
    print(f"Canvas size: {args.canvas_size}")
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

    if args.colored_frames:
        # RGB mode: show canvas and label
        num_cols = 2
        fig, axes = plt.subplots(B, num_cols, figsize=(4 * num_cols, 2.5 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        for i in range(B):
            # Canvas RGB (permute from [C, H, W] to [H, W, C])
            axes[i, 0].imshow(x[i].permute(1, 2, 0).cpu().clip(0, 1))
            if i == 0:
                axes[i, 0].set_title("Canvas (RGB)")
            axes[i, 0].axis("off")
            # Label
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
        for i in range(B):
            axes[i, 0].imshow(x[i, 0].cpu(), cmap="gray")
            if i == 0:
                axes[i, 0].set_title("Canvas")
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
        for i in range(B):
            axes[i, 0].imshow(x[i, 0].cpu(), cmap="gray")
            if i == 0:
                axes[i, 0].set_title("Canvas")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(y[i, 0].cpu(), cmap="gray")
            if i == 0:
                axes[i, 1].set_title("Label")
            axes[i, 1].axis("off")

    fig.tight_layout()

    # Build filename
    mode_str = args.placement
    if args.with_mask:
        mode_str += "_mask"
    if args.colored_frames:
        mode_str += "_colored"
    if args.num_items > 1:
        mode_str += f"_{args.num_items}items"

    out_path = os.path.join(args.output_dir, f"spatial_recall_{args.dataset}_{mode_str}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
