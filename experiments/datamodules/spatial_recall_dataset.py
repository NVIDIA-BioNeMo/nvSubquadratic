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
    - 3D motion recall: Moving-image video blocks on cubic canvases, with depth as time

Usage:
    # 2D mode
    PYTHONPATH=. python experiments/datamodules/spatial_recall_dataset.py

    # 1D mode
    PYTHONPATH=. python experiments/datamodules/spatial_recall_dataset.py --mode 1d
"""

import math
from typing import Literal, Optional, Tuple

import numpy as np
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
        use_colored_frames: Use RGB canvas with colored boundary markers around items.
            In 1D, "frames" are short colored markers (``frame_width`` elements) placed
            immediately before and after each segment.
        num_items: Number of items to place (1 = target only, >1 = target + distractors).
        readout_value: Value to fill the readout region with (default 0.0). Use e.g. -1.0 to
            explicitly mark the readout region so the model knows where to output.
        colored_label: If True and use_colored_frames=True, the label will be RGB with the
            digit colored using the same color as its frame.
        frame_width: Number of elements for each colored boundary marker (default 2).
    """

    # Fixed RGB palette for colored frames (same as 2D)
    PALETTE = SpatialRecallDataModule.PALETTE

    def __init__(
        self,
        base_datamodule_cfg: LazyConfig,
        target_size: int,
        canvas_size: int,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        use_colored_frames: bool = False,
        num_items: int = 1,
        readout_value: float = 0.0,
        colored_label: bool = False,
        frame_width: int = 2,
    ) -> None:
        """Initialize the SpatialRecall1DDataModule."""
        super().__init__()

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
        self.canvas_length = canvas_size * canvas_size  # Computed from canvas_size
        self.segment_length = target_size * target_size
        self.placement = placement
        self.with_mask = with_mask
        self.use_colored_frames = use_colored_frames
        self.num_items = num_items
        self.readout_value = readout_value
        self.colored_label = colored_label
        self.frame_width = frame_width

        # Total footprint of one item including frames
        self.framed_segment_length = (
            self.segment_length + 2 * frame_width if use_colored_frames else self.segment_length
        )

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

    def _colored_frames_collate(self, batch: list) -> Tuple[Tensor, Tensor]:
        """Collate function for colored frames mode in 1D.

        Creates an RGB 1D canvas where each item is flanked by colored boundary
        markers (``frame_width`` elements on each side).  The readout region at the
        end of the canvas gets the same colour as the target item.

        Layout of one framed segment (``framed_segment_length`` total)::

            [frame_width color] [segment_length grayscale-as-RGB] [frame_width color]
        """
        xs, ys = zip(*batch)
        xs = list(xs)
        ys = list(ys)

        batch_size = len(xs)
        seg_len = self.segment_length
        framed_len = self.framed_segment_length
        fw = self.frame_width
        L = self.canvas_length

        # Readout occupies the last seg_len elements (same as simple-copy readout).
        # Valid placement zone: 0 … L - seg_len - framed_len  (leaves room for
        # readout *and* the framed segment not overlapping with readout)
        max_start = L - seg_len - framed_len
        valid_positions = torch.arange(0, max(max_start + 1, 1), dtype=torch.long)

        palette = self.PALETTE.to(dtype=xs[0].dtype, device=xs[0].device)
        g = self._generator

        x_rgb_list = []
        y_sel_list = []

        for i in range(batch_size):
            canvas_rgb = torch.zeros((3, L), dtype=xs[0].dtype, device=xs[0].device)

            # Collect item indices (target first, then distractors)
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

            num_positions = valid_positions.shape[0]
            perm_pos = torch.randperm(num_positions, generator=g)
            pos_cursor = 0
            occupied = []  # list of (start, end) inclusive of frames
            placed_meta = []  # (pos, cidx, item_idx)

            def overlaps_any(pos: int) -> bool:
                p_end = pos + framed_len
                for o_start, o_end in occupied:
                    if not (p_end <= o_start or o_end <= pos):
                        return True
                return False

            for k_idx, j in enumerate(indices_to_place):
                placed = False
                attempts = 0
                while attempts < num_positions and pos_cursor < num_positions:
                    pos = int(valid_positions[perm_pos[pos_cursor]].item())
                    pos_cursor += 1
                    attempts += 1
                    if overlaps_any(pos):
                        continue

                    cidx = int(color_order[k_idx].item())
                    color = palette[cidx]  # [3]

                    # Left frame marker
                    canvas_rgb[:, pos : pos + fw] = color.view(3, 1)
                    # Grayscale digit broadcast to RGB
                    seg_start = pos + fw
                    canvas_rgb[:, seg_start : seg_start + seg_len] = ys[j][0].unsqueeze(0).expand(3, -1)
                    # Right frame marker
                    canvas_rgb[:, seg_start + seg_len : seg_start + seg_len + fw] = color.view(3, 1)

                    occupied.append((pos, pos + framed_len))
                    placed_meta.append((pos, cidx, j))
                    placed = True
                    break

                if not placed:
                    break

            # Readout region: last seg_len elements, coloured with target's colour
            target_meta = None
            for pos, cidx, pj in placed_meta:
                if pj == i:
                    target_meta = (pos, cidx, pj)
                    break

            if target_meta is not None:
                _, sel_cidx, _ = target_meta
                color = palette[sel_cidx]
                readout_start = L - seg_len

                if self.readout_value != 0.0:
                    canvas_rgb[:, readout_start + fw : L - fw] = self.readout_value

                # Colour markers at readout boundaries
                canvas_rgb[:, readout_start : readout_start + fw] = color.view(3, 1)
                canvas_rgb[:, L - fw : L] = color.view(3, 1)

                x_rgb_list.append(canvas_rgb)

                if self.colored_label:
                    # [1, seg_len] * [3, 1] -> [3, seg_len]
                    label_rgb = ys[i] * color.view(3, 1)
                    y_sel_list.append(label_rgb)
                else:
                    y_sel_list.append(ys[i])
            else:
                x_rgb_list.append(canvas_rgb)
                if self.colored_label:
                    y_sel_list.append(ys[i].repeat(3, 1))
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


def _random_sweep_offsets(
    length: int,
    limit: int,
    max_step: float,
    generator: torch.Generator,
) -> Tensor:
    """Generate a monotone eased sweep over ``[0, limit]``, rounded to voxels.

    A random direction and speed profile determine the continuous path. If
    its largest step exceeds ``max_step``, its span shrinks about the centre.
    Rounding to integer voxels can increase a step to ``ceil(max_step)``.
    The path need not visit both endpoints or produce nonempty image slices.

    Args:
        length: Number of time steps (= depth slices of the block).
        limit: Largest allowed offset; the sweep spans ``[0, limit]``.
        max_step: Cap on the continuous slice-to-slice step, before voxel rounding.  Spans too
            wide to sweep within the cap are shrunk around their centre.
        generator: Random generator for reproducibility.

    Returns:
        Long tensor of ``length`` integer offsets in ``[0, limit]``.
    """
    if limit == 0:
        return torch.zeros(length, dtype=torch.long)
    gamma = 0.7 + float(torch.rand(1, generator=generator)) * 0.8  # random speed profile
    s = torch.linspace(0.0, 1.0, length) ** gamma
    ease = s * s * (3.0 - 2.0 * s)  # smoothstep: gentle start/stop, monotone
    x = limit * ease
    if float(torch.rand(1, generator=generator)) < 0.5:
        x = limit - x  # random sweep direction
    if length > 1:
        step = float((x[1:] - x[:-1]).abs().max())
        if step > max_step:
            # Shrink wide sweeps about their centre before voxel rounding.
            centre = limit / 2.0
            x = centre + (x - centre) * (max_step / step)
    offsets = torch.clamp(torch.round(x), 0, limit).long()
    # Half-to-even ties can expand an integer-sized step by one voxel.
    # Clamp rounded increments as well, preserving monotonicity and bounds.
    cap = min(math.ceil(max_step), limit)
    steps = offsets.diff().clamp(min=-cap, max=cap)
    return torch.cat((offsets[:1], offsets[:1] + steps.cumsum(dim=0)))


def _make_motion_block(
    img: Tensor,
    digit_size: int,
    block_size: int,
    generator: torch.Generator,
    max_step: float,
    spin: bool,
) -> Tensor:
    """Build a ``[C, b, b, b]`` motion block from a ``[C, H, W]`` image.

    The image is stamped (resized to ``digit_size²``) on every depth slice
    ``t`` at offset ``(h(t), w(t))`` given by two independent monotone sweep
    paths (see :func:`_random_sweep_offsets`), and optionally spun in-plane in
    exact 90° turns via ``torch.rot90`` — a random initial orientation
    (0/90/180/270°), then 1-3 evenly spaced quarter turns across the tube in a
    random direction.  Being pure pixel permutations, quarter turns stay
    perfectly crisp at any stamp size.

    The motion range is *ink-aware*: the constraint is that the digit's ink
    (pixels above a small threshold over the background) stays inside the
    block at every step — the stamp's bounding box may clip at the block edges
    (low-intensity pixels below the ink threshold may be clipped).  Narrow digits (e.g. a "1") therefore get more
    room to move than digits that fill their stamp (e.g. a "0"), and the
    in-plane spin is unconstrained: the safe range accounts for every rotated
    frame, so the ink never leaves the block at any angle.

    The block is filled with the image's own background value (its per-channel
    minimum), so the moving digit blends seamlessly — there is no visible
    stamp-box boundary between image background and block background.

    Args:
        img: ``[C, H, W]`` source image (e.g. a normalised EMNIST digit).
        digit_size: Size the image is resized to before stamping.
        block_size: Edge length of the cubic block (= number of time steps).
        generator: Random generator for reproducibility.
        max_step: Continuous per-axis step cap; integer steps can reach ``ceil(max_step)``.
        spin: If True, turn the image in-plane in 90° steps.

    Returns:
        ``[C, block_size, block_size, block_size]`` motion tube.
    """
    b = block_size
    t_sz = digit_size

    q0 = n_turns = direction = 0
    if spin:
        q0 = int(torch.randint(0, 4, (1,), generator=generator))
        n_turns = 1 + int(torch.randint(0, min(3, max(b - 1, 1)), (1,), generator=generator))
        direction = 1 if float(torch.rand(1, generator=generator)) < 0.5 else -1

    # Pre-render the (rotated) stamp for every time step.
    stamps = []
    for t in range(b):
        if spin:
            q = (q0 + direction * ((t * n_turns) // max(b - 1, 1))) % 4
            frame = torch.rot90(img, k=q, dims=(1, 2))
        else:
            frame = img
        stamps.append(
            torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=(t_sz, t_sz), mode="bilinear", align_corners=False
            ).squeeze(0)
        )

    # Ink-aware safe offset range per axis: intersect, over all steps, the
    # offsets that keep that step's ink bounding box inside the block.
    bg = float(img.min())
    ink_thresh = bg + 0.15 * (float(img.max()) - bg)
    lo_h, hi_h = -(t_sz - 1), b - 1
    lo_w, hi_w = -(t_sz - 1), b - 1
    for stamp in stamps:
        mask = stamp.amax(dim=0) > ink_thresh
        if not bool(mask.any()):
            continue
        rows = mask.any(dim=1).nonzero()
        cols = mask.any(dim=0).nonzero()
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        lo_h, hi_h = max(lo_h, -r0), min(hi_h, b - 1 - r1)
        lo_w, hi_w = max(lo_w, -c0), min(hi_w, b - 1 - c1)
    if hi_h < lo_h or hi_w < lo_w:  # Fall back to the stamp-box range.
        lo_h = lo_w = 0
        hi_h = hi_w = b - t_sz

    hs = lo_h + _random_sweep_offsets(b, hi_h - lo_h, max_step, generator)
    ws = lo_w + _random_sweep_offsets(b, hi_w - lo_w, max_step, generator)

    # Background-matched block: filled with the image's own background so the
    # stamp boundary is invisible (per-channel minimum of the source image).
    bg_per_channel = img.amin(dim=(1, 2)).view(-1, 1, 1, 1)
    block = bg_per_channel.expand(img.shape[0], b, b, b).clone().to(dtype=img.dtype, device=img.device)
    for t, stamp in enumerate(stamps):
        h0, w0 = int(hs[t]), int(ws[t])
        # Clipped paste: the stamp box may stick out of the block, the ink never does.
        hd0, hd1 = max(0, h0), min(b, h0 + t_sz)
        wd0, wd1 = max(0, w0), min(b, w0 + t_sz)
        block[:, t, hd0:hd1, wd0:wd1] = stamp[:, hd0 - h0 : hd1 - h0, wd0 - w0 : wd1 - w0]
    return block


def _validate_motion_geometry(
    digit_size: int, block_size: int, canvas_size: int, placement: str, max_step: float
) -> None:
    """Reject invalid geometry and motion bounds before loading source data."""
    if not 0 < digit_size <= block_size:
        raise ValueError("Require 0 < digit_size <= block_size.")
    if canvas_size < 2 * block_size:
        raise ValueError("canvas_size must be >= 2 * block_size to separate source and readout.")
    if placement not in ("fixed", "random"):
        raise ValueError("placement must be 'fixed' or 'random'.")
    if not 0 <= max_step < float("inf"):
        raise ValueError("max_step must be finite and nonnegative.")


class SpatialRecall3DMotionDataset(Dataset):
    """3D *motion* spatial recall dataset (moving-digit copy task).

    A 2D image is stamped on every depth slice of a ``block_size³`` cube
    (depth = time) while translating along a random continuous path and
    optionally turning in-plane — a spatio-temporal tube (see
    :func:`_make_motion_block`).  The block is placed on a cubic
    ``canvas_size³`` volume and must be recalled at the back-bottom-right
    ``block_size³`` readout corner.

    Args:
        base_dataset: Base dataset providing (image, label) pairs.
        digit_size: Positive size the 2D image is resized to before stamping.
            Must be ``<= block_size``. Motion bounds use the resized ink mask,
            which can allow the stamp background to extend beyond the block.
        block_size: Edge length of the cubic motion block (= number of time
            steps).
        canvas_size: Edge length of the cubic canvas.
        generator: Random generator for reproducibility.
        placement: "fixed" (front-top-left corner) or "random".
        readout_value: Value to fill the readout region with (default 0.0).
        max_step: Continuous per-axis step cap; integer steps can reach ``ceil(max_step)``.
        spin: If True (default), turn the digit in-plane in 90° steps — see
            :func:`_make_motion_block`.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        digit_size: int,
        block_size: int,
        canvas_size: int,
        generator: torch.Generator,
        placement: Literal["fixed", "random"] = "fixed",
        readout_value: float = 0.0,
        max_step: float = 2.0,
        spin: bool = True,
    ) -> None:
        """Initialize the SpatialRecall3DMotionDataset."""
        super().__init__()

        _validate_motion_geometry(digit_size, block_size, canvas_size, placement, max_step)

        self.base_dataset = base_dataset
        self.digit_size = digit_size
        self.block_size = block_size
        self.canvas_size = canvas_size
        self.generator = generator
        self.placement = placement
        self.readout_value = readout_value
        self.max_step = max_step
        self.spin = spin

        if placement == "random":
            self._precompute_valid_positions()

    def _precompute_valid_positions(self) -> None:
        """Precompute cubic-block start positions that don't overlap the readout.

        A position is invalid iff all three coordinates are ``> S - 2b``,
        where S is the canvas size and b is the block size.
        """
        S = self.canvas_size
        b = self.block_size
        starts = torch.arange(0, S - b + 1, dtype=torch.long)
        grid_d, grid_y, grid_x = torch.meshgrid(starts, starts, starts, indexing="ij")
        invalid_start = S - 2 * b
        invalid = (grid_d > invalid_start) & (grid_y > invalid_start) & (grid_x > invalid_start)
        valid = ~invalid
        self.valid_positions = torch.stack([grid_d[valid], grid_y[valid], grid_x[valid]], dim=1)

    def __len__(self) -> int:
        """Return the number of source samples.

        Returns:
            int: Length of the wrapped base dataset.
        """
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Build one sample.

        Args:
            idx: Index into the base dataset (selects the digit).

        Returns:
            ``(canvas, label)``: the ``[C, S, S, S]`` cubic canvas containing
            the motion tube, and the ``[C, b, b, b]`` tube itself (the recall
            target at the readout corner).
        """
        img, _ = self.base_dataset[idx]
        # img: [C, H, W] from base dataset

        b = self.block_size
        S = self.canvas_size

        block = _make_motion_block(
            img,
            self.digit_size,
            b,
            self.generator,
            self.max_step,
            self.spin,
        )
        num_channels = block.shape[0]

        # Cubic canvas.
        canvas = torch.zeros((num_channels, S, S, S), dtype=block.dtype, device=block.device)

        # Determine placement position (d, y, x).
        if self.placement == "fixed":
            d0, y0, x0 = 0, 0, 0  # front-top-left corner
        else:  # random
            num_pos = self.valid_positions.shape[0]
            idx_pos = int(torch.randint(low=0, high=num_pos, size=(1,), generator=self.generator).item())
            d0, y0, x0 = self.valid_positions[idx_pos].tolist()

        canvas[:, d0 : d0 + b, y0 : y0 + b, x0 : x0 + b] = block

        # Fill the readout region (back-bottom-right cube) with readout_value.
        if self.readout_value != 0.0:
            canvas[:, S - b :, S - b :, S - b :] = self.readout_value

        # Label is the motion block, to be recalled at the readout location.
        label = block

        return canvas, label


def _seed_motion_worker(worker_id: int) -> None:
    """Seed a worker's motion generator from PyTorch's unique worker seed."""
    worker = torch.utils.data.get_worker_info()
    if worker is not None:
        worker.dataset.generator.manual_seed(worker.seed)


class SpatialRecall3DMotionDataModule(pl.LightningDataModule):
    """DataModule for the 3D motion spatial recall (moving-digit copy) task.

    Wraps an image base datamodule (EMNIST, MNIST, ...) to produce cubic
    ``canvas_size³`` volumes containing a moving-digit ``block_size³`` tube
    that must be recalled at the readout corner.

    Args:
        base_datamodule_cfg: A LazyConfig for the base datamodule.
        digit_size: Size the 2D image is resized to before stamping.
        block_size: Edge length of the cubic motion block (= time steps).
        canvas_size: Edge length of the cubic canvas.
        data_type: "volume" ([B, D, H, W, C]) or "sequence" ([B, D*H*W, C]).
        placement: "fixed" or "random" block placement.
        readout_value: Value to fill the readout region with.
        max_step: Continuous per-axis step cap; integer steps can reach ``ceil(max_step)``.
        spin: Turn the digit in-plane in 90° steps.
    """

    def __init__(
        self,
        base_datamodule_cfg: LazyConfig,
        digit_size: int,
        block_size: int,
        canvas_size: int,
        data_type: Literal["sequence", "volume"] = "volume",
        placement: Literal["fixed", "random"] = "fixed",
        readout_value: float = 0.0,
        max_step: float = 2.0,
        spin: bool = True,
    ) -> None:
        """Initialize the SpatialRecall3DMotionDataModule."""
        super().__init__()

        if data_type not in ("sequence", "volume"):
            raise ValueError("data_type must be 'sequence' or 'volume'.")
        # Validate before setup/download, using the same contract as the dataset.
        _validate_motion_geometry(digit_size, block_size, canvas_size, placement, max_step)

        self._base_datamodule_cfg = base_datamodule_cfg
        self._base_datamodule: Optional[pl.LightningDataModule] = None

        self.digit_size = digit_size
        self.block_size = block_size
        self.canvas_size = canvas_size
        self.data_type = data_type
        self.placement = placement
        self.readout_value = readout_value
        self.max_step = max_step
        self.spin = spin

        # Computed property.
        self.canvas_volume = canvas_size * canvas_size * canvas_size

        # Properties from base datamodule.
        self._batch_size: Optional[int] = None
        self._num_workers: Optional[int] = None
        self._pin_memory: Optional[bool] = None
        self._seed: Optional[int] = None

        # Generators.
        self._generator: Optional[torch.Generator] = None
        self._train_generator: Optional[torch.Generator] = None
        self._val_generator: Optional[torch.Generator] = None
        self._test_generator: Optional[torch.Generator] = None

        # Datasets.
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _instantiate_base_datamodule(self) -> pl.LightningDataModule:
        """Instantiate the base datamodule from LazyConfig."""
        if self._base_datamodule is None:
            self._base_datamodule = instantiate(self._base_datamodule_cfg)
        return self._base_datamodule

    def _extract_base_properties(self) -> None:
        """Extract properties from the base datamodule."""
        base = self._base_datamodule
        self._batch_size = base.batch_size
        self._num_workers = base.num_workers
        self._pin_memory = base.pin_memory
        self._seed = base.seed

        # Include rank for both DataLoader worker seeds and num_workers=0.
        # Keep the base seed unchanged: dataset splits must agree across ranks.
        rank = self.trainer.global_rank if self.trainer is not None else 0
        if self.trainer is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        def generator(stream: int) -> torch.Generator:
            seed = int(np.random.SeedSequence([self._seed, rank, stream]).generate_state(1, dtype=np.uint64)[0])
            return torch.Generator().manual_seed(seed)

        self._generator = generator(0)
        self._train_generator = generator(1)
        self._val_generator = generator(2)
        self._test_generator = generator(3)

    @property
    def input_channels(self) -> int:
        """Read source image channels, available before setup or download.

        Returns:
            Number of image channels advertised by the base datamodule.
        """
        return self._instantiate_base_datamodule().input_channels

    @property
    def output_channels(self) -> int:
        """Read the recall target's channel count.

        Returns:
            Source image channels; the base class-label count is irrelevant.
        """
        return self.input_channels

    @property
    def batch_size(self) -> int:
        """Read the batch size.

        Returns:
            Batch size from the initialized base datamodule.
        """
        if self._batch_size is None:
            raise RuntimeError("Call setup() before accessing batch_size.")
        return self._batch_size

    @property
    def num_workers(self) -> int:
        """Read the worker count.

        Returns:
            Worker count from the initialized base datamodule.
        """
        if self._num_workers is None:
            raise RuntimeError("Call setup() before accessing num_workers.")
        return self._num_workers

    @property
    def pin_memory(self) -> bool:
        """Read the pinned-memory setting.

        Returns:
            Pinned-memory setting from the initialized base datamodule.
        """
        if self._pin_memory is None:
            raise RuntimeError("Call setup() before accessing pin_memory.")
        return self._pin_memory

    @property
    def seed(self) -> int:
        """Read the base random seed.

        Returns:
            Base random seed from the initialized base datamodule.
        """
        if self._seed is None:
            raise RuntimeError("Call setup() before accessing seed.")
        return self._seed

    def prepare_data(self) -> None:
        """Delegate source-data preparation to the base datamodule.

        Returns:
            None. The base datamodule may download data.
        """
        base = self._instantiate_base_datamodule()
        base.prepare_data()

    def _make_dataset(self, base_dataset: Dataset, generator: torch.Generator) -> Dataset:
        """Construct a SpatialRecall3DMotionDataset over ``base_dataset``.

        Args:
            base_dataset: Split-specific base dataset (train/val/test).
            generator: Split-specific random generator.

        Returns:
            The wrapped motion recall dataset.
        """
        return SpatialRecall3DMotionDataset(
            base_dataset,
            self.digit_size,
            self.block_size,
            self.canvas_size,
            generator,
            self.placement,
            self.readout_value,
            self.max_step,
            self.spin,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets and seeded generators for the requested stage.

        Args:
            stage: Lightning stage: "fit", "validate", "test", or None for all.

        Returns:
            None. Populates the corresponding dataset attributes.
        """
        base = self._instantiate_base_datamodule()
        # MNIST/EMNIST initialize their validation split during fit setup.
        base.setup("fit" if stage == "validate" else stage)
        self._extract_base_properties()

        if stage in ("fit", None):
            self.train_dataset = self._make_dataset(base.train_dataset, self._train_generator)
        if stage in ("fit", "validate", None):
            self.val_dataset = self._make_dataset(base.val_dataset, self._val_generator)

        if stage in ("test", None):
            self.test_dataset = self._make_dataset(base.test_dataset, self._test_generator)

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
            worker_init_fn=_seed_motion_worker,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader.

        Returns:
            DataLoader: Batches of canvases [B, C, S, S, S] and targets
            [B, C, b, b, b], before the batch-transfer hook.
        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train dataloader.")
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader.

        Returns:
            DataLoader: Batches of canvases [B, C, S, S, S] and targets
            [B, C, b, b, b], before the batch-transfer hook.
        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') or setup('validate') before requesting val dataloader.")
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader.

        Returns:
            DataLoader: Batches of canvases [B, C, S, S, S] and targets
            [B, C, b, b, b], before the batch-transfer hook.
        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test dataloader.")
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(self, batch, dataloader_idx) -> dict:
        """Rearrange batch tensors to the expected format.

        For volume: input [B, C, D, H, W] -> [B, D, H, W, C]; label
        [B, C, b, b, b] -> [B, b, b, b, C].
        For sequence: input -> [B, D*H*W, C]; label -> [B, b*b*b, C].

        Args:
            batch: Canvas/target tensors [B, C, S, S, S] and [B, C, b, b, b].
            dataloader_idx: Lightning loader index; unused.

        Returns:
            dict: Channels-last or flattened input and label, plus condition=None.
        """
        x, y = batch

        if self.data_type == "volume":
            x = rearrange(x, "b c d h w -> b d h w c")
            y = rearrange(y, "b c d h w -> b d h w c")
        elif self.data_type == "sequence":
            x = rearrange(x, "b c d h w -> b (d h w) c")
            y = rearrange(y, "b c d h w -> b (d h w) c")
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
            use_colored_frames=args.colored_frames,
            num_items=args.num_items,
            readout_value=args.readout_value,
            colored_label=args.colored_label,
        )

    dm.prepare_data()
    dm.setup("fit")

    print(f"Mode: {args.mode.upper()}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "emnist":
        print(f"EMNIST split: {args.emnist_split}")
    print(f"Placement: {args.placement}")
    print(f"With mask: {args.with_mask}")
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
        import numpy as np

        if args.colored_frames:
            # RGB 1D mode: x is [B, 3, L], y is [B, 3, seg_len] (or [B, 1, seg_len])
            # Show R/G/B channels as separate lines + label as RGB 2D image
            num_cols = 2
            fig, axes = plt.subplots(B, num_cols, figsize=(12, 2.5 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            for i in range(B):
                # Canvas: plot each RGB channel
                for ch, ch_color in enumerate(["red", "green", "blue"]):
                    axes[i, 0].plot(x[i, ch, :].cpu().numpy(), linewidth=0.4, color=ch_color, alpha=0.8)
                axes[i, 0].set_ylim(-0.1, 1.1)
                # Shade readout region
                seg_len = args.target_size * args.target_size
                canvas_len = x.shape[2]
                axes[i, 0].axvspan(canvas_len - seg_len, canvas_len, alpha=0.1, color="green", label="readout")
                if i == 0:
                    axes[i, 0].set_title("Canvas (1D RGB)")
                    axes[i, 0].legend(loc="upper right", fontsize=6)
                axes[i, 0].set_xlabel("Position")

                # Label reshaped to 2D
                if y.shape[1] == 3:
                    # RGB label: [3, seg_len] -> [H, W, 3]
                    label_rgb = y[i].cpu().numpy().reshape(3, args.target_size, args.target_size)
                    label_rgb = np.transpose(label_rgb, (1, 2, 0))
                    axes[i, 1].imshow(np.clip(label_rgb, 0, 1))
                    if i == 0:
                        axes[i, 1].set_title(f"Label RGB ({args.target_size}×{args.target_size})")
                else:
                    label_2d = y[i, 0, :].cpu().reshape(args.target_size, args.target_size)
                    axes[i, 1].imshow(label_2d, cmap="gray")
                    if i == 0:
                        axes[i, 1].set_title(f"Label ({args.target_size}×{args.target_size})")
                axes[i, 1].axis("off")
        elif args.with_mask:
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
