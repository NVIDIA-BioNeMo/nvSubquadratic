"""ARC-AGI (Abstraction and Reasoning Corpus) datamodule.

Data format (from official ARC-AGI repository):
    <data_dir>/
        training/   # 400 tasks, each is a JSON file
        evaluation/ # 400 tasks, used as test set

Each JSON file has the structure::

    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test":  [{"input": [[...]], "output": [[...]]}, ...]
    }

where grids are lists-of-lists of integers in [0, 9].

Augmentation (ported from VARC: github.com/lillian039/VARC):
    - Resolution augmentation: upscale grids by an integer factor sampled
      uniformly in [1, floor((max_size - 2) / max_grid_dim)].
    - Spatial-translation augmentation: randomly place the scaled grid on the
      *max_size × max_size* canvas with 1-cell minimum border on each side.
    - Colour-permutation augmentation: randomly permute the 10 ARC colours.

Sentinel constants:
    IGNORE_INDEX = 10  — padding pixels, excluded from loss via ignore_index
    PAD_INDEX    = 11  — border markers indicating the output grid's shape
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Sentinel constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = 10  # Marks canvas padding — excluded from cross-entropy loss
PAD_INDEX = 11  # Border markers on output canvas — indicate output shape
NUM_COLORS = 10  # ARC colours 0-9


# ---------------------------------------------------------------------------
# Low-level grid helpers
# ---------------------------------------------------------------------------


def _scale_grid(grid: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour integer upscaling."""
    if scale == 1:
        return grid
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)


def _place_on_canvas(
    grid: np.ndarray,
    canvas_size: int,
    y_offset: int,
    x_offset: int,
    fill: int,
) -> np.ndarray:
    """Place *grid* on a *canvas_size × canvas_size* canvas filled with *fill*."""
    canvas = np.full((canvas_size, canvas_size), fill, dtype=np.int64)
    h, w = grid.shape
    canvas[y_offset : y_offset + h, x_offset : x_offset + w] = grid
    return canvas


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ARCDataset(Dataset):
    """Flat dataset of (input_grid, output_grid) pairs drawn from ARC tasks.

    Args:
        task_files: List of *(task_id, path)* pairs — one per task file.
        subset: ``"train"`` or ``"test"`` — which inner split to use.
        max_size: Canvas size (square).  Grids are padded to *max_size × max_size*.
        disable_translation: Pin grid at offset (1, 1) with no random jitter.
        disable_resolution_aug: Use *fix_scale_factor* as the fixed scale.
        fix_scale_factor: Scale used when resolution augmentation is disabled.
        num_color_permutations: Extra copies of each example with shuffled
            colour palettes (0 = no permutation augmentation).
        augment: Master toggle.  When *False* all augmentations are disabled
            (intended for validation / test).
        seed: RNG seed for reproducible permutation sampling.
    """

    def __init__(
        self,
        task_files: list[tuple[int, Path]],
        subset: str,
        max_size: int = 32,
        disable_translation: bool = False,
        disable_resolution_aug: bool = False,
        fix_scale_factor: int = 1,
        num_color_permutations: int = 9,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        """Build the flat index of (task_id, examples, example_idx, colour_perm) entries."""
        super().__init__()
        self.subset = subset
        self.max_size = max_size
        self.disable_translation = disable_translation
        self.disable_resolution_aug = disable_resolution_aug
        self.fix_scale_factor = fix_scale_factor
        self.augment = augment

        rng = random.Random(seed)

        # Maximum grid dimension that can fit on the canvas with a 1-cell border
        # on each side (same constraint as VARC's pad_grid_with_translation).
        max_grid_dim = max_size - 2

        # Build flat index: list of (task_id, examples_list, example_idx, colour_perm | None)
        # Colour perm is a length-10 list mapping old colour → new colour.
        self._index: list[tuple[int, list, int, list[int] | None]] = []
        for task_id, path in task_files:
            task_json = json.loads(path.read_text())
            examples = task_json.get(subset, [])
            n_perms = num_color_permutations if augment else 0
            for ei, example in enumerate(examples):
                # Skip grids that cannot fit on the canvas at scale=1 with a
                # 1-cell border. VARC skips these implicitly (randint(1,0) crash).
                inp = np.array(example["input"], dtype=np.int64)
                out = np.array(example.get("output", [[0]]), dtype=np.int64)
                if max(inp.shape[0], inp.shape[1], out.shape[0], out.shape[1]) > max_grid_dim:
                    continue
                # Always include one identity (no permutation) copy
                self._index.append((task_id, examples, ei, None))
                # Extra colour-permuted copies
                for _ in range(n_perms):
                    perm = list(range(NUM_COLORS))
                    rng.shuffle(perm)
                    self._index.append((task_id, examples, ei, perm))

    def __len__(self) -> int:
        """Return the total number of (example, colour-perm) entries."""
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return an augmented (input, label, mask, task_id) dict for index *idx*."""
        task_id, examples, ei, perm = self._index[idx]
        example = examples[ei]

        inp = np.array(example["input"], dtype=np.int64)  # (H_in, W_in)
        out = np.array(example["output"], dtype=np.int64)  # (H_out, W_out)

        # ── Colour permutation ──────────────────────────────────────────────
        if perm is not None:
            perm_arr = np.array(perm, dtype=np.int64)
            inp = perm_arr[inp]
            out = perm_arr[out]

        # ── Resolution augmentation ─────────────────────────────────────────
        max_dim = max(inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
        if self.disable_resolution_aug or not self.augment:
            scale = self.fix_scale_factor
        else:
            # Reserve 2 cells for borders → max usable canvas is (max_size - 2)
            max_scale = max(1, (self.max_size - 2) // max_dim)
            scale = random.randint(1, max_scale)

        inp = _scale_grid(inp, scale)
        out = _scale_grid(out, scale)

        in_h, in_w = inp.shape
        out_h, out_w = out.shape

        # ── Spatial translation augmentation ───────────────────────────────
        scaled_max = max(in_h, in_w, out_h, out_w)
        avail = max(0, self.max_size - scaled_max - 2)
        if self.disable_translation or not self.augment:
            y_off = x_off = 1
        else:
            y_off = random.randint(1, 1 + avail)
            x_off = random.randint(1, 1 + avail)

        # ── Place on canvas ─────────────────────────────────────────────────
        inp_canvas = _place_on_canvas(inp, self.max_size, y_off, x_off, IGNORE_INDEX)
        out_canvas = _place_on_canvas(out, self.max_size, y_off, x_off, IGNORE_INDEX)

        # ── Attention mask (1 = valid pixel, 0 = padding) ───────────────────
        attn_mask = (inp_canvas != IGNORE_INDEX).astype(np.int64)

        # ── Output-shape border markers ─────────────────────────────────────
        # Right border column of the output region
        if x_off + out_w < self.max_size:
            out_canvas[y_off : y_off + out_h, x_off + out_w] = PAD_INDEX
        # Bottom border row of the output region (includes corner)
        if y_off + out_h < self.max_size:
            out_canvas[y_off + out_h, x_off : x_off + out_w + 1] = PAD_INDEX

        return {
            "input": torch.from_numpy(inp_canvas),  # [max_size, max_size] long
            "label": torch.from_numpy(out_canvas),  # [max_size, max_size] long
            "attention_mask": torch.from_numpy(attn_mask),  # [max_size, max_size] long
            "task_id": torch.tensor(task_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def _collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack dict items into batch tensors."""
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class ARCDataModule(pl.LightningDataModule):
    """Lightning DataModule for the ARC-AGI benchmark.

    Expected directory layout (official ARC-AGI-1 structure)::

        data_dir/
            training/   *.json   # used for train + val
            evaluation/ *.json   # used for test

    Args:
        data_dir: Root of the ARC data directory (contains ``training/`` and
            ``evaluation/`` subdirectories with JSON task files).
        batch_size: Batch size per GPU.
        num_workers: DataLoader worker processes.
        pin_memory: Pin CPU tensors before transfer to GPU.
        seed: Random seed for val split and augmentation sampling.
        max_size: Canvas size (square).  Grids are padded to *max_size × max_size*.
        val_fraction: Fraction of training tasks held out for validation.
        disable_translation: Disable spatial-translation augmentation.
        disable_resolution_aug: Disable resolution-scaling augmentation.
        fix_scale_factor: Fixed scale when resolution augmentation is disabled.
        num_color_permutations: Colour-permutation copies per example during
            training (0 = no colour augmentation).
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int = 42,
        max_size: int = 32,
        val_fraction: float = 0.1,
        disable_translation: bool = False,
        disable_resolution_aug: bool = False,
        fix_scale_factor: int = 1,
        num_color_permutations: int = 9,
    ) -> None:
        """Store datamodule configuration; datasets are built lazily in ``setup()``."""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.max_size = max_size
        self.val_fraction = val_fraction
        self.disable_translation = disable_translation
        self.disable_resolution_aug = disable_resolution_aug
        self.fix_scale_factor = fix_scale_factor
        self.num_color_permutations = num_color_permutations

        self.train_dataset: ARCDataset | None = None
        self.val_dataset: ARCDataset | None = None
        self.test_dataset: ARCDataset | None = None

        # num_tasks is populated in setup() and used by the network configs
        self.num_tasks: int | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_task_files(self, split_dir: str) -> list[tuple[int, Path]]:
        """Return sorted list of *(task_id, path)* from a split directory."""
        paths = sorted((self.data_dir / split_dir).glob("*.json"))
        return [(i, p) for i, p in enumerate(paths)]

    def _build_dataset(
        self,
        task_files: list[tuple[int, Path]],
        subset: str,
        augment: bool,
    ) -> ARCDataset:
        return ARCDataset(
            task_files=task_files,
            subset=subset,
            max_size=self.max_size,
            disable_translation=self.disable_translation,
            disable_resolution_aug=self.disable_resolution_aug,
            fix_scale_factor=self.fix_scale_factor,
            num_color_permutations=self.num_color_permutations,
            augment=augment,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Split training tasks into train/val and set up test from evaluation."""
        train_files = self._load_task_files("training")
        test_files = self._load_task_files("evaluation")

        # Assign global task ids: training tasks 0..N-1, eval tasks N..N+M-1
        n_train_tasks = len(train_files)
        test_files = [(task_id + n_train_tasks, p) for task_id, p in test_files]

        self.num_tasks = n_train_tasks + len(test_files)

        # Split training tasks into train / val
        rng = random.Random(self.seed)
        shuffled = list(train_files)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * self.val_fraction))
        val_task_files = shuffled[:n_val]
        train_task_files = shuffled[n_val:]

        if stage in ("fit", None):
            # "train" subset = the 'train' examples inside each task JSON
            self.train_dataset = self._build_dataset(train_task_files, "train", augment=True)
            # Val uses the same 'train' examples (ARC test inputs have no labels)
            self.val_dataset = self._build_dataset(val_task_files, "train", augment=False)

        if stage in ("test", None):
            self.test_dataset = self._build_dataset(test_files, "train", augment=False)

    def _build_loader(self, dataset: ARCDataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            collate_fn=_collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader with shuffling and drop_last."""
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader (ARC evaluation split)."""
        return self._build_loader(self.test_dataset, shuffle=False)

    def on_before_batch_transfer(self, batch: dict[str, torch.Tensor], dataloader_idx: int) -> dict[str, torch.Tensor]:
        """Reformat raw batch into the codebase's standard ``{input, label, condition}`` layout.

        The ``condition`` dict carries per-example context that is not part of the
        primary input grid: the task ID (for task-token lookup) and the attention
        mask (marks valid vs. padding pixels).

        Returns:
            dict with keys:

            - ``"input"``         — ``[B, H, W]`` long, colour indices 0-11
            - ``"label"``         — ``[B, H, W]`` long, colour indices 0-9 + IGNORE_INDEX
            - ``"condition"``     — dict with:
                - ``"task_id"``       — ``[B]`` long
                - ``"attention_mask"`` — ``[B, H, W]`` long (1=valid, 0=padding)
        """
        return {
            "input": batch["input"],
            "label": batch["label"],
            "condition": {
                "task_id": batch["task_id"],
                "attention_mask": batch["attention_mask"],
            },
        }
