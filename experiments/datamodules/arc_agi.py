"""DataModule for the ARC-AGI-1 challenge hosted on Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


_DEFAULT_INPUT_PAD_VALUE = 0
_DEFAULT_LABEL_PAD_VALUE = -100  # Plays nicely with CrossEntropyLoss(ignore_index=-100).


@dataclass(frozen=True)
class _ArcPair:
    task_id: str
    pair_type: Literal["train", "test"]
    pair_index: int
    input_grid: list[list[int]]
    output_grid: Optional[list[list[int]]]


def _max_grid_dim(grid: list[list[int]] | None) -> int:
    if not grid:
        return 0
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return max(height, width)


def _pad_to_square(grid: list[list[int]], size: int, pad_value: int) -> torch.Tensor:
    """Pad a 2D list to size x size with pad_value."""
    tensor = torch.full((size, size), pad_value, dtype=torch.long)
    if not grid:
        return tensor

    src = torch.tensor(grid, dtype=torch.long)
    height, width = src.shape
    tensor[:height, :width] = src
    return tensor


class _ArcAGIPairDataset(Dataset):
    """Flatten ARC-AGI tasks into individual (input, output) pairs."""

    def __init__(
        self,
        *,
        split: str,
        dataset_name: str,
        dataset_config: Optional[str],
        cache_dir: Path,
        hf_token: Optional[str],
        include_test_pairs: bool,
        max_grid_size: Optional[int],
        input_pad_value: int,
        label_pad_value: int,
        num_colors: int,
        normalize_inputs: bool,
        one_hot_inputs: bool,
    ) -> None:
        super().__init__()
        self.input_pad_value = input_pad_value
        self.label_pad_value = label_pad_value
        self.normalize_inputs = normalize_inputs
        self.one_hot_inputs = one_hot_inputs
        self.num_colors = num_colors

        raw_dataset = load_dataset(
            path=dataset_name,
            name=dataset_config,
            split=split,
            streaming=False,
            cache_dir=str(cache_dir),
            token=hf_token,
        )

        pairs: list[_ArcPair] = []
        observed_max = 0
        for task_idx, task in enumerate(raw_dataset):
            task_id = task.get("id", str(task_idx))

            for pair_idx, pair in enumerate(task.get("train", [])):
                pairs.append(
                    _ArcPair(
                        task_id=task_id,
                        pair_type="train",
                        pair_index=pair_idx,
                        input_grid=pair["input"],
                        output_grid=pair.get("output"),
                    )
                )
                observed_max = max(
                    observed_max,
                    _max_grid_dim(pair.get("input")),
                    _max_grid_dim(pair.get("output")),
                )

            if include_test_pairs:
                for pair_idx, pair in enumerate(task.get("test", [])):
                    pairs.append(
                        _ArcPair(
                            task_id=task_id,
                            pair_type="test",
                            pair_index=pair_idx,
                            input_grid=pair["input"],
                            output_grid=pair.get("output"),
                        )
                    )
                    observed_max = max(
                        observed_max,
                        _max_grid_dim(pair.get("input")),
                        _max_grid_dim(pair.get("output")),
                    )

        if not pairs:
            raise RuntimeError(f"No pairs were found in split '{split}' for dataset '{dataset_name}'.")

        if max_grid_size is None:
            max_grid_size = observed_max

        self.max_grid_size = max_grid_size
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]

        input_tensor = _pad_to_square(pair.input_grid, size=self.max_grid_size, pad_value=self.input_pad_value)

        if pair.output_grid is not None:
            label_tensor = _pad_to_square(pair.output_grid, size=self.max_grid_size, pad_value=self.label_pad_value)
            label_mask = label_tensor != self.label_pad_value
        else:
            label_tensor = torch.full(
                (self.max_grid_size, self.max_grid_size),
                self.label_pad_value,
                dtype=torch.long,
            )
            label_mask = torch.zeros_like(label_tensor, dtype=torch.bool)

        if self.normalize_inputs:
            input_tensor = input_tensor.float() / float(max(self.num_colors - 1, 1))
        else:
            input_tensor = input_tensor.float()

        if self.one_hot_inputs:
            # Append a channel dimension with one-hot encoding over colors.
            input_tensor = F.one_hot(
                input_tensor.long().clamp(min=0, max=self.num_colors - 1),
                num_classes=self.num_colors,
            ).float()
        else:
            input_tensor = input_tensor.unsqueeze(-1)  # (H, W, 1)

        return {
            "input": input_tensor,  # (H, W, C)
            "label": label_tensor,  # (H, W)
            "label_mask": label_mask,  # (H, W)
            "task_id": pair.task_id,
            "pair_type": pair.pair_type,
            "pair_index": pair.pair_index,
        }


class ArcAGIDataModule(pl.LightningDataModule):
    """Lightning DataModule for the ARC-AGI-1 challenge on Hugging Face."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        hf_dataset_name: str = "dataartist/arc-agi",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        max_grid_size: Optional[int] = None,
        include_test_pairs: bool = False,
        normalize_inputs: bool = True,
        one_hot_inputs: bool = False,
        num_colors: int = 10,
        input_pad_value: int = _DEFAULT_INPUT_PAD_VALUE,
        label_pad_value: int = _DEFAULT_LABEL_PAD_VALUE,
        train_split: str = "training",
        val_split: str = "evaluation",
        test_split: Optional[str] = None,
        condition_on_label_mask: bool = False,
        condition_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset_config = hf_dataset_config
        self.hf_auth_token = hf_auth_token
        self.max_grid_size = max_grid_size
        self.include_test_pairs = include_test_pairs
        self.normalize_inputs = normalize_inputs
        self.one_hot_inputs = one_hot_inputs
        self.num_colors = num_colors
        self.input_pad_value = input_pad_value
        self.label_pad_value = label_pad_value
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split or val_split
        self.condition_on_label_mask = condition_on_label_mask
        self.condition_dim = condition_dim

        # Interfaces with downstream network instantiation.
        self.input_channels = num_colors if one_hot_inputs else 1
        self.output_channels = num_colors
        self.num_classes = num_colors
        self.ignore_index = label_pad_value

        self.train_dataset: Optional[_ArcAGIPairDataset] = None
        self.val_dataset: Optional[_ArcAGIPairDataset] = None
        self.test_dataset: Optional[_ArcAGIPairDataset] = None

        self.generator = torch.Generator().manual_seed(seed)

    def prepare_data(self) -> None:
        """Download/cache the requested splits."""
        load_dataset(
            path=self.hf_dataset_name,
            name=self.hf_dataset_config,
            split=self.train_split,
            streaming=False,
            cache_dir=str(self.data_dir),
            token=self.hf_auth_token,
        )

        # Validation split may be None for some datasets, hence best-effort here.
        if self.val_split:
            load_dataset(
                path=self.hf_dataset_name,
                name=self.hf_dataset_config,
                split=self.val_split,
                streaming=False,
                cache_dir=str(self.data_dir),
                token=self.hf_auth_token,
            )

    def _build_dataset(self, split: str) -> _ArcAGIPairDataset:
        return _ArcAGIPairDataset(
            split=split,
            dataset_name=self.hf_dataset_name,
            dataset_config=self.hf_dataset_config,
            cache_dir=self.data_dir,
            hf_token=self.hf_auth_token,
            include_test_pairs=self.include_test_pairs,
            max_grid_size=self.max_grid_size,
            input_pad_value=self.input_pad_value,
            label_pad_value=self.label_pad_value,
            num_colors=self.num_colors,
            normalize_inputs=self.normalize_inputs,
            one_hot_inputs=self.one_hot_inputs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct datasets for the requested stage."""
        if stage in ("fit", None):
            self.train_dataset = self._build_dataset(self.train_split)
            self.val_dataset = self._build_dataset(self.val_split)
        elif stage == "validate":
            self.val_dataset = self._build_dataset(self.val_split)
        elif stage == "test":
            self.test_dataset = self._build_dataset(self.test_split)

    def _build_loader(self, dataset: _ArcAGIPairDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            generator=self.generator,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataloader called before setup('fit').")
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataloader called before setup('fit' or 'validate').")
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.test_dataset = self._build_dataset(self.test_split)
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(self, batch: dict[str, Any], dataloader_idx: int) -> dict[str, Any]:
        """Map batch dicts to the {input, label, condition} structure expected downstream."""
        inputs = batch["input"]  # (B, H, W, C)
        labels = batch["label"]  # (B, H, W)
        masks = batch["label_mask"]  # (B, H, W)

        # Channel-last layout for compatibility with existing wrappers.
        inputs = inputs.contiguous()

        condition = {
            "label_mask": masks,
            "task_id": batch["task_id"],
            "pair_type": batch["pair_type"],
            "pair_index": batch["pair_index"],
        }

        batch_dict = {
            "input": inputs,
            "label": labels,
            "condition": condition,
        }
        if self.condition_on_label_mask:
            model_condition = masks.float().unsqueeze(-1)
            if self.condition_dim is not None:
                model_condition = model_condition.expand(-1, -1, -1, self.condition_dim)
            batch_dict["model_condition"] = model_condition

        return batch_dict
