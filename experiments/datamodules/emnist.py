"""EMNIST datamodule for PyTorch Lightning.

Supports all EMNIST splits:
    - digits:   10 classes (0-9), 280k samples, balanced
    - letters:  26 classes (A-Z merged), 145.6k samples, balanced
    - balanced: 47 classes (digits + letters), 131.6k samples, balanced
    - bymerge:  47 classes (similar letters merged), 814k samples, unbalanced
    - byclass:  62 classes (all separate), 814k samples, unbalanced

Usage:
    PYTHONPATH=. python experiments/datamodules/emnist.py
"""

from typing import Literal, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


EMNIST_MEAN: Sequence[float] = (0.1736,)
EMNIST_STD: Sequence[float] = (0.3317,)


def _fix_emnist_orientation(img: torch.Tensor) -> torch.Tensor:
    """Rotate EMNIST samples to match upright MNIST orientation."""
    # torchvision's EMNIST ships transposed; rotate 90 degrees clockwise
    return torch.rot90(torch.flip(img, dims=[2]), k=1, dims=[1, 2])


class EMNISTDataModule(pl.LightningDataModule):
    """EMNIST datamodule mirroring the MNIST interface used in the project.

    Args:
        data_dir: Path to store/load the EMNIST data.
        batch_size: Batch size for dataloaders.
        data_type: Either "sequence" (flattened 784-length) or "image" (28x28).
        num_workers: Number of dataloader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        permuted: If True and data_type="sequence", apply a fixed permutation.
        seed: Random seed for reproducibility.
        normalize_input: Whether to normalize inputs with EMNIST mean/std.
        split: Which EMNIST split to use:
            - "digits": 10 classes (0-9), 280k samples, balanced
            - "letters": 26 classes (A-Z merged), 145.6k samples, balanced
            - "balanced": 47 classes (digits + letters), 131.6k samples, balanced
            - "bymerge": 47 classes (similar letters merged), 814k samples, unbalanced
            - "byclass": 62 classes (all separate), 814k samples, unbalanced
        use_test_as_val: If True, use test set for validation instead of splitting train.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_type: Literal["sequence", "image"],
        num_workers: int,
        pin_memory: bool,
        permuted: bool,
        seed: int,
        normalize_input: bool = True,
        split: Literal["digits", "balanced", "letters", "byclass", "bymerge"] = "digits",
        use_test_as_val: bool = False,
    ) -> None:
        """Initialize the EMNISTDataModule."""
        super().__init__()

        assert data_type in ("sequence", "image"), "data_type must be 'sequence' or 'image'."
        assert split in {"digits", "balanced", "letters", "byclass", "bymerge"}, f"Unsupported EMNIST split '{split}'."

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_type = data_type
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.permuted = permuted
        self.seed = seed
        self.normalize_input = normalize_input
        self.split = split
        self.use_test_as_val = use_test_as_val

        # Deterministic RNGs
        self.split_gen = torch.Generator().manual_seed(seed)
        self.train_gen_loader = torch.Generator().manual_seed(seed + 1000)
        self.val_gen_loader = torch.Generator().manual_seed(seed + 1001)
        self.test_gen_loader = torch.Generator().manual_seed(seed + 1002)

        # dataset metadata
        self.input_channels = 1
        self.output_channels = {
            "digits": 10,
            "letters": 26,
            "balanced": 47,
            "bymerge": 47,
            "byclass": 62,
        }[split]

        # build transform
        transforms_list = [
            transforms.ToTensor(),
            transforms.Lambda(_fix_emnist_orientation),
        ]
        if self.normalize_input:
            transforms_list.append(transforms.Normalize(EMNIST_MEAN, EMNIST_STD))
        self.transform = transforms.Compose(transforms_list)

        self.normalizer = transforms.Normalize(EMNIST_MEAN, EMNIST_STD)

        # Containers populated in setup
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        if self.data_type == "sequence" and self.permuted:
            permutation = np.random.RandomState(seed=self.seed).permutation(28 * 28).astype(np.int64)
            self.permutation = torch.from_numpy(permutation).long()
        else:
            self.permutation = None

    def prepare_data(self) -> None:
        """Download the data."""
        # download data, train then test
        datasets.EMNIST(self.data_dir, split=self.split, train=True, download=True)
        datasets.EMNIST(self.data_dir, split=self.split, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the datamodules."""
        # we set up only relevant datamodules when stage is specified
        if stage in ("fit", None):
            full_train = datasets.EMNIST(
                self.data_dir,
                split=self.split,
                train=True,
                transform=self.transform,
            )

            if self.use_test_as_val:
                self.train_dataset = full_train
                self.val_dataset = datasets.EMNIST(
                    self.data_dir,
                    split=self.split,
                    train=False,
                    transform=self.transform,
                )
            else:
                num_total = len(full_train)
                num_val_samples = max(1, round(num_total * 0.1))
                num_val_samples = min(num_val_samples, num_total - 1)
                num_train_samples = num_total - num_val_samples
                self.train_dataset, self.val_dataset = random_split(
                    full_train,
                    [num_train_samples, num_val_samples],
                    generator=self.split_gen,
                )

        if stage in ("test", None):
            self.test_dataset = datasets.EMNIST(
                self.data_dir,
                split=self.split,
                train=False,
                transform=self.transform,
            )

    def _build_dataloader(
        self, dataset: Dataset, generator: torch.Generator, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Create a dataloader for the given dataset.

        This function is reused for train, val and test dataloaders.

        Args:
            dataset: Dataset to create a dataloader for.
            generator: Random generator for reproducibility.
            shuffle: Whether to shuffle the dataset.
            drop_last: Whether to drop the last batch if it's not complete.

        Returns:
            DataLoader for the dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=generator,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the train dataloader.")
        return self._build_dataloader(self.train_dataset, self.train_gen_loader, shuffle=True, drop_last=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        """Val dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting the val dataloader.")
        return self._build_dataloader(self.val_dataset, self.val_gen_loader, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting the test dataloader.")
        return self._build_dataloader(self.test_dataset, self.test_gen_loader, shuffle=False, drop_last=False)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Function to rearrange the input.

        For image data_type, from [B, C, H, W] to [B, H, W, C].
        For sequence data_type, from [B, C, H, W] to [B, -1, C].
        """
        x, y = batch
        if self.data_type == "image":
            # If image, rearrange the input [B, C, H, W] -> [B, H, W, C]
            x = rearrange(x, "b c h w -> b h w c")
        elif self.data_type == "sequence":
            # If sequential, flatten the input [B, C, H, W] -> [B, -1, C]
            x = rearrange(x, "b c h w -> b (h w) c")
            if self.permutation is not None:
                x = x[:, self.permutation, :]
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        return x, y


if __name__ == "__main__":
    import argparse
    import math
    import os

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Visualize EMNIST samples")
    parser.add_argument(
        "--split",
        type=str,
        default="digits",
        choices=["digits", "letters", "balanced", "bymerge", "byclass"],
        help="EMNIST split to use (default: digits)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: EMNIST_DATA_DIR env or ./.data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="_tmp",
        help="Output directory for saved images (default: _tmp/)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    data_dir = args.data_dir or os.environ.get("EMNIST_DATA_DIR", "./.data")
    os.makedirs(args.output_dir, exist_ok=True)

    dm = EMNISTDataModule(
        data_dir=data_dir,
        batch_size=16,
        data_type="image",
        num_workers=0,
        pin_memory=False,
        permuted=False,
        seed=0,
        normalize_input=False,
        split=args.split,
    )
    dm.prepare_data()
    dm.setup("fit")

    print(f"Split: {args.split}")
    print(f"Number of classes: {dm.output_channels}")
    print(f"Train samples: {len(dm.train_dataset)}")
    print(f"Val samples: {len(dm.val_dataset)}")

    loader = dm.train_dataloader()
    imgs, labels = next(iter(loader))

    imgs = imgs[:16].detach().cpu()
    labels = labels[:16].detach().cpu()

    n_images = imgs.shape[0]
    n_cols = 8
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6), dpi=120)
    axes = axes.flatten()

    for idx in range(n_rows * n_cols):
        ax = axes[idx]
        if idx < n_images:
            img = imgs[idx]
            ax.imshow(img.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"label: {int(labels[idx])}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"EMNIST {args.split} (orientation check)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(args.output_dir, f"emnist_{args.split}_grid.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
