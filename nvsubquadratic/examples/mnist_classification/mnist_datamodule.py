# David W. Romero, 2025-09-09

"""MNIST datamodule."""

from typing import Literal

import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Global seed value used for worker initialization
_BASE_SEED = 0


def set_base_seed(seed):
    """Set the base seed for worker initialization"""
    global _BASE_SEED
    _BASE_SEED = seed


# Define a worker initialization function to set seeds for data loading workers
def deterministic_worker_init_fn(worker_id):
    """
    Initialize the worker with a deterministic seed derived from base_seed and worker_id.
    Each worker gets a unique but deterministic seed: base_seed + worker_id
    """
    # Use the global base seed plus worker_id as the seed for this worker
    global _BASE_SEED
    seed = _BASE_SEED + worker_id

    # Set Python hash seed for this process
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set all relevant random states with this seed
    random.seed(seed)  # Set Python's random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU RNG seed
    torch.cuda.manual_seed(seed)  # Set CUDA RNG seed for current device
    torch.cuda.manual_seed_all(seed)  # Set CUDA RNG seed for all devices


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_type: Literal["image"],
        num_workers: int,
        pin_memory: bool,
        permuted: bool,
        use_deterministic_worker_init: bool,
        seed: int,
    ):
        super().__init__()

        # Save parameters to self
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.permuted = permuted
        self.seed = seed

        # Create a generator with the given seed for reproducibility
        self.generator = torch.Generator().manual_seed(seed)

        # Handle worker initialization. Use deterministic worker initialization if specified.
        self.worker_init_fn = deterministic_worker_init_fn if use_deterministic_worker_init else None

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 10

        # Assert that data_type is in the allowed options
        assert data_type == "image", f"data_type must be 'image', got {data_type}"
        self.data_type = data_type

        # Create transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download data, train then test
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            mnist = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.transform,
            )
            # Use deterministic split with our generator
            self.train_dataset, self.val_dataset = random_split(mnist, [55000, 5000], generator=self.generator)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
            )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
            persistent_workers=self.num_workers > 0,  # Keep workers alive between epochs
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.data_type == "image":
            # If image, rearrange the input [B, C, Y, X] -> [B, Y, X, C]
            x, y = batch
            x = rearrange(x, "b c y x -> b y x c")
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        batch = x, y
        return batch
