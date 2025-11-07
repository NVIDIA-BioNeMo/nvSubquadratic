# TODO: Add license header here


"""Shallow Water Equations (SWE) datamodule.

This datamodule loads and prepares 2D Shallow Water Equations data from PDEArena,
following the experimental setup from https://github.com/PredictiveIntelligenceLab/cvit/tree/main/swe
"""

import os
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


KEYS = ["vor", "pres"]
NUM_EVAL = 16

PREV_STEPS = 2
PRED_STEPS = 1
EVAL_ROLLOUT_STEPS = 5


class SWEDataset(Dataset):
    """
    Dataset handling trajectory splitting and normalization for the Shallow Water Equations (SWE) data.
    
    Args:
        directory: Directory where the preprocessed train/valid/test .npy files are stored.
        mode: One of "train", "valid", or "test" to specify the dataset split.
        rollout_steps: Number of prediction steps to roll out.
    """
    def __init__(
        self, 
        directory: str, 
        mode: Literal["train", "valid", "test"], 
        rollout_steps: int, 
        prev_steps: int = 2
    ):
        super().__init__()
        
        self.directory = directory
        self.mode = mode
        self.rollout_steps = rollout_steps
        self.prev_steps = prev_steps

        data_path = os.path.join(directory, f"{mode}.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Preprocessed data file not found: {data_path}\n"
                f"Please run preprocess_swe_data.py first to generate the .npy files."
            )

        self.data = torch.from_numpy(np.load(data_path)).float()  # [N, T, H, W, C]
        
        if mode != "train":
            # For validation and test, use only a subset of the data for faster evaluation
            self.data = self.data[:NUM_EVAL]
            
        self.n_samples, self.T, self.H, self.W, self.C = self.data.shape
        
        normstats = torch.load(os.path.join(directory, "normstats.pt"))
        self.means = torch.stack([normstats[key]["mean"].squeeze() for key in KEYS], dim=-1)
        self.stds = torch.stack([normstats[key]["std"].squeeze() for key in KEYS], dim=-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        """Randomly sample and normalize a time window of length prev_steps + rollout_steps."""
        start_time = np.random.randint(0, self.T - self.rollout_steps - self.pred_steps + 1)
        end_time = start_time + self.prev_steps + self.rollout_steps
        sequence = self.data[index, start_time:end_time]
        return (sequence - self.means) / self.stds


class SWEDataModule(pl.LightningDataModule):
    """PDE-SWE Lightning data module."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_type: Literal["sequence", "image"],
        num_workers: int,
        pin_memory: bool,
        use_deterministic_worker_init: bool,
        seed: int,
    ):
        """Initialize the PDE-SWE DataModule.

        Args:
            data_dir: Directory to save the data
            batch_size: Batch size
            data_type: Type of data. Can be "sequence" or "image".
            num_workers: Number of workers
            pin_memory: Whether to pin memory
            use_deterministic_worker_init: Whether to use deterministic worker initialization
            seed: Seed for the data
        """
        super().__init__()

        # Save parameters to self
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        # Create a generator with the given seed for reproducibility
        self.generator = torch.Generator().manual_seed(seed)

        # Handle worker initialization. Use deterministic worker initialization if specified.
        # self.worker_init_fn = deterministic_worker_init_fn if use_deterministic_worker_init else None

        # Determine sizes of dataset
        self.input_channels = PREV_STEPS * len(KEYS)   # 2 time steps x (vorticity, pressure)
        self.output_channels = PRED_STEPS * len(KEYS)  # 1 time step x (vorticity, pressure)

        # Assert that data_type is in the allowed options
        assert data_type in ["sequence", "image"], f"data_type must be 'sequence' or 'image', got {data_type}"
        self.data_type = data_type

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Function to prepare the data."""
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SWEDataset(
                directory=self.data_dir,
                mode="train",
                rollout_steps=PRED_STEPS,
            )
            self.val_dataset = SWEDataset(
                directory=self.data_dir,
                mode="valid",
                rollout_steps=EVAL_ROLLOUT_STEPS,
            )
        if stage == "test" or stage is None:
            self.test_dataset = SWEDataset(
                directory=self.data_dir,
                mode="test",
                rollout_steps=EVAL_ROLLOUT_STEPS,
            )

    def _build_loader(self, dataset, shuffle: bool, drop_last: bool = False):
        """Function to create dataloaders given a dataset and a few arguments.

        Reused for train, val and test dataloaders.

        Args:
            dataset: Dataset to create a dataloader for.
            shuffle: Whether to shuffle the dataset.
            drop_last: Whether to drop the last batch if it's not complete.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            # worker_init_fn=self.worker_init_fn,  # No longer needed with pl.seed_everything(workers=True)
            generator=self.generator,
            persistent_workers=self.num_workers > 0,
        )

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        """Function to create the train dataloader."""
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        """Function to create the validation dataloader."""
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        """Function to create the test dataloader."""
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)