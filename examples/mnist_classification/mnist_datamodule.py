# TODO: Add license header here

"""MNIST datamodule with Context Parallelism support."""

from typing import Literal

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from nvsubquadratic.datamodules import CPAwareDataMixin


# # Global seed value used for worker initialization
# _BASE_SEED = 0


# def set_base_seed(seed):
#     """Set the base seed for worker initialization."""
#     global _BASE_SEED
#     _BASE_SEED = seed


# # Define a worker initialization function to set seeds for data loading workers
# def deterministic_worker_init_fn(worker_id: int):
#     """Initialize the worker with a deterministic seed derived from base_seed and worker_id.

#     Each worker gets a unique but deterministic seed: base_seed + worker_id
#     """
#     # Use the global base seed plus worker_id as the seed for this worker
#     global _BASE_SEED
#     seed = _BASE_SEED + worker_id

#     # Set Python hash seed for this process
#     os.environ["PYTHONHASHSEED"] = str(seed)

#     # Set all relevant random states with this seed
#     random.seed(seed)  # Set Python's random seed
#     np.random.seed(seed)  # Set NumPy's random seed
#     torch.manual_seed(seed)  # Set PyTorch's CPU RNG seed
#     torch.cuda.manual_seed(seed)  # Set CUDA RNG seed for current device
#     torch.cuda.manual_seed_all(seed)  # Set CUDA RNG seed for all devices


class MNISTDataModule(pl.LightningDataModule, CPAwareDataMixin):
    """MNIST Lightning data module with Context Parallelism support."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_type: Literal["sequence", "image"],
        num_workers: int,
        pin_memory: bool,
        use_deterministic_worker_init: bool,
        seed: int,
        enable_cp: bool = False,
        cp_seq_dim: int = 1,
    ):
        """Initialize the MNISTDataModule.

        Args:
            data_dir: Directory to save the data
            batch_size: Batch size
            data_type: Type of data. Can be "sequence" or "image".
            num_workers: Number of workers
            pin_memory: Whether to pin memory
            use_deterministic_worker_init: Whether to use deterministic worker initialization
            seed: Seed for the data
            enable_cp: Whether to enable Context Parallelism data splitting
            cp_seq_dim: Dimension to split along for CP
        """
        super().__init__()

        # Save parameters to self
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.enable_cp = enable_cp
        self.cp_seq_dim = cp_seq_dim
        # Create a generator with the given seed for reproducibility
        self.generator = torch.Generator().manual_seed(seed)

        # Handle worker initialization. Use deterministic worker initialization if specified.
        # self.worker_init_fn = deterministic_worker_init_fn if use_deterministic_worker_init else None

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 10

        # Assert that data_type is in the allowed options
        assert data_type in ["sequence", "image"], f"data_type must be 'sequence' or 'image', got {data_type}"
        self.data_type = data_type

        # Create transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Function to prepare the data."""
        # download data, train then test
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Function to setup the datamodule."""
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
        # Create distributed sampler for CP (from mixin)
        # This ensures all CP ranks in the same DP group get the same data samples
        sampler = self._create_distributed_sampler(dataset, shuffle, drop_last)
        if sampler is not None:
            # When using sampler, disable shuffle and drop_last in DataLoader
            shuffle = False
            drop_last = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            generator=self.generator if sampler is None else None,
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

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Function to rearrange the input.

        For image data_type, from [B, C, Y, X] to [B, Y, X, C].
        For sequence data_type, from [B, C, T] to [B, T, C].
        """
        x, y = batch
        if self.data_type == "image":
            x = rearrange(x, "b c y x -> b y x c")
        elif self.data_type == "sequence":
            x = rearrange(x, "b c y x -> b (y x) c")
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        batch = x, y
        return batch
