"""DataModule for WELL benchmark datasets."""

import pytorch_lightning as pl
import torch
from the_well.data import WellDataModule as BaseWellDataModule


class _DownsampledDataLoader:
    """Wraps a dataloader to apply spatial downsampling on-the-fly."""

    def __init__(self, loader, downsample_fn):
        self.loader = loader
        self.downsample_fn = downsample_fn

    def __iter__(self):
        for batch in self.loader:
            yield self.downsample_fn(batch)

    def __len__(self):
        return len(self.loader)


class WellDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapper for WELL benchmark datasets.

    This wrapper provides a unified interface compatible with the nvSubquadratic
    training infrastructure while using the WELL benchmark's data loading.

    Args:
        well_base_path: Path to the WELL datasets directory
        well_dataset_name: Name of the dataset (e.g., 'active_matter')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        use_normalization: Whether to use normalization
        n_steps_input: Number of input timesteps
        n_steps_output: Number of output timesteps (for training)
        max_rollout_steps: Maximum number of rollout steps for validation
        min_dt_stride: Minimum time stride
        max_dt_stride: Maximum time stride (for data augmentation during training)
        seed: Random seed for deterministic behavior
        use_deterministic_worker_init: Whether to use deterministic worker initialization
        prefetch_factor: Number of batches to prefetch
        spatial_downsample_factor: Factor for spatial downsampling (e.g., 4 means take every 4th point)
    """

    def __init__(
        self,
        well_base_path: str,
        well_dataset_name: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_normalization: bool = True,
        n_steps_input: int = 4,
        n_steps_output: int = 1,
        max_rollout_steps: int = 32,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        seed: int = 0,
        use_deterministic_worker_init: bool = False,
        prefetch_factor: int = 2,
        spatial_downsample_factor: int = 1,
    ):
        super().__init__()
        self.well_base_path = well_base_path
        self.well_dataset_name = well_dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_normalization = use_normalization
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.max_rollout_steps = max_rollout_steps
        self.min_dt_stride = min_dt_stride
        self.max_dt_stride = max_dt_stride
        self.seed = seed
        self.use_deterministic_worker_init = use_deterministic_worker_init
        self.prefetch_factor = prefetch_factor
        self.spatial_downsample_factor = spatial_downsample_factor

        self._well_datamodule = None

    def prepare_data(self):
        """Download or prepare data (called on single process)."""
        # WELL datasets are assumed to be already downloaded
        pass

    def setup(self, stage=None):
        """Setup datasets (called on each process)."""
        from the_well.data.normalization import ZScoreNormalization

        # Create the WELL datamodule
        self._well_datamodule = BaseWellDataModule(
            well_base_path=self.well_base_path,
            well_dataset_name=self.well_dataset_name,
            batch_size=self.batch_size,
            use_normalization=self.use_normalization,
            normalization_type=ZScoreNormalization if self.use_normalization else None,
            n_steps_input=self.n_steps_input,
            n_steps_output=self.n_steps_output,
            max_rollout_steps=self.max_rollout_steps,
            min_dt_stride=self.min_dt_stride,
            max_dt_stride=self.max_dt_stride,
            data_workers=self.num_workers,
        )

        # BUGFIX: The WELL library doesn't apply normalization to val/test datasets
        # We need to manually set it using the train dataset's normalization
        if self.use_normalization and hasattr(self._well_datamodule.train_dataset, "norm"):
            train_normalization = self._well_datamodule.train_dataset.norm

            # Apply same normalization to val dataset
            self._well_datamodule.val_dataset.use_normalization = True
            self._well_datamodule.val_dataset.norm = train_normalization

            # Apply same normalization to rollout val dataset
            self._well_datamodule.rollout_val_dataset.use_normalization = True
            self._well_datamodule.rollout_val_dataset.norm = train_normalization

            # Apply same normalization to test dataset
            self._well_datamodule.test_dataset.use_normalization = True
            self._well_datamodule.test_dataset.norm = train_normalization

            # Apply same normalization to rollout test dataset
            self._well_datamodule.rollout_test_dataset.use_normalization = True
            self._well_datamodule.rollout_test_dataset.norm = train_normalization

        # Get metadata from the training dataset
        metadata = self._well_datamodule.train_dataset.metadata

        # Calculate input and output channels
        # Input: n_steps_input timesteps × n_fields + constant fields
        # Output: n_fields (single timestep prediction)
        self._input_channels = self.n_steps_input * metadata.n_fields + metadata.n_constant_fields
        self._output_channels = metadata.n_fields

        # Store metadata for use in model/wrapper
        self.metadata = metadata

    def _downsample_batch(self, batch):
        """Apply spatial downsampling to a batch.

        Args:
            batch: Dict with tensor fields in [B, T, H, W, C] format

        Returns:
            Downsampled batch
        """
        if self.spatial_downsample_factor == 1:
            return batch

        stride = self.spatial_downsample_factor
        downsampled_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim >= 4 and key in ["input_fields", "output_fields", "constant_fields", "space_grid"]:
                    if key in ["input_fields", "output_fields"]:
                        if value.ndim == 5:  # 2D: [B, T, H, W, C]
                            downsampled_batch[key] = value[:, :, ::stride, ::stride, :]
                        elif value.ndim == 6:  # 3D: [B, T, D, H, W, C]
                            downsampled_batch[key] = value[:, :, ::stride, ::stride, ::stride, :]
                        else:
                            downsampled_batch[key] = value
                    elif key == "constant_fields":
                        if value.ndim == 4:  # 2D: [B, H, W, C]
                            downsampled_batch[key] = value[:, ::stride, ::stride, :]
                        elif value.ndim == 5:  # 3D: [B, D, H, W, C]
                            downsampled_batch[key] = value[:, ::stride, ::stride, ::stride, :]
                        else:
                            downsampled_batch[key] = value
                    elif key == "space_grid":
                        if value.ndim == 4:  # 2D: [B, H, W, D]
                            downsampled_batch[key] = value[:, ::stride, ::stride, :]
                        elif value.ndim == 5:  # 3D: [B, D, H, W, D]
                            downsampled_batch[key] = value[:, ::stride, ::stride, ::stride, :]
                        else:
                            downsampled_batch[key] = value
                    else:
                        downsampled_batch[key] = value
                else:
                    downsampled_batch[key] = value
            else:
                downsampled_batch[key] = value

        return downsampled_batch

    def _wrap_loader(self, base_loader):
        """Wrap a dataloader with spatial downsampling if needed."""
        if self.spatial_downsample_factor == 1:
            return base_loader
        return _DownsampledDataLoader(base_loader, self._downsample_batch)

    def train_dataloader(self):
        return self._wrap_loader(self._well_datamodule.train_dataloader())

    def val_dataloader(self):
        return self._wrap_loader(self._well_datamodule.val_dataloader())

    def test_dataloader(self):
        return self._wrap_loader(self._well_datamodule.test_dataloader())

    @property
    def input_channels(self):
        """Number of input channels for the model."""
        if self._well_datamodule is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")
        return self._input_channels

    @property
    def output_channels(self):
        """Number of output channels for the model."""
        if self._well_datamodule is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")
        return self._output_channels

    @property
    def rollout_val_dataloader(self):
        """Return long rollout validation dataloader."""
        return self._well_datamodule.rollout_val_dataloader

    @property
    def rollout_test_dataloader(self):
        """Return long rollout test dataloader."""
        return self._well_datamodule.rollout_test_dataloader
