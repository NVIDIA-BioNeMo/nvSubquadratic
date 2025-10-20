# TODO: Add license header here

"""Mixin class for Context Parallelism-aware data loading.

This module provides utilities for dataloaders to properly handle data splitting
when Context Parallelism is enabled. Works in conjunction with DistributedSampler
that uses DP-only rank/world_size to ensure all CP ranks get the same data samples.
"""

import logging
from typing import Optional

import torch.distributed as dist
from torch.utils.data import DistributedSampler


logger = logging.getLogger(__name__)


class CPAwareDataMixin:
    """Mixin class for dataloaders that support Context Parallelism.

    This mixin provides CP splitting that happens after batch transfer to GPU,
    in the main process where distributed operations work.

    Key insight: Can't do distributed ops in DataLoader workers (collate_fn),
    must do them in main process after GPU transfer.

    Usage:
        class MNISTDataModule(pl.LightningDataModule, CPAwareDataMixin):
            def __init__(self, ..., enable_cp=False, seed=0, cp_seq_dim=1):
                super().__init__()
                self.enable_cp = enable_cp
                self.seed = seed
                self.cp_seq_dim = cp_seq_dim

            def _build_loader(self, dataset, shuffle, drop_last):
                # Create distributed sampler for CP (from mixin)
                sampler = self._create_distributed_sampler(dataset, shuffle, drop_last)
                if sampler is not None:
                    shuffle = False
                    drop_last = False
                return DataLoader(dataset, sampler=sampler, shuffle=shuffle, drop_last=drop_last, ...)

        # transfer_batch_to_device is provided by the mixin!
        # Override only if you need custom behavior.
    """

    def _create_distributed_sampler(
        self,
        dataset,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> Optional[DistributedSampler]:
        """Create DistributedSampler that's CP-aware using DP-only rank/world_size.

        When CP is enabled, this creates a DistributedSampler that uses only the
        Data Parallel (DP) rank and world size, ensuring all CP ranks within the
        same DP group receive identical data samples.

        When CP is not enabled, returns None (no sampler needed).

        Args:
            dataset: Dataset to create sampler for.
            shuffle: Whether to shuffle the dataset.
            drop_last: Whether to drop the last incomplete batch.
            seed: Random seed for shuffling. If None, uses self.seed.

        Returns:
            DistributedSampler if CP is enabled and backend is available, None otherwise.
            When a sampler is returned, the caller should set shuffle=False and
            drop_last=False in the DataLoader since these are handled by the sampler.
        """
        if not self.enable_cp:
            return None

        from nvsubquadratic.distributed.backend import get_global_backend

        backend = get_global_backend()

        if backend is None:
            return None

        # Use provided seed or fall back to self.seed
        if seed is None:
            seed = getattr(self, "seed", 0)

        return DistributedSampler(
            dataset,
            num_replicas=backend.get_data_parallel_world_size(),  # DP-only
            rank=backend.get_data_parallel_rank(),  # DP-only
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer batch to device and apply CP splitting if enabled.

        This method is called by PyTorch Lightning in the main process after
        the batch is on GPU, so distributed operations work correctly.

        This implementation provides the standard CP workflow:
        1. Transfer to device (standard Lightning behavior)
        2. Apply CP splitting if enabled

        Subclasses can override this method if they need custom behavior,
        but in most cases this default implementation should work.

        The sequence dimension for splitting is controlled by self.cp_seq_dim.

        Args:
            batch: Batch data to transfer.
            device: Target device.
            dataloader_idx: Index of the dataloader.

        Returns:
            Batch on device, with CP splitting applied if enabled.
        """
        # First transfer to device (standard Lightning behavior)
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        # Then apply CP splitting if enabled
        if self.enable_cp:
            batch = self._apply_cp_split_on_device(batch, device, seq_dim=self.cp_seq_dim)

        return batch

    def _apply_cp_split_on_device(self, batch, device, seq_dim: int = 1):
        """Apply CP splitting after batch is on device.

        This should be called from transfer_batch_to_device, after the batch
        is already on GPU, in the main process where distributed ops work.

        Note: With DistributedSampler using DP-only rank/world_size, all CP ranks
        within the same DP group already have the same data. We just need to split
        by CP rank directly - no broadcast needed!

        Args:
            batch: Batch data (already on device).
            device: Device batch is on.
            seq_dim: Dimension to split along.

        Returns:
            CP-split batch.
        """
        from nvsubquadratic.distributed.backend import get_global_backend

        backend = get_global_backend()

        if backend is None or backend.get_context_parallel_world_size() <= 1:
            return batch

        # Unpack
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            data = batch[0]
            labels = batch[1]
            rest = batch[2:] if len(batch) > 2 else ()
        else:
            logger.warning("Unexpected batch format for CP")
            return batch

        cp_group = backend.get_context_parallel_group()

        if dist.is_initialized() and cp_group is not None:
            # All CP ranks in the same DP group already have the same data
            # from DistributedSampler - just split by CP rank directly
            from nvsubquadratic.parallel.utils import zigzag_split_across_group_ranks

            data = zigzag_split_across_group_ranks(data, group=cp_group, seq_dim=seq_dim)

            logger.debug(
                f"CP split applied: "
                f"CP rank={backend.get_context_parallel_rank()}/{backend.get_context_parallel_world_size()}, "
                f"DP rank={backend.get_data_parallel_rank()}/{backend.get_data_parallel_world_size()}, "
                f"split shape={data.shape}"
            )

        # Repack
        if rest:
            return (data, labels, *rest)
        else:
            return (data, labels)
