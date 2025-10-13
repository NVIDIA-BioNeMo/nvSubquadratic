# TODO: Add license header here

"""Mixin class for Context Parallelism-aware data loading.

This module provides utilities for dataloaders to properly handle data splitting
when Context Parallelism is enabled, using a custom collate function.
"""

import logging

import torch.distributed as dist


logger = logging.getLogger(__name__)


class CPAwareDataMixin:
    """Mixin class for dataloaders that support Context Parallelism.

    This mixin provides CP splitting that happens after batch transfer to GPU,
    in the main process where distributed operations work.

    Key insight: Can't do distributed ops in DataLoader workers (collate_fn),
    must do them in main process after GPU transfer.

    Usage:
        class MNISTDataModule(pl.LightningDataModule, CPAwareDataMixin):
            def __init__(self, ..., enable_cp=False):
                super().__init__()
                self.enable_cp = enable_cp

            def transfer_batch_to_device(self, batch, device, dataloader_idx):
                # Transfer to GPU first (standard Lightning)
                batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

                # Then apply CP splitting (from mixin)
                if self.enable_cp:
                    batch = self._apply_cp_split_on_device(batch, device, seq_dim=1)

                return batch
    """

    def _apply_cp_split_on_device(self, batch, device, seq_dim: int = 1):
        """Apply CP splitting after batch is on device.

        This should be called from transfer_batch_to_device, after the batch
        is already on GPU, in the main process where distributed ops work.

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
            # Get source rank
            cp_group_ranks = dist.get_process_group_ranks(cp_group)
            source_rank = min(cp_group_ranks)

            # Broadcast to sync across CP ranks
            dist.broadcast(data, src=source_rank, group=cp_group)
            dist.broadcast(labels, src=source_rank, group=cp_group)

            # Zigzag split
            from nvsubquadratic.parallel.utils import zigzag_split_across_group_ranks

            data = zigzag_split_across_group_ranks(data, group=cp_group, seq_dim=seq_dim)

            logger.debug(
                f"CP split applied: "
                f"rank={backend.get_context_parallel_rank()}/{backend.get_context_parallel_world_size()}, "
                f"split shape={data.shape}"
            )

        # Repack
        if rest:
            return (data, labels, *rest)
        else:
            return (data, labels)
