# TODO: Add license header here

"""Mixin class for Context Parallelism-aware data loading.

This module provides utilities for dataloaders to properly handle data splitting
when Context Parallelism is enabled.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


class CPAwareDataMixin:
    """Mixin class for dataloaders that support Context Parallelism.

    This mixin provides methods for handling data splitting across context parallel
    devices. When CP is enabled, each CP rank receives the same batch but with
    different sequence chunks (using zigzag splitting pattern).

    Key concept:
    - All CP ranks see the SAME batch (same data samples)
    - Each CP rank gets a DIFFERENT portion of the sequence
    - Labels are NOT split (all ranks need full labels)

    Usage:
        class MNISTDataModule(pl.LightningDataModule, CPAwareDataMixin):
            def __init__(self, ..., enable_cp=False):
                super().__init__()
                self.enable_cp = enable_cp

            def train_dataloader(self):
                return DataLoader(
                    ...,
                    collate_fn=self.cp_aware_collate if self.enable_cp else None,
                )
    """

    def cp_aware_collate(
        self,
        batch: list,
        seq_dim: int = 1,
        default_collate_fn: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function that handles Context Parallelism data splitting.

        This function:
        1. Collates the batch using default_collate_fn (or torch.utils.data.default_collate)
        2. Gets the CP group from the global backend
        3. Broadcasts batch across CP group so all ranks see the same data
        4. Splits sequences using zigzag pattern across CP devices
        5. Keeps labels unsplit (all ranks need full labels)

        Args:
            batch: List of samples from the dataset.
            seq_dim: Dimension along which to split sequences (default: 1).
            default_collate_fn: Optional custom collate function. If None, uses torch default.

        Returns:
            Tuple of (split_data, labels) where:
            - split_data: [batch_size, seq_len/CP_size, ...] (zigzag split)
            - labels: [batch_size, ...] (unsplit)

        Note:
            If CP is not enabled (cp_group is None or size=1), this behaves
            exactly like the default collate function.
        """
        # Use default collate if no custom one provided
        if default_collate_fn is None:
            collate_fn = torch.utils.data.default_collate
        else:
            collate_fn = default_collate_fn

        # Standard collate to create batched tensors
        batch_data = collate_fn(batch)

        # Unpack data and labels
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            data, labels = batch_data
        else:
            # No labels, just return data
            logger.warning("Batch does not contain (data, labels) tuple. CP splitting may not work correctly.")
            return batch_data

        # Get CP group from global backend
        from nvsubquadratic.distributed.backend import get_global_backend

        backend = get_global_backend()

        # If no backend or CP is disabled, return as-is
        if backend is None or backend.get_context_parallel_world_size() <= 1:
            return data, labels

        cp_group = backend.get_context_parallel_group()
        cp_rank = backend.get_context_parallel_rank()
        cp_size = backend.get_context_parallel_world_size()

        logger.debug(
            f"CP-aware collate: CP rank={cp_rank}/{cp_size}, data shape before split={data.shape}, seq_dim={seq_dim}"
        )

        # Broadcast data and labels across CP group to ensure all ranks see same batch
        # This is critical: CP ranks must process the SAME samples, just different sequence chunks
        if dist.is_initialized() and cp_group is not None:
            # Get the source rank (first rank in CP group)
            cp_group_ranks = dist.get_process_group_ranks(cp_group)
            source_rank = min(cp_group_ranks)

            # Broadcast data and labels
            dist.broadcast(data, src=source_rank, group=cp_group)
            dist.broadcast(labels, src=source_rank, group=cp_group)

            logger.debug(f"Broadcasted batch from rank {source_rank} to CP group")

            # Split data along sequence dimension using zigzag pattern
            from nvsubquadratic.parallel.utils import zigzag_split_across_group_ranks

            data_split = zigzag_split_across_group_ranks(data, group=cp_group, seq_dim=seq_dim)

            logger.debug(
                f"Split data using zigzag: shape after split={data_split.shape} "
                f"(expected seq_len={data.shape[seq_dim] // cp_size})"
            )

            # Labels are NOT split - all ranks need full labels for loss computation
            return data_split, labels
        else:
            # Not in distributed mode or CP not initialized
            return data, labels

    def cp_aware_collate_1d(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """CP-aware collate for 1D sequences (e.g., MNIST flattened to [B, L, C]).

        Convenience wrapper that sets seq_dim=1.

        Args:
            batch: List of samples from the dataset.

        Returns:
            Tuple of (split_data, labels).
        """
        return self.cp_aware_collate(batch, seq_dim=1)

    def cp_aware_collate_2d(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """CP-aware collate for 2D data (e.g., images as [B, H, W, C]).

        For 2D data, we typically split along the height dimension (seq_dim=1).

        Args:
            batch: List of samples from the dataset.

        Returns:
            Tuple of (split_data, labels).
        """
        return self.cp_aware_collate(batch, seq_dim=1)

    def cp_aware_collate_3d(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """CP-aware collate for 3D data (e.g., videos as [B, T, H, W, C]).

        For 3D data, we typically split along the temporal dimension (seq_dim=1).

        Args:
            batch: List of samples from the dataset.

        Returns:
            Tuple of (split_data, labels).
        """
        return self.cp_aware_collate(batch, seq_dim=1)
