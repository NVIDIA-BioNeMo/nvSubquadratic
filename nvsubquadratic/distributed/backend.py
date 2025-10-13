# TODO: Add license header here

"""Backend abstraction for distributed training.

This module provides an abstract interface for different distributed training backends,
allowing easy switching between Megatron-Core, PyTorch DeviceMesh, and future backends.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch.distributed as dist


logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel training.

    Attributes:
        backend_type: Which backend to use ("megatron" or "device_mesh").
        context_parallel_size: Number of devices for context/sequence parallelism.
        tensor_parallel_size: Number of devices for tensor parallelism.
        pipeline_parallel_size: Number of stages for pipeline parallelism (Megatron only).
        data_parallel_size: Number of data parallel replicas (auto-computed if -1).
    """

    backend_type: Literal["megatron", "device_mesh"] = "megatron"
    context_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = -1  # Auto-computed


class ParallelBackend(ABC):
    """Abstract interface for distributed parallelism backends.

    This class defines the interface that all backends must implement,
    ensuring consistent behavior across different distributed training
    frameworks.
    """

    def __init__(self, config: ParallelConfig):
        """Initialize the backend with configuration.

        Args:
            config: Parallel configuration specifying parallelism dimensions.
        """
        self.config = config
        self._initialized = False

    @abstractmethod
    def initialize(self, world_size: int, rank: int) -> None:
        """Initialize the distributed backend.

        Args:
            world_size: Total number of processes in the job.
            rank: Global rank of this process.

        Raises:
            RuntimeError: If initialization fails.
        """
        pass

    @abstractmethod
    def get_context_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get the context parallel process group.

        Returns:
            ProcessGroup for context parallelism, or None if CP is disabled.
        """
        pass

    @abstractmethod
    def get_data_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get the data parallel process group.

        This group is used for gradient synchronization (DDP/FSDP).

        Returns:
            ProcessGroup for data parallelism, or None if DP is disabled.
        """
        pass

    @abstractmethod
    def get_tensor_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get the tensor parallel process group.

        Returns:
            ProcessGroup for tensor parallelism, or None if TP is disabled.
        """
        pass

    @abstractmethod
    def get_context_parallel_rank(self) -> int:
        """Get rank within the context parallel group.

        Returns:
            Local rank within CP group (0 to CP size - 1).
        """
        pass

    @abstractmethod
    def get_context_parallel_world_size(self) -> int:
        """Get size of the context parallel group.

        Returns:
            Number of devices in CP group.
        """
        pass

    @abstractmethod
    def get_data_parallel_rank(self) -> int:
        """Get rank within the data parallel group.

        Returns:
            Local rank within DP group (0 to DP size - 1).
        """
        pass

    @abstractmethod
    def get_data_parallel_world_size(self) -> int:
        """Get size of the data parallel group.

        Returns:
            Number of devices in DP group.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, state_dict: Dict[str, Any], path: str) -> None:
        """Save checkpoint with backend-specific sharding.

        Args:
            state_dict: State dictionary to save.
            path: Path to save checkpoint to.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint with backend-specific resharding.

        Args:
            path: Path to load checkpoint from.

        Returns:
            Loaded state dictionary.
        """
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Cleanup backend resources.

        Should be called at the end of training to properly clean up
        process groups and other resources.
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized.

        Returns:
            True if backend is initialized and ready to use.
        """
        return self._initialized


class MegatronBackend(ParallelBackend):
    """Megatron-Core based parallelism backend.

    Provides support for:
    - Context Parallelism (CP): Sequence splitting
    - Tensor Parallelism (TP): Model layer splitting
    - Pipeline Parallelism (PP): Model stage splitting
    - Data Parallelism (DP): Gradient synchronization

    This backend uses Megatron-Core's parallel state management,
    which has been battle-tested on large-scale LLM training.
    """

    def initialize(self, world_size: int, rank: int) -> None:
        """Initialize Megatron parallel state.

        Args:
            world_size: Total number of processes.
            rank: Global rank of this process.

        Raises:
            AssertionError: If world size doesn't match parallelism configuration.
        """
        from nvsubquadratic.parallel.utils import init_parallel_state

        logger.info(
            f"Initializing Megatron backend: "
            f"world_size={world_size}, rank={rank}, "
            f"TP={self.config.tensor_parallel_size}, "
            f"PP={self.config.pipeline_parallel_size}, "
            f"CP={self.config.context_parallel_size}"
        )

        # Initialize Megatron's parallel state
        init_parallel_state(
            tensor_model_parallel_size=self.config.tensor_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
        )

        self._initialized = True

        logger.info(
            f"Megatron backend initialized successfully: "
            f"CP rank={self.get_context_parallel_rank()}/{self.get_context_parallel_world_size()}, "
            f"DP rank={self.get_data_parallel_rank()}/{self.get_data_parallel_world_size()}"
        )

    def get_context_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get CP group from Megatron parallel state."""
        from megatron.core import parallel_state

        return parallel_state.get_context_parallel_group()

    def get_data_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get DP group from Megatron parallel state.

        Note: Uses with_context_parallel=True to get the correct DP group
        that accounts for CP sharding.
        """
        from megatron.core import parallel_state

        return parallel_state.get_data_parallel_group(with_context_parallel=True)

    def get_tensor_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get TP group from Megatron parallel state."""
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_group()

    def get_context_parallel_rank(self) -> int:
        """Get CP rank from Megatron parallel state."""
        from megatron.core import parallel_state

        return parallel_state.get_context_parallel_rank()

    def get_context_parallel_world_size(self) -> int:
        """Get CP world size from Megatron parallel state."""
        from megatron.core import parallel_state

        return parallel_state.get_context_parallel_world_size()

    def get_data_parallel_rank(self) -> int:
        """Get DP rank from Megatron parallel state."""
        from megatron.core import parallel_state

        return parallel_state.get_data_parallel_rank()

    def get_data_parallel_world_size(self) -> int:
        """Get DP world size from Megatron parallel state."""
        from megatron.core import parallel_state

        return parallel_state.get_data_parallel_world_size()

    def save_checkpoint(self, state_dict: Dict[str, Any], path: str) -> None:
        """Save checkpoint using Megatron's distributed checkpoint.

        Args:
            state_dict: State dictionary to save.
            path: Directory path to save checkpoint to.
        """
        import os

        from megatron.core import dist_checkpointing

        # Create checkpoint directory if it doesn't exist
        # Only rank 0 creates to avoid race conditions
        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created checkpoint directory: {path}")

        # Wait for directory to be created
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"Saving Megatron distributed checkpoint to {path}")
        dist_checkpointing.save(state_dict, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint using Megatron's distributed checkpoint.

        Args:
            path: Directory path to load checkpoint from.

        Returns:
            Loaded state dictionary with proper resharding for current topology.
        """
        from megatron.core import dist_checkpointing

        logger.info(f"Loading Megatron distributed checkpoint from {path}")

        # Megatron's load() requires an empty state_dict structure to fill
        # We need to provide the structure that matches what was saved
        state_dict = {}

        # Load will populate this state_dict with the checkpoint data
        loaded_state = dist_checkpointing.load(state_dict, path)

        return loaded_state

    def destroy(self) -> None:
        """Cleanup Megatron parallel state."""
        from megatron.core import parallel_state

        logger.info("Destroying Megatron parallel state")
        parallel_state.destroy_model_parallel()
        self._initialized = False


class DeviceMeshBackend(ParallelBackend):
    """PyTorch DeviceMesh based parallelism backend (Future implementation).

    This backend will provide PyTorch-native N-D parallelism using DeviceMesh.
    Implementation planned for Phase 2.
    """

    def initialize(self, world_size: int, rank: int) -> None:
        """Initialize DeviceMesh backend."""
        raise NotImplementedError(
            "DeviceMesh backend not yet implemented. "
            "This will be added in Phase 2. "
            "Please use backend_type='megatron' for now."
        )

    def get_context_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get CP group (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def get_data_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get DP group (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def get_tensor_parallel_group(self) -> Optional[dist.ProcessGroup]:
        """Get TP group (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def get_context_parallel_rank(self) -> int:
        """Get CP rank (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def get_context_parallel_world_size(self) -> int:
        """Get CP world size (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def get_data_parallel_rank(self) -> int:
        """Get DP rank (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def get_data_parallel_world_size(self) -> int:
        """Get DP world size (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def save_checkpoint(self, state_dict: Dict[str, Any], path: str) -> None:
        """Save checkpoint (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")

    def destroy(self) -> None:
        """Cleanup resources (not yet implemented)."""
        raise NotImplementedError("DeviceMesh backend not yet implemented")


def create_backend(config: ParallelConfig) -> ParallelBackend:
    """Create a parallel backend instance based on configuration.

    Args:
        config: Parallel configuration specifying backend type and parallelism dimensions.

    Returns:
        Initialized backend instance (not yet initialized - call initialize() separately).

    Raises:
        ValueError: If backend_type is unknown.

    Example:
        >>> config = ParallelConfig(backend_type="megatron", context_parallel_size=2)
        >>> backend = create_backend(config)
        >>> backend.initialize(world_size=4, rank=0)
    """
    backends = {
        "megatron": MegatronBackend,
        "device_mesh": DeviceMeshBackend,
    }

    backend_type = config.backend_type.lower()
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Supported backends: {list(backends.keys())}")

    logger.info(f"Creating {backend_type} backend")
    return backends[backend_type](config)


# ============================================================================
# GLOBAL CONTEXT (for easy access throughout the codebase)
# ============================================================================

_global_backend: Optional[ParallelBackend] = None


def set_global_backend(backend: Optional[ParallelBackend]) -> None:
    """Set the global backend instance.

    This allows modules to easily access the backend without passing it around.

    Args:
        backend: Backend instance to set as global, or None to clear.
    """
    global _global_backend
    _global_backend = backend


def get_global_backend() -> Optional[ParallelBackend]:
    """Get the global backend instance.

    Returns:
        The global backend instance, or None if not set.
    """
    return _global_backend


def get_context_parallel_group() -> Optional[dist.ProcessGroup]:
    """Convenience function to get CP group from global backend.

    This is a shortcut for get_global_backend().get_context_parallel_group().

    Returns:
        CP process group, or None if backend is not initialized or CP is disabled.
    """
    backend = get_global_backend()
    return backend.get_context_parallel_group() if backend else None
