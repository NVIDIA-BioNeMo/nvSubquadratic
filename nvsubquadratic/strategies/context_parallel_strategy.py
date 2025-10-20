# TODO: Add license header here

"""PyTorch Lightning strategy for Context Parallelism.

This strategy enables Context Parallelism (sequence splitting) with DDP gradient synchronization,
supporting multiple backend implementations (currently Megatron-Core).
"""

import logging
from typing import Literal, Optional

import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from nvsubquadratic.distributed.backend import ParallelBackend, ParallelConfig, create_backend, set_global_backend


logger = logging.getLogger(__name__)


class ContextParallelStrategy(DDPStrategy):
    """PyTorch Lightning strategy for Context Parallelism with DDP.

    This strategy extends Lightning's DDPStrategy to support Context Parallelism,
    where sequences are split across multiple devices while gradients are synchronized
    across data parallel replicas.

    Key features:
    - Backend-agnostic: Supports Megatron-Core and future backends
    - Automatic process group management for CP and DP
    - Optional distributed checkpointing
    - Seamless integration with Lightning's training loop

    Example:
        >>> from examples.strategies import ContextParallelStrategy
        >>> strategy = ContextParallelStrategy(
        ...     backend_type="megatron",
        ...     context_parallel_size=2,
        ... )
        >>> trainer = pl.Trainer(strategy=strategy, devices=4)
        >>> # Result: 2 DP replicas x 2 CP devices = 4 GPUs

    Args:
        backend_type: Backend to use ("megatron" or "device_mesh").
        context_parallel_size: Number of devices for context parallelism.
        tensor_parallel_size: Number of devices for tensor parallelism (future).
        pipeline_parallel_size: Number of pipeline stages (Megatron only, future).
        use_distributed_checkpoint: Whether to use distributed checkpointing.
        checkpoint_dir: Directory for distributed checkpoints.
        **ddp_kwargs: Additional arguments passed to DDPStrategy.
    """

    def __init__(
        self,
        backend_type: Literal["megatron", "device_mesh"] = "megatron",
        context_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        use_distributed_checkpoint: bool = False,
        checkpoint_dir: str = "./checkpoints",
        **ddp_kwargs,
    ):
        """Initialize Context Parallel strategy."""
        # Create parallel configuration
        self.parallel_config = ParallelConfig(
            backend_type=backend_type,
            context_parallel_size=context_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

        self.use_distributed_checkpoint = use_distributed_checkpoint
        self.checkpoint_dir = checkpoint_dir

        # Backend will be created during setup
        self.backend: Optional[ParallelBackend] = None
        self.cp_group: Optional[torch.distributed.ProcessGroup] = None

        super().__init__(**ddp_kwargs)

        logger.info(f"ContextParallelStrategy created with backend={backend_type}, CP size={context_parallel_size}")

    def setup(self, trainer) -> None:
        """Initialize backend and setup distributed training."""
        super().setup(trainer)

        # Only initialize backend if CP is enabled
        if self.parallel_config.context_parallel_size > 1:
            logger.info(
                f"Initializing {self.parallel_config.backend_type} backend "
                f"with CP size={self.parallel_config.context_parallel_size}"
            )

            # Create and initialize backend
            self.backend = create_backend(self.parallel_config)
            self.backend.initialize(
                world_size=self.world_size,
                rank=self.global_rank,
            )

            # Store backend globally for easy access
            set_global_backend(self.backend)

            # Get CP group
            self.cp_group = self.backend.get_context_parallel_group()

            # Inject CP group into Lightning module
            if hasattr(trainer.lightning_module, "set_context_parallel_group"):
                trainer.lightning_module.set_context_parallel_group(self.cp_group)
                logger.info("Injected CP group into Lightning module")

            logger.info(
                f"Context Parallel strategy initialized: "
                f"CP size={self.backend.get_context_parallel_world_size()}, "
                f"DP size={self.backend.get_data_parallel_world_size()}, "
                f"CP rank={self.backend.get_context_parallel_rank()}, "
                f"DP rank={self.backend.get_data_parallel_rank()}"
            )
        else:
            logger.info("Context Parallelism disabled (CP size=1), using standard DDP")

    def _setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup model with DDP on correct process group.

        When Context Parallelism is enabled, DDP wraps the model using the
        data parallel process group, which excludes context parallel devices
        (they share sequence chunks, not full batches).

        Args:
            model: The model to wrap with DDP.

        Returns:
            DDP-wrapped model configured for the correct process group.
        """
        # Get the DP+CP group from backend for gradient synchronization
        dp_group = None
        if self.backend is not None:
            dp_group = self.backend.get_data_context_parallel_group()
            logger.info("Wrapping model with DDP using data+context parallel group for gradient sync")
        else:
            logger.info("Wrapping model with DDP using default process group")

        return DDP(
            model,
            device_ids=[self.local_rank] if self.local_rank is not None else None,
            process_group=dp_group,
            find_unused_parameters=True,  # More forgiving during development
        )

    def teardown(self) -> None:
        """Cleanup backend resources."""
        if self.backend is not None:
            logger.info("Cleaning up distributed backend")
            self.backend.destroy()
            set_global_backend(None)

        super().teardown()

    @property
    def is_context_parallel_enabled(self) -> bool:
        """Check if context parallelism is enabled.

        Returns:
            True if CP is enabled (CP size > 1).
        """
        return self.parallel_config.context_parallel_size > 1

    @property
    def context_parallel_size(self) -> int:
        """Get the context parallel size.

        Returns:
            Number of devices in context parallel group.
        """
        if self.backend is not None:
            return self.backend.get_context_parallel_world_size()
        return 1

    @property
    def data_parallel_size(self) -> int:
        """Get the data parallel size.

        Returns:
            Number of data parallel replicas.
        """
        if self.backend is not None:
            return self.backend.get_data_parallel_world_size()
        return self.world_size
