# TODO: Add license header here

"""Distributed training backend abstraction for nvSubquadratic.

This module provides a backend-agnostic interface for distributed training,
supporting multiple parallelism strategies:
- Context Parallelism (CP): Sequence splitting across devices
- Data Parallelism (DP): Gradient synchronization
- Tensor Parallelism (TP): Model layer splitting
- Pipeline Parallelism (PP): Model stage splitting

Currently supported backends:
- Megatron-Core: Full support for CP/TP/PP/DP
- DeviceMesh: (TODO) PyTorch-native N-D parallelism
"""

from .backend import (
    MegatronBackend,
    ParallelBackend,
    ParallelConfig,
    create_backend,
    get_context_parallel_group,
    get_global_backend,
    set_global_backend,
)


__all__ = [
    "MegatronBackend",
    "ParallelBackend",
    "ParallelConfig",
    "create_backend",
    "get_context_parallel_group",
    "get_global_backend",
    "set_global_backend",
]
