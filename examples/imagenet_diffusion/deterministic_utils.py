# TODO: Add license header here


"""Utility helpers for deterministic dataloader seeding."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

_BASE_SEED: Optional[int] = None


def set_base_seed(seed: int) -> None:
    """Record a deterministic base seed used by dataloader workers."""
    global _BASE_SEED
    _BASE_SEED = int(seed)


def _resolve_seed(worker_id: int) -> int:
    """Combine the recorded base seed with the worker id or fall back to PyTorch's seed."""
    if _BASE_SEED is None:
        return torch.initial_seed() % np.iinfo(np.uint32).max
    return (_BASE_SEED + worker_id) % np.iinfo(np.uint32).max


def worker_init_fn(worker_id: int) -> None:
    """Initialize RNG state for dataloader workers in a reproducible way."""
    seed = _resolve_seed(worker_id)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
