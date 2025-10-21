# TODO: Add license header here

"""Shared test utilities for distributed training validation."""

import json
from pathlib import Path
from typing import Dict, Optional

import torch


def compute_relative_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Compute relative error between two tensors: ||t1 - t2|| / ||t1||.

    This metric is scale-invariant and useful for comparing gradients or activations
    in distributed training. Following the TTrace methodology (arXiv:2506.09280),
    this helps distinguish between floating-point round-off errors and actual bugs
    in distributed implementations.

    Args:
        tensor1: Reference tensor
        tensor2: Tensor to compare against reference

    Returns:
        Relative error as a float scalar. Returns absolute difference if ||t1|| is near zero.

    Example:
        >>> grad_ref = torch.randn(100, 100)
        >>> grad_test = grad_ref + 0.001 * torch.randn(100, 100)  # Add 0.1% noise
        >>> rel_err = compute_relative_error(grad_ref, grad_test)
        >>> assert rel_err < 0.01  # Less than 1% relative error

    Reference:
        TTrace: Lightweight Error Checking and Diagnosis for Distributed Training
        https://arxiv.org/abs/2506.09280
    """
    # Cast to float32 for numerical stability
    diff_norm = torch.linalg.norm(tensor1.float() - tensor2.float())
    ref_norm = torch.linalg.norm(tensor1.float())

    # Avoid division by zero - if reference is ~0, return absolute difference
    if ref_norm < 1e-10:
        return diff_norm.item()

    return (diff_norm / ref_norm).item()


def load_gradient_stats(save_dir: Path, rank: int = 0, step: Optional[int] = None) -> Dict:
    """Load gradient statistics from disk.

    Args:
        save_dir: Directory containing gradient files
        rank: Rank to load gradients from (default: 0)
        step: Optional step number

    Returns:
        Dictionary of gradient statistics per parameter

    Raises:
        FileNotFoundError: If no gradient files are found
    """
    save_dir = Path(save_dir)

    # Build filename
    if step is not None:
        filename = f"gradients_rank{rank}_step{step}.json"
    else:
        filename = f"gradients_rank{rank}.json"

    save_path = save_dir / filename

    if not save_path.exists():
        raise FileNotFoundError(f"Gradient file not found: {save_path}")

    with open(save_path, "r") as f:
        return json.load(f)
