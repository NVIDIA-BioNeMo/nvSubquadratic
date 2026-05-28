# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Add license header here

"""Shared test utilities for distributed training validation."""

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
