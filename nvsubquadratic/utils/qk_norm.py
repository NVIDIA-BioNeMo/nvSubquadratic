# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""QK normalization utilities."""

import torch


def apply_qk_norm(query: torch.Tensor, key: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """L2-normalize queries and keys along the specified dimension.

    Args:
        query: torch.Tensor - The query tensor of shape (batch_size, hidden_dim, * spatial_dims) or (batch_size, * spatial_dims, hidden_dim).
        key: torch.Tensor - The key tensor of shape (batch_size, hidden_dim, * spatial_dims) or (batch_size, * spatial_dims, hidden_dim).
        dim: int - The dimension along which to normalize the query and key. This should be the hidden dimension.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The normalized query and key of corresponding shape.

    """
    query = torch.nn.functional.normalize(query, p=2.0, dim=dim)
    key = torch.nn.functional.normalize(key, p=2.0, dim=dim)
    return query, key
