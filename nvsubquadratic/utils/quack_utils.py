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

"""QuACK kernel availability utilities.

Shared helpers used by modules that optionally dispatch to QuACK fused kernels
(e.g. RMSNorm, MLP).
"""

import torch


def cuda_supports_quack(device: torch.device) -> bool:
    """Return ``True`` if ``device`` supports QuACK fused kernels.

    QuACK kernels require compute capability SM ≥ 9.0 (Hopper: H100;
    Blackwell: B200, B300).  On older architectures (e.g. Ampere A100,
    SM 8.0) the QuACK backward kernel is incompatible and must not be
    called; callers should fall back to the PyTorch reference path.

    Args:
        device: A ``torch.device`` of type ``"cuda"`` with a device index.
            Non-CUDA devices (CPU, MPS) immediately return ``False``.

    Returns:
        ``True`` if ``device`` is a CUDA device with SM major version ≥ 9,
        ``False`` otherwise.
    """
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9
