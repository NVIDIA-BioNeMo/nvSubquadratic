"""QuACK kernel availability utilities.

Shared helpers used by modules that optionally dispatch to QuACK fused kernels
(e.g. RMSNorm, MLP).
"""

import torch


def cuda_supports_quack(device: torch.device) -> bool:
    """Return True if *device* supports QuACK kernels (Hopper/Blackwell, SM >= 9)."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9
