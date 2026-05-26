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
