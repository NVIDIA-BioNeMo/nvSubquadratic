"""Shared helpers for detecting channel-first normalization modules."""

import torch.nn as nn


def is_channels_first_norm(module: nn.Module) -> bool:
    """Return True if *module* normalizes over dim=1 and accepts [B, C, *spatial] directly.

    Covers ``nn.GroupNorm``, ``RMSNormChannelFirst``, and any future norm that
    sets the ``channels_first`` class/instance attribute to ``True``.
    """
    if isinstance(module, nn.GroupNorm):
        return True
    return getattr(module, "channels_first", False)
