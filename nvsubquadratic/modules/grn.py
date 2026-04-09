"""Global Response Normalization (GRN) layer.

Promotes inter-channel feature competition via divisive normalization,
as proposed in ConvNeXt V2 (Woo et al., 2023, arXiv:2301.00808).
"""

import torch
import torch.nn as nn


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization.

    Aggregates spatial activations per channel (L2 norm), then applies
    divisive normalization across channels to promote feature diversity.
    Gamma and beta are zero-initialized so the layer starts as identity.

    Args:
        dim: Number of channels (last dimension of input).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize GRN with zero-initialized gamma and beta."""
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def flop_count(self, num_tokens: int) -> int:
        """Count FLOPs for GRN.

        Using T = num_tokens and C = dim:
        - L2 norm per channel over spatial dims: T*C (square) + T*C (sum) + C (sqrt) = 2*T*C + C
        - mean over channels: C (sum)
        - division: C (div)
        - apply to input: T*C (mult) + T*C (mult by gamma) + T*C (add beta) + T*C (residual)

        Total: ~6 * T * C FLOPs
        """
        return 6 * num_tokens * self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRN to input tensor.

        Args:
            x: Input of shape ``[B, *spatial, C]`` (channels-last).

        Returns:
            Tensor of same shape as input.
        """
        spatial_dims = tuple(range(1, x.ndim - 1))
        gx = torch.norm(x, p=2, dim=spatial_dims, keepdim=True)  # [B, 1..., C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)  # [B, 1..., C]
        return self.gamma * (x * nx) + self.beta + x
