"""Register classification heads.

RegisterReductionHead: implements the register recycling head from
"Mamba-R: Vision Mamba ALSO Needs Registers" (arXiv:2405.14858).
Given n d-dimensional register tokens, a linear layer reduces each to d/r,
then all reduced tokens are concatenated and projected to num_classes.

RegisterGAPHead: simple global average pooling over register tokens,
followed by a single linear projection to num_classes. Ablation baseline.
"""

import torch
import torch.nn as nn


class RegisterReductionHead(nn.Module):
    """Mamba-R register reduction head.

    Condenses n register tokens of dimension d into class logits via:
    1. Linear reduction: d → d // reduction_factor per token
    2. Concatenation: [B, n, d/r] → [B, n * d/r]
    3. Linear projection: n * d/r → num_classes

    Concatenation (rather than averaging) is motivated by multi-head attention,
    where each register can specialize in different aspects of the representation.

    Args:
        hidden_dim: Token embedding dimension d.
        num_registers: Number of register tokens n.
        reduction_factor: Dimensionality reduction factor r (d must be divisible by r).
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_registers: int,
        reduction_factor: int,
        num_classes: int,
    ):
        super().__init__()
        assert hidden_dim % reduction_factor == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by reduction_factor ({reduction_factor})"
        )
        reduced_dim = hidden_dim // reduction_factor
        self.reduce = nn.Linear(hidden_dim, reduced_dim)
        self.proj = nn.Linear(num_registers * reduced_dim, num_classes, bias=False)

        nn.init.trunc_normal_(self.reduce.weight, std=0.02)
        nn.init.zeros_(self.reduce.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, registers: torch.Tensor) -> torch.Tensor:
        """Reduce and project register tokens to class logits.

        Args:
            registers: [B, num_registers, hidden_dim]

        Returns:
            logits: [B, num_classes]
        """
        out = self.reduce(registers)  # [B, R, d/r]
        out = out.flatten(1)          # [B, R * d/r]
        return self.proj(out)         # [B, num_classes]

    def extra_repr(self) -> str:
        in_features = self.reduce.in_features
        reduced = self.reduce.out_features
        n = self.proj.in_features // reduced
        return f"n={n}, d={in_features}, r={in_features // reduced}, num_classes={self.proj.out_features}"


class RegisterGAPHead(nn.Module):
    """Global average pooling over register tokens → linear classification head.

    Averages the R register token representations and projects to class logits.
    Ablation baseline for RegisterReductionHead.

    Args:
        hidden_dim: Token embedding dimension d.
        num_classes: Number of output classes.
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, num_classes, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, registers: torch.Tensor) -> torch.Tensor:
        """Average register tokens and project to logits.

        Args:
            registers: [B, num_registers, hidden_dim]

        Returns:
            logits: [B, num_classes]
        """
        out = registers.mean(dim=1)  # [B, hidden_dim]
        return self.proj(out)        # [B, num_classes]

    def extra_repr(self) -> str:
        return f"d={self.proj.in_features}, num_classes={self.proj.out_features}"
