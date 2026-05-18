"""Visual Mamba (Vim) baseline for ARC-AGI.

Adapts the Vision Mamba architecture ("Vision Mamba: Efficient Visual Representation
Learning with Bidirectional State Space Models", Zhu et al. 2024) to the ARC-AGI
dense-prediction task.

Architecture:
    colour embedding → patch embedding → [task token | patch tokens] →
    absolute positional embed → Mamba SSM blocks (bidirectional) →
    pixel-shuffle head → [B, num_colors, H, W]

The Mamba SSM is implemented in pure PyTorch so it requires no external
``mamba_ssm`` package.  A sequential selective-scan loop is used, which is
correct but slower than fused CUDA kernels; this is intentional for a
self-contained baseline.

Bidirectionality follows the Vim ``if_bidirectional=True`` strategy: each pair
of layers processes the sequence forward and backward independently, then the
outputs are summed before the next pair.

Input / Output contract (same as ARCViT):
    input:  ``{"input": [B, H, W] int, "condition": {"task_id": [B]}}``
    output: ``{"logits": [B, num_colors, H, W]}``
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed
from torch import nn


# ── Mamba SSM block ───────────────────────────────────────────────────────────


class MambaBlock(nn.Module):
    """Single Mamba SSM block (pure-PyTorch selective scan).

    Implements the Mamba layer from "Mamba: Linear-Time Sequence Modeling
    with Selective State Spaces" (Gu & Dao, 2023) without external CUDA deps.

    Args:
        d_model: Input / output feature dimension.
        d_state: SSM state size (N in the paper).  Default 16.
        d_conv: Width of the 1-D depthwise conv over the sequence.  Default 4.
        expand: Channel expansion factor (E in the paper).  Default 2.
        dt_rank: Rank of the Δ projection.  ``"auto"`` → ``ceil(d_model / 16)``.
        dt_min: Minimum Δ value after softplus.
        dt_max: Maximum Δ value after softplus.
        dt_scale: Scale applied to the dt_proj bias init.
        dt_init_floor: Floor for random dt init.
        bias: Whether in_proj / out_proj have biases.
        conv_bias: Whether conv1d has a bias.
    """

    def __init__(  # noqa: D107
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: x → [z, x_inner]
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # Short depthwise conv over the sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # Selective SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Δ projection initialisation following the Mamba paper
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus inverse
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

        # A: [d_inner, d_state], initialised as log(-arange)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    # ------------------------------------------------------------------
    # Selective scan (pure PyTorch, sequential over L)
    # ------------------------------------------------------------------

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential selective scan.

        Args:
            u:     [B, d_inner, L]
            delta: [B, d_inner, L]
            A:     [d_inner, d_state]
            B:     [B, d_state, L]
            C:     [B, d_state, L]
            D:     [d_inner]

        Returns:
            y: [B, d_inner, L]
        """
        batch, d_in, seq_len = u.shape
        n = A.shape[1]

        # Discretise A and B*u at every timestep
        # deltaA: [B, d_in, L, n]
        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        # deltaB_u: [B, d_in, L, n]
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

        # Sequential scan over L
        x = u.new_zeros(batch, d_in, n)
        ys = []
        for i in range(seq_len):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]  # [B, d_in, n]
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])  # [B, d_in]
            ys.append(y)
        y = torch.stack(ys, dim=2)  # [B, d_in, L]
        y = y + D[None, :, None] * u
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mamba block.

        Args:
            x: [B, L, d_model]

        Returns:
            out: [B, L, d_model]
        """
        _B, L, _ = x.shape

        # Input projection → z (gate) and x_inner (SSM input)
        xz = self.in_proj(x)  # [B, L, 2 * d_inner]
        x_inner, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Depthwise conv over sequence (causal)
        x_inner = rearrange(x_inner, "b l d -> b d l")
        x_inner = self.conv1d(x_inner)[:, :, :L]  # trim padding
        x_inner = F.silu(x_inner)

        # SSM
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        x_dbl = self.x_proj(rearrange(x_inner, "b d l -> b l d"))  # [B, L, dt_rank+2*d_state]
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        # Rearrange to channels-first for scan
        dt = rearrange(dt, "b l d -> b d l")
        B_ssm = rearrange(B_ssm, "b l n -> b n l")
        C_ssm = rearrange(C_ssm, "b l n -> b n l")

        y = self._selective_scan(x_inner, dt, A, B_ssm, C_ssm, self.D)
        y = rearrange(y, "b d l -> b l d")

        # Gate
        y = y * F.silu(z)

        return self.out_proj(y)


class MambaResidualBlock(nn.Module):
    """Pre-norm residual wrapper around a MambaBlock.

    Args:
        d_model: Feature dimension.
        drop_path: Stochastic depth drop probability.
        **mamba_kwargs: Forwarded to ``MambaBlock``.
    """

    def __init__(self, d_model: int, drop_path: float = 0.0, **mamba_kwargs: Any) -> None:  # noqa: D107
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, **mamba_kwargs)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm Mamba and residual connection."""
        return x + self.drop_path(self.mamba(self.norm(x)))


class _DropPath(nn.Module):
    """Stochastic depth per-sample drop (from timm, reproduced to avoid import)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand = torch.floor(rand + keep)
        return x / keep * rand


# ── ARCVim ─────────────────────────────────────────────────────────────────────


class ARCVim(nn.Module):
    """Visual Mamba (Vim) baseline for ARC-AGI.

    Replicates the Vim architecture adapted for dense ARC prediction:
    colour embedding → patch embedding → [task token | patch tokens] →
    absolute positional embed → bidirectional Mamba blocks →
    pixel-shuffle head.

    Bidirectionality: every consecutive pair of Mamba blocks processes the
    sequence in opposite directions; their outputs are summed (equivalent to
    Vim's ``if_bidirectional=True`` mode).

    Args:
        num_tasks:       Number of training tasks (for per-task learnable token).
        max_size:        Canvas side length in pixels.  Default 32.
        num_colors:      Number of output classes.  Default 12.
        embed_dim:       Token / model dimension.
        depth:           Total number of Mamba blocks (must be even for
                         bidirectional pairing).
        patch_size:      Pixel patch size.  Default 2.
        num_task_tokens: Number of prepended task tokens.
        d_state:         Mamba SSM state size.
        d_conv:          Mamba depthwise-conv width.
        expand:          Mamba channel expand factor.
        drop_path_rate:  Max stochastic-depth rate (linearly annealed per layer).
        dropout:         Dropout applied after positional embedding.
    """

    def __init__(  # noqa: D107
        self,
        num_tasks: int,
        max_size: int = 32,
        num_colors: int = 12,
        embed_dim: int = 256,
        depth: int = 12,
        patch_size: int = 2,
        num_task_tokens: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if depth % 2 != 0:
            raise ValueError(f"depth must be even for bidirectional pairing, got {depth}")

        self.max_size = max_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_task_tokens = num_task_tokens

        self.grid_size = max_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Embeddings (shared convention with ARCViT)
        from nvsubquadratic.networks.arc_embedding import ARCColorTaskEmbedding

        self.embedding = ARCColorTaskEmbedding(
            num_colors=num_colors,
            num_tasks=num_tasks,
            hidden_dim=embed_dim,
            num_task_tokens=num_task_tokens,
        )

        # Patch embedding — takes channels-first [B, C, H, W]
        self.patch_embed = PatchEmbed(
            img_size=max_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            bias=True,
        )

        # Absolute positional embedding (patches only, not task tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth schedule (per layer)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = nn.ModuleList(
            [
                MambaResidualBlock(
                    d_model=embed_dim,
                    drop_path=dpr[i],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = nn.LayerNorm(embed_dim)

        # Pixel-shuffle prediction head (same as ARCViT)
        self.head = nn.Linear(embed_dim, num_colors * patch_size * patch_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _forward_mamba_bidirectional(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional Mamba layers.

        Pairs consecutive layers: even-indexed runs forward, odd-indexed runs
        on the reversed sequence.  Outputs of each pair are summed.

        Args:
            x: [B, L, D]

        Returns:
            x: [B, L, D]
        """
        for i in range(0, len(self.layers), 2):
            x_fwd = self.layers[i](x)
            x_bwd = self.layers[i + 1](x.flip(1)).flip(1)
            x = x_fwd + x_bwd
        return x

    def forward(self, input_and_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass: embed → patch → Mamba → pixel-shuffle.

        Args:
            input_and_condition: dict with keys
                ``"input"``     — [B, H, W] int color tensor
                ``"condition"`` — dict with ``"task_id"`` [B] int tensor

        Returns:
            dict with ``"logits"`` — [B, num_colors, H, W] float tensor
        """
        pixel_values: torch.Tensor = input_and_condition["input"]
        condition: Optional[Dict[str, Any]] = input_and_condition["condition"]
        task_ids: torch.Tensor = condition["task_id"]

        if pixel_values.dim() != 3:
            raise ValueError("`input` must be (batch, height, width).")

        batch_size = pixel_values.size(0)

        # 1. Colour + task embedding
        x, task_tokens = self.embedding(pixel_values, task_ids)
        # x: [B, H, W, embed_dim] — channels-last; PatchEmbed wants [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # 2. Patch embedding → [B, num_patches, embed_dim]
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        # 3. Prepend task token(s) → [B, num_task_tokens + num_patches, embed_dim]
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        hidden = torch.cat([task_tokens, tokens], dim=1)
        hidden = self.pos_drop(hidden)

        # 4. Bidirectional Mamba blocks
        hidden = self._forward_mamba_bidirectional(hidden)

        # 5. Final norm
        hidden = self.norm_f(hidden)

        # 6. Discard task tokens; keep only patch tokens
        patch_tokens = hidden[:, self.num_task_tokens :, :]  # [B, num_patches, embed_dim]

        # 7. Pixel-shuffle head → [B, num_colors, H, W]
        logits_patched = self.head(patch_tokens)  # [B, num_patches, num_colors * p * p]
        logits = logits_patched.reshape(
            batch_size, self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.num_colors
        )
        logits = logits.permute(0, 1, 3, 2, 4, 5)  # [B, gh, p, gw, p, C]
        logits = logits.reshape(batch_size, self.max_size, self.max_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)  # [B, num_colors, H, W]

        return {"logits": logits}
