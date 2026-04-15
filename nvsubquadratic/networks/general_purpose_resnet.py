# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Simple implementation of a ResNet for general purpose tasks."""

from typing import Sequence

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualNetwork(nn.Module):
    """Simple implementation of a Residual Network for general purpose tasks.

    It assumes:
    - the input tensor is of shape (batch_size, *spatial_dims, in_channels).
    - the output tensor is of shape (batch_size, *spatial_dims, out_channels).

    Optionally supports **register tokens** (following the ViT-5 pattern):
    when ``num_registers > 0``, learnable register embeddings are prepended as
    an extra row along the first spatial dimension after the input projection.
    The row is ``[reg_0, reg_1, ..., reg_{R-1}, 0, 0, ..., 0]`` padded to the
    full width of that dimension.  Registers participate in every block's mixing
    and are extracted/pooled inside each ``ResidualBlock`` (via
    ``register_pooling_cfg``) to produce a FiLM conditioning vector.  The
    register row is stripped before the output projection and readout.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_blocks (int): Number of blocks
        hidden_dim (int): Number of hidden dimensions
        in_proj_cfg (LazyConfig): Configuration for the input projection
        out_proj_cfg (LazyConfig): Configuration for the output projection
        norm_cfg (LazyConfig): Configuration for the normalization
        block_cfg (LazyConfig): Configuration for the residual block
        dropout_in_cfg (LazyConfig): Configuration for the dropout in layer (applied to the input)
        condition_in_proj_cfg (LazyConfig | None): Configuration for the condition input projection or None if no condition is used.
            If provided, the condition tensor is of shape [B, * spatial_dims_condition, hidden_dim].
            If not provided, the condition tensor is None.
        target_size (int | tuple | None): Size of the readout region. Can be:
            - int: Same size for all spatial dimensions (e.g., 16 for 16x16 in 2D)
            - tuple: Different size per dimension. For 3D spatial recall where the target
              is a 2D image on the last depth slice, use (1, H, W) to extract only the
              last depth slice with HxW spatial region.
            - None: No readout extraction (return full output)
        num_registers (int): Number of learnable register tokens to prepend.
            When > 0, a register row is inserted along the first spatial dim.
        reg_init (str): Initialization for register tokens: ``"zeros"`` or
            ``"trunc_normal"``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        hidden_dim: int,
        data_dim: int,
        in_proj_cfg: LazyConfig,
        out_proj_cfg: LazyConfig,
        norm_cfg: LazyConfig,
        block_cfg: LazyConfig,
        dropout_in_cfg: LazyConfig,
        condition_in_proj_cfg: LazyConfig | None = None,
        target_size: int | Sequence[int] | None = None,
        num_registers: int = 0,
        reg_init: str = "zeros",
    ):
        """Initialize the ResidualNetwork."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.num_registers = num_registers

        # Instantiate dropout_in
        self.dropout_in = instantiate(dropout_in_cfg)

        # Instantiate input projection for the network
        self.in_proj = instantiate(in_proj_cfg)

        if condition_in_proj_cfg is not None:
            # Instantiate condition input projection for the network
            self.condition_in_proj = instantiate(
                condition_in_proj_cfg, in_features=hidden_dim, out_features=hidden_dim
            )
        else:
            self.condition_in_proj = None

        # Create residual blocks
        self.blocks = nn.ModuleList([instantiate(block_cfg) for _ in range(num_blocks)])

        # Instantiate output norm
        self.out_norm = instantiate(norm_cfg)
        # Exclude self.out_norm from the parameter group with weight decay
        for param in self.out_norm.parameters():
            param._no_weight_decay = True

        # Instantiate output projection
        self.out_proj = instantiate(out_proj_cfg)

        # Target size for readout -- only used for spatial recall tasks for now.
        # Convert to tuple for consistent handling
        if target_size is None:
            self.target_size = None
        elif isinstance(target_size, int):
            self.target_size = (target_size,) * data_dim
        else:
            self.target_size = tuple(target_size)

        # Register tokens (ViT-5 pattern adapted for ND spatial tensors).
        # Registers are prepended as an extra row along dim=1 (first spatial dim).
        # Layout: [regs, zero_pad] where pad fills the row to width spatial_dim_1.
        if num_registers > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, num_registers, hidden_dim))
            self.reg_token._no_weight_decay = True
            if reg_init == "trunc_normal":
                nn.init.trunc_normal_(self.reg_token, std=0.02)
        else:
            self.reg_token = None

    def _get_readout_region(self, x: torch.Tensor) -> torch.Tensor:
        """Get the readout region (bottom-right target_size region) of the input tensor.

        Args:
            x: Input tensor of shape [batch_size, *spatial_dims, out_channels].

        Returns:
            torch.Tensor: Readout region. Shape depends on target_size:
                - For target_size=(L,): [batch_size, L, out_channels]
                - For target_size=(H, W): [batch_size, H, W, out_channels]
                - For target_size=(D, H, W): [batch_size, D, H, W, out_channels]
                - For target_size=(1, H, W) on 3D input: [batch_size, H, W, out_channels] (squeezed)
        """
        spatial_ndim = x.ndim - 2  # Exclude batch and channel dims

        if len(self.target_size) != spatial_ndim:
            raise ValueError(
                f"target_size has {len(self.target_size)} dimensions but input has {spatial_ndim} spatial dimensions. "
                f"target_size={self.target_size}, input shape={x.shape}"
            )

        # Build slice/index for each spatial dimension
        # x shape: [batch, *spatial_dims, channels]
        # Using integer index (-1) auto-removes dimension, slice(-size, None) keeps it
        slices = [slice(None)]  # batch dimension
        for size in self.target_size:
            if size == 1:
                slices.append(-1)  # integer index removes dimension
            else:
                slices.append(slice(-size, None))
        slices.append(slice(None))  # channel dimension

        return x[tuple(slices)]

    def _prepend_register_row(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend a register row along the first spatial dimension.

        For a 2D input ``[B, H, W, C]`` this inserts a new row at dim=1 giving
        ``[B, H+1, W, C]``.  The first ``num_registers`` positions of the new
        row contain the learnable register embeddings; the rest are zero-padded.

        Works for arbitrary ``data_dim`` (1D, 2D, 3D).
        """
        B = x.shape[0]
        first_spatial_width = x.shape[2] if self.data_dim >= 2 else 1
        remaining_spatial = x.shape[3:-1] if self.data_dim >= 3 else ()

        # reg_token: [1, R, C] → expand to [B, R, C]
        regs = self.reg_token.expand(B, -1, -1)

        # Zero-pad to fill the first spatial width
        pad_size = first_spatial_width - self.num_registers
        if pad_size > 0:
            pad = x.new_zeros(B, pad_size, self.hidden_dim)
            row = torch.cat([regs, pad], dim=1)  # [B, first_spatial_width, C]
        else:
            row = regs[:, :first_spatial_width, :]

        # Reshape row to match x's full spatial layout:
        # 1D: [B, W, C] → already [B, 1_row_elements, C], just unsqueeze not needed
        # 2D: [B, W, C] → [B, 1, W, C]
        # 3D: [B, W, C] → [B, 1, W, *remaining, C] — broadcast remaining dims
        if self.data_dim == 1:
            # 1D: x is [B, L, C], row is [B, R_padded, C] but we want a single
            # "row" of width 1 containing registers.  For 1D we just prepend
            # the register tokens directly along the sequence dim.
            reg_row = row  # [B, num_regs_padded, C]
        elif self.data_dim == 2:
            reg_row = row.unsqueeze(1)  # [B, 1, W, C]
        else:
            # 3D+: insert singleton dims for remaining spatial dims, then expand
            reg_row = row.unsqueeze(1)  # [B, 1, W, C]
            for _ in remaining_spatial:
                reg_row = reg_row.unsqueeze(-2)  # add dims before C
            reg_row = reg_row.expand(B, 1, first_spatial_width, *remaining_spatial, self.hidden_dim)

        return torch.cat([reg_row, x], dim=1)

    def _strip_register_row(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the prepended register row (first slice along dim=1)."""
        return x[:, 1:]

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the ResidualNetwork.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

            - input: Input tensor of shape [B, * spatial_dims, hidden_dim].
            - condition: Condition tensor of shape [B, * spatial_dims_condition, hidden_dim].

        Returns:
            Dict[str, torch.Tensor]:
                - "logits": tensor of shape [B, * spatial_dims, out_channels].
        """
        # Extract the input and condition from the dictionary
        x, condition = input_and_condition["input"], input_and_condition["condition"]

        # Apply in_dropout to the input
        x = self.dropout_in(x)
        # Apply input projection
        x = self.in_proj(x)

        # Prepend register row if configured (ViT-5 pattern)
        if self.reg_token is not None:
            x = self._prepend_register_row(x)

        # Apply condition input projection if provided
        if self.condition_in_proj is not None:
            assert condition is not None, "Condition must be provided if condition input projection is provided"
            condition = self.condition_in_proj(condition)

        # Apply residual blocks (with or without condition)
        for block in self.blocks:
            x = block(x, condition)

        # Strip register row before output projection
        if self.reg_token is not None:
            x = self._strip_register_row(x)

        # Apply output norm
        x = self.out_norm(x)
        # Apply output projection
        x = self.out_proj(x)

        # Get the readout region if target size is provided
        if self.target_size is not None:
            x = self._get_readout_region(x)

        return {"logits": x}
