"""Residual block implementation for ND signals, composed of a sequence mixer and an MLP."""

from typing import Union

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ResidualBlock(torch.nn.Module):
    """Residual block for ND signals, composed of a sequence mixer and an MLP."""

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        condition_mixer_cfg: LazyConfig,
        condition_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        dropout_cfg: LazyConfig,
    ):
        """Initialize the ResidualBlock.

        Args:
            sequence_mixer_cfg: LazyConfig for the sequence mixer layer.
            sequence_mixer_norm_cfg: LazyConfig for the sequence mixer norm.
            condition_mixer_cfg: LazyConfig for the condition mixer layer.
            condition_mixer_norm_cfg: LazyConfig for the condition mixer norm.
            mlp_cfg: LazyConfig for the MLP layer.
            mlp_norm_cfg: LazyConfig for the MLP norm.
            dropout_cfg: LazyConfig for the dropout layer.
        """
        if sequence_mixer_cfg.__target__ == torch.nn.Identity:
            assert sequence_mixer_norm_cfg.__target__ == torch.nn.Identity, (
                "Sequence mixer norm must be Identity if sequence mixer is Identity"
            )
        if mlp_cfg.__target__ == torch.nn.Identity:
            assert mlp_norm_cfg.__target__ == torch.nn.Identity, "MLP norm must be Identity if MLP is Identity"
        if condition_mixer_cfg.__target__ == torch.nn.Identity:
            assert condition_mixer_norm_cfg.__target__ == torch.nn.Identity, (
                "Condition mixer norm must be Identity if condition mixer is Identity"
            )

        super().__init__()
        # Instantiate sequence mixer layer
        self.sequence_mixer = instantiate(sequence_mixer_cfg)
        # Instantiate input norm
        self.input_norm = instantiate(sequence_mixer_norm_cfg)
        # Exclude self.input_norm from the parameter group with weight decay
        for param in self.input_norm.parameters():
            param._no_weight_decay = True

        # Instantiate cross attention layer
        self.condition_mixer = instantiate(condition_mixer_cfg)
        # Instantiate cross attention norm
        self.condition_mixer_norm = instantiate(condition_mixer_norm_cfg)
        # Exclude self.condition_mixer_norm from the parameter group with weight decay
        for param in self.condition_mixer_norm.parameters():
            param._no_weight_decay = True

        # Instantiate MLP layer
        self.mlp = instantiate(mlp_cfg)
        # Instantiate MLP norm
        self.mlp_norm = instantiate(mlp_norm_cfg)
        # Exclude self.mlp_norm from the parameter group with weight decay
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        # Instantiate dropout
        self.dropout = instantiate(dropout_cfg)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
            condition (torch.Tensor): Condition tensor of shape (batch_size, *spatial_dims_condition, num_hidden_channels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, *spatial_dims, num_hidden_channels)
        """
        # Mixer branch
        if not isinstance(self.sequence_mixer, torch.nn.Identity):
            residual = x
            x = self.input_norm(x)
            x = self.sequence_mixer(x)
            x = self.dropout(x)
            x = x + residual

        # Cross attention branch
        if not isinstance(self.condition_mixer, torch.nn.Identity):
            assert condition is not None, "Condition must be provided if condition mixer is not Identity."
            residual = x
            x = self.condition_mixer_norm(x)
            x = self.condition_mixer(x, condition)
            x = self.dropout(x)
            x = x + residual

        # MLP branch
        if not isinstance(self.mlp, torch.nn.Identity):
            residual = x
            x = self.mlp_norm(x)
            x = self.mlp(x)
            x = self.dropout(x)
            x = x + residual
        return x


class AdaLNZeroResidualBlock(torch.nn.Module):
    """Residual block with inline AdaLN-Zero modulation for mixer and MLP branches."""

    def __init__(
        self,
        sequence_mixer_cfg: LazyConfig,
        sequence_mixer_norm_cfg: LazyConfig,
        mlp_cfg: LazyConfig,
        mlp_norm_cfg: LazyConfig,
        condition_norm_cfg: LazyConfig,
        dropout_cfg: LazyConfig,
        hidden_dim: int,
    ):
        """Initialize the AdaLNZeroResidualBlock."""
        super().__init__()

        # Mixer branch handles spatial/temporal interactions on the residual stream.
        self.sequence_mixer = instantiate(sequence_mixer_cfg)
        self.sequence_norm = instantiate(sequence_mixer_norm_cfg)
        for param in self.sequence_norm.parameters():
            param._no_weight_decay = True

        # MLP branch refines each position independently.
        self.mlp = instantiate(mlp_cfg)
        self.mlp_norm = instantiate(mlp_norm_cfg)
        for param in self.mlp_norm.parameters():
            param._no_weight_decay = True

        # Optional pre-normalization for the conditioning vector.
        self.condition_norm = instantiate(condition_norm_cfg)
        for param in self.condition_norm.parameters():
            param._no_weight_decay = True

        # Shared dropout applied after each residual branch.
        self.dropout = instantiate(dropout_cfg)

        # Single zero-initialised projection (DiT style) producing shift/scale/gate for both branches.
        self.condition_proj = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim * 6))
        torch.nn.init.zeros_(self.condition_proj[1].weight)
        torch.nn.init.zeros_(self.condition_proj[1].bias)

    def forward(self, x: torch.Tensor, condition: Union[torch.Tensor | None]) -> torch.Tensor:
        """Apply AdaLN-Zero residual mixing conditioned on the provided tensor."""
        if condition is None:
            raise ValueError("AdaLNZeroResidualBlock requires a conditioning tensor.")

        # Collapse any spatial conditioning down to a single latent vector per item.
        cond = condition  # (B, *spatial?, hidden_dim)
        if cond.ndim >= 3:
            cond = cond.mean(dim=tuple(range(1, cond.ndim - 1)))  # (B, hidden_dim)
        cond = self.condition_norm(cond)  # (B, hidden_dim)

        # Map the conditioning vector to shift/scale/gate triplets for both branches.
        cond_mapped = self.condition_proj(cond)  # (B, 6 * hidden_dim)
        shift_seq, scale_seq, gate_seq, shift_mlp, scale_mlp, gate_mlp = cond_mapped.chunk(
            6, dim=-1
        )  # each (B, hidden_dim)

        def expand(param: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            """Broadcast a [B, hidden_dim] vector across ref's spatial axes."""
            while param.ndim < ref.ndim:
                param = param.unsqueeze(1)  # add singleton dims for broadcasting
            return param.expand(*ref.shape[:-1], param.shape[-1])  # match ref spatial layout

        # Modulate the sequence mixer with AdaLN-Zero and add its residual output.
        seq_norm = self.sequence_norm(x)  # (B, *spatial_dims, hidden_dim)
        seq_mod = seq_norm * (1.0 + expand(scale_seq, seq_norm)) + expand(
            shift_seq, seq_norm
        )  # (B, *spatial_dims, hidden_dim)
        seq_out = self.sequence_mixer(seq_mod)  # (B, *spatial_dims, hidden_dim)
        seq_out = self.dropout(seq_out)  # (B, *spatial_dims, hidden_dim)
        seq_out = seq_out * expand(gate_seq, seq_out)  # (B, *spatial_dims, hidden_dim)
        x = x + seq_out  # (B, *spatial_dims, hidden_dim)

        # Apply the same AdaLN-Zero recipe to the MLP branch.
        mlp_norm = self.mlp_norm(x)  # (B, *spatial_dims, hidden_dim)
        mlp_mod = mlp_norm * (1.0 + expand(scale_mlp, mlp_norm)) + expand(
            shift_mlp, mlp_norm
        )  # (B, *spatial_dims, hidden_dim)
        mlp_out = self.mlp(mlp_mod)  # (B, *spatial_dims, hidden_dim)
        mlp_out = self.dropout(mlp_out)  # (B, *spatial_dims, hidden_dim)
        mlp_out = mlp_out * expand(gate_mlp, mlp_out)  # (B, *spatial_dims, hidden_dim)
        x = x + mlp_out  # (B, *spatial_dims, hidden_dim)

        return x
