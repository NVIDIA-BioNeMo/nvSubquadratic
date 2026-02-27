"""Centralized kernel generator for all layers (MetaSIRENKernelND).

A single SIREN MLP generates convolutional kernels for every layer at once.
Instead of N independent SIRENKernelND instances (one per block), a single
shared positional embedding and hidden backbone feeds one large output linear
that projects to ``num_layers * out_dim`` channels.  The output is then split
into per-layer kernel slices.

Typical parameter flow::

    ViT5ClassificationNet
        └── MetaSIRENKernelND  (lives on the network, called once per forward)
                ├── SIRENPositionalEmbeddingND  (shared)
                ├── SIREN hidden layers          (shared)
                └── out_linear                   (hidden → num_layers * out_dim)
                        │
                        ▼
                split(out_dim, dim=-1) → [kernel₀, kernel₁, ..., kernel_{N-1}]
                        │
        distributed to ──┘
        Block₀.CKConvND, Block₁.CKConvND, ...

Each CKConvND receives its precomputed kernel and only performs the
shortcut + FFT convolution.
"""

import math

import torch
import torch.nn as nn

from nvsubquadratic.modules.kernels_nd import (
    Sine,
    SIRENPositionalEmbeddingND,
    _init_siren_weights,
)


class MetaSIRENKernelND(nn.Module):
    """Single SIREN MLP that generates convolutional kernels for all layers at once.

    Architecture::

        SIRENPositionalEmbeddingND
            → SIREN hidden layers (shared)
            → Linear(mlp_hidden_dim, num_layers * out_dim)
            → split into num_layers chunks of out_dim

    Args:
        num_layers: Number of network blocks (= number of kernels to generate).
        out_dim: Per-layer kernel output channels (e.g. hidden_dim=384).
        data_dim: Spatial dimensionality (1 for sequences, 2 for images, 3 for video).
        mlp_hidden_dim: Hidden width of the shared SIREN backbone.
        num_mlp_layers: Total SIREN layers including first (embedding→hidden) and
            hidden layers (>= 2).
        embedding_dim: Dimensionality of the SIREN positional embedding.
        omega_0: Frequency scaling for the first SIREN layer.
        hidden_omega_0: Frequency scaling for subsequent SIREN layers.
        L_cache: Cache extent for the positional embedding grid.
        use_bias: Whether to use bias in SIREN linear layers.
    """

    def __init__(
        self,
        num_layers: int,
        out_dim: int,
        data_dim: int,
        mlp_hidden_dim: int,
        num_mlp_layers: int,
        embedding_dim: int,
        omega_0: float,
        hidden_omega_0: float,
        L_cache: int,
        use_bias: bool = True,
    ):
        assert num_mlp_layers >= 2, f"num_mlp_layers must be >= 2, got {num_mlp_layers}"

        super().__init__()

        self.num_layers = num_layers
        self.out_dim = out_dim
        self.data_dim = data_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        # ── Shared positional embedding ──────────────────────────────────────
        self.positional_embedding = SIRENPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            omega_0=omega_0,
            L_cache=L_cache,
            use_bias=use_bias,
        )

        # ── Shared SIREN hidden layers ───────────────────────────────────────
        self.kernel_network = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim, bias=use_bias),
            Sine(),
        )
        for _ in range(num_mlp_layers - 2):
            self.kernel_network.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=use_bias))
            self.kernel_network.append(Sine())

        # ── Single output projection: hidden → all layers at once ────────────
        total_out_dim = num_layers * out_dim
        self.out_linear = nn.Linear(mlp_hidden_dim, total_out_dim, bias=use_bias)

        # ── Initialization ───────────────────────────────────────────────────
        # SIREN init for hidden layers
        for layer in self.kernel_network:
            if isinstance(layer, nn.Linear):
                _init_siren_weights(layer, is_first_layer=False, w0=hidden_omega_0)

        # SIREN init + Wang scaling for output layer
        _init_siren_weights(self.out_linear, is_first_layer=False, w0=hidden_omega_0)
        with torch.no_grad():
            self.out_linear.weight.data *= math.sqrt(1.0 / (L_cache**data_dim))

        # Exclude backbone params from weight decay (output layer keeps decay)
        for param in self.kernel_network.parameters():
            param._no_weight_decay = True

    def forward(self, seq_lens: tuple[int, ...]) -> list[torch.Tensor]:
        """Generate kernels for all layers in a single forward pass.

        Args:
            seq_lens: Grid dimensions to generate kernels for, e.g. ``(14, 14)``
                for a 14x14 2D grid.  Same semantics as
                :meth:`SIRENKernelND.forward`.

        Returns:
            List of ``num_layers`` kernel tensors, each of shape
            ``[1, *spatial_dims, out_dim]``.
        """
        pos_emb, _ = self.positional_embedding(seq_lens)
        features = self.kernel_network(pos_emb)
        all_kernels = self.out_linear(features)  # [1, *spatial, num_layers * out_dim]

        return list(all_kernels.split(self.out_dim, dim=-1))

    def extra_repr(self) -> str:
        return (
            f"num_layers={self.num_layers}, out_dim={self.out_dim}, "
            f"data_dim={self.data_dim}, mlp_hidden_dim={self.mlp_hidden_dim}"
        )
