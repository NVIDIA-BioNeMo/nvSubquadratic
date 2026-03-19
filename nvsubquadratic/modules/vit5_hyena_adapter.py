"""Adapter to plug 2D sequence mixers (e.g. Hyena) into the ViT5 token-sequence architecture.

The ViT5 architecture processes [B, T, C] sequences. 2D mixers like Hyena expect
[B, H, W, C] spatial grids. This adapter reshapes the flat token sequence to a 2D
grid, applies the inner mixer, and reshapes back.

All token ordering (CLS position, register placement) is handled upstream by the
network (e.g. ViT5ClassificationNet with prepend_registers=True), so this adapter
treats the entire sequence as a flat spatial grid — it does not know or care about
which tokens are CLS, registers, or patches.

When ``distribute_registers=True``, registers are evenly interleaved among patches
in the 1D sequence. The adapter strips them before reshaping to the 2D grid, runs
Hyena on a pure patch grid, updates registers via a communication module (cross-
attention or local pooling), and reinserts them into the output sequence.
"""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5HyenaAdapter(nn.Module):
    """Bridges ViT5's [B, T, C] token sequences and Hyena's [B, H, W, C] spatial interface.

    Args:
        inner_mixer_cfg: LazyConfig for the 2D sequence mixer (e.g. QKVSequenceMixer wrapping Hyena).
        grid_w: Width of the 2D spatial grid. The height is inferred as T // grid_w
            (or num_patches // grid_w when distribute_registers is True).
        distribute_registers: If True, strip registers from their distributed positions
            before the 2D reshape and reinsert them after mixing.
        num_registers: Number of register tokens. Only used when distribute_registers=True.
        num_patches: Number of patch tokens. Only used when distribute_registers=True.
        register_comm_cfg: LazyConfig for the register-patch communication module
            (RegisterCrossAttention or RegisterLocalPooling). Only used when
            distribute_registers=True. Required when distribute_registers=True.
    """

    def __init__(
        self,
        inner_mixer_cfg: LazyConfig,
        grid_w: int,
        distribute_registers: bool = False,
        num_registers: int = 0,
        num_patches: int = 0,
        register_comm_cfg: LazyConfig | None = None,
    ):
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.grid_w = grid_w
        self.distribute_registers = distribute_registers
        self.num_registers = num_registers
        self.num_patches = num_patches

        if distribute_registers:
            assert num_registers > 0, "num_registers must be > 0 when distribute_registers=True"
            assert num_patches > 0, "num_patches must be > 0 when distribute_registers=True"
            assert register_comm_cfg is not None, (
                "register_comm_cfg is required when distribute_registers=True"
            )
            self.register_comm = instantiate(register_comm_cfg)

            # Precompute register indices in the interleaved sequence
            stride = num_patches // num_registers
            register_indices = torch.tensor(
                [stride * (i + 1) + i for i in range(num_registers)], dtype=torch.long
            )
            self.register_buffer("register_indices", register_indices)

            # Precompute patch indices in the interleaved sequence
            total_len = num_patches + num_registers
            register_index_set = set(register_indices.tolist())
            patch_indices = torch.tensor(
                [i for i in range(total_len) if i not in register_index_set],
                dtype=torch.long,
            )
            self.register_buffer("patch_indices", patch_indices)
        else:
            self.register_comm = None

    def forward(self, x: torch.Tensor, **mixer_kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] token sequence. When distribute_registers=False, T must be
                divisible by grid_w. When distribute_registers=True, T = num_patches +
                num_registers.
            **mixer_kwargs: Forwarded to the inner mixer (e.g. ``conditioning`` for FiLM).

        Returns:
            [B, T, C] with tokens mixed via the 2D inner mixer.
        """
        if not self.distribute_registers:
            B, T, C = x.shape
            x = x.reshape(B, T // self.grid_w, self.grid_w, C)
            x = self.inner_mixer(x, **mixer_kwargs)
            x = x.reshape(B, T, C)
            return x

        # Distributed registers: strip → Hyena on patches → comm → reinsert
        B, T, C = x.shape

        # Strip registers from distributed positions
        # Use gather/scatter instead of index_select/index_copy_ for torch.compile compatibility
        # (index_select backward uses nonzero() which has dynamic output shapes)
        patch_idx = self.patch_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        reg_idx = self.register_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        patches = torch.gather(x, 1, patch_idx)       # [B, num_patches, C]
        regs = torch.gather(x, 1, reg_idx)             # [B, num_registers, C]

        # Reshape patches to 2D grid and run Hyena
        H = self.num_patches // self.grid_w
        patches = patches.reshape(B, H, self.grid_w, C)
        patches = self.inner_mixer(patches, **mixer_kwargs)
        patches = patches.reshape(B, self.num_patches, C)

        # Update registers from mixed patches
        regs = self.register_comm(regs, patches)  # [B, num_registers, C]

        # Reinsert into full sequence
        out = torch.zeros(B, T, C, dtype=x.dtype, device=x.device)
        out = out.scatter(1, patch_idx, patches.to(out.dtype))
        out = out.scatter(1, reg_idx, regs.to(out.dtype))
        return out

    def extra_repr(self) -> str:
        s = f"grid_w={self.grid_w}"
        if self.distribute_registers:
            s += f", distribute_registers=True, num_registers={self.num_registers}, num_patches={self.num_patches}"
        return s
