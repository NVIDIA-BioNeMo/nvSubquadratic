"""Unified UNet with pluggable blocks via LazyConfig.

A single encoder-decoder UNet backbone where the per-stage block is
specified as a ``block_cfg`` LazyConfig — following the same pattern as
:class:`ResidualNetwork` / :class:`ResidualBlock`.

The UNet fills in ``dim`` and ``spatial_res`` per stage when instantiating
the block config, so the config only needs to specify block-specific
parameters (e.g. ``num_heads`` for attention, ``omega_0`` for Hyena).

Built-in block types (importable from this module):
    - :class:`ConvNeXtBlock` — depthwise-conv + channel MLP (Liu et al., 2022).
    - :class:`AttentionBlock` — multi-head self-attention + FFN.
    - :class:`HyenaBlock` — gated global convolution via CKConvND/SIREN + FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint

from nvsubquadratic.lazy_config import instantiate
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Shared helpers ──────────────────────────────────────────────────────────

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

_TO_CHANNELS_LAST = {
    2: "N C H W -> N H W C",
    3: "N C D H W -> N D H W C",
}
_TO_CHANNELS_FIRST = {
    2: "N H W C -> N C H W",
    3: "N D H W C -> N C D H W",
}


class _LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first data formats."""

    def __init__(self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        if data_format == "channels_last":
            self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            return F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight


class _Upsample(nn.Module):
    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            _LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class _Downsample(nn.Module):
    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            _LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


# ─── Block implementations ──────────────────────────────────────────────────
#
# All blocks share a uniform constructor interface:
#   (dim, n_spatial_dims, spatial_res=None, drop_path=0.0, ...)
# so they are interchangeable inside the UNet via LazyConfig.
# The UNet fills in ``dim`` and ``spatial_res`` per stage.


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: DwConv 7x7 -> LN -> Linear(4x) -> GELU -> Linear -> residual."""

    def __init__(self, dim, n_spatial_dims, spatial_res=None, drop_path=0.0, layer_scale_init_value=1e-6):
        """Initialize ConvNeXtBlock with depthwise conv, layer norm, and pointwise MLP."""
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Apply ConvNeXt block: dwconv -> norm -> MLP -> residual."""
        residual = x
        x = self.dwconv(x)
        x = rearrange(x, _TO_CHANNELS_LAST[self.n_spatial_dims])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, _TO_CHANNELS_FIRST[self.n_spatial_dims])
        x = residual + self.drop_path(x)
        return x


class AttentionBlock(nn.Module):
    """Transformer block: LN -> MHSA -> residual -> LN -> FFN -> residual.

    Operates channels-first externally (NCHW) but converts to channels-last
    internally for attention and FFN.
    """

    def __init__(self, dim, n_spatial_dims, spatial_res=None, drop_path=0.0, num_heads=6, mlp_ratio=4):
        """Initialize AttentionBlock with MHSA, output projection, and FFN."""
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn = Attention(
            hidden_dim=dim,
            num_heads=num_heads,
            apply_qk_norm=True,
            use_rope=False,
            is_causal=False,
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Apply attention block: MHSA -> residual -> FFN -> residual."""
        x = rearrange(x, _TO_CHANNELS_LAST[self.n_spatial_dims])

        # Self-attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        x = self.attn(q, k, v)
        x = self.out_proj(x)
        x = residual + self.drop_path(x)

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = residual + self.drop_path(x)

        x = rearrange(x, _TO_CHANNELS_FIRST[self.n_spatial_dims])
        return x


def _build_hyena(dim, n_spatial_dims, spatial_res, omega_0, siren_layers, siren_hidden_dim):
    """Create a Hyena module configured for the given channel width and spatial resolution."""
    global_conv = CKConvND(
        data_dim=n_spatial_dims,
        hidden_dim=dim,
        fft_padding="circular",
        use_fp16_fft=False,
        kernel_cfg=SIRENKernelND(
            data_dim=n_spatial_dims,
            out_dim=dim,
            mlp_hidden_dim=siren_hidden_dim,
            num_layers=siren_layers,
            embedding_dim=siren_hidden_dim,
            omega_0=omega_0,
            L_cache=spatial_res,
            use_bias=True,
            hidden_omega_0=1.0,
        ),
        mask_cfg=nn.Identity(),
        grid_type="single",
    )
    short_conv = conv_modules[n_spatial_dims](
        in_channels=3 * dim,
        out_channels=3 * dim,
        kernel_size=3,
        groups=3 * dim,
        padding=1,
        bias=False,
    )
    return Hyena(
        global_conv_cfg=global_conv,
        short_conv_cfg=short_conv,
        gate_nonlinear_cfg=nn.SiLU(),
        gate_nonlinear_2_cfg=nn.Sigmoid(),
        pixelhyena_norm_cfg=RMSNorm(dim=dim),
        output_norm_cfg=RMSNorm(dim=dim),
        qk_norm_cfg=L2Norm(),
        use_rope=False,
    )


class HyenaBlock(nn.Module):
    """Hyena block: LN -> QKV -> Hyena gated global conv -> residual -> LN -> FFN -> residual.

    Operates channels-first externally (NCHW) but converts to channels-last
    internally for the Hyena mixer and FFN.  ``spatial_res`` is **required** to
    set the SIREN kernel's L_cache correctly for each UNet stage.
    """

    def __init__(
        self,
        dim,
        n_spatial_dims,
        spatial_res=None,
        drop_path=0.0,
        mlp_ratio=4,
        omega_0=30.0,
        siren_layers=3,
        siren_hidden_dim=64,
    ):
        """Initialize HyenaBlock with gated global conv, output projection, and FFN."""
        super().__init__()
        assert spatial_res is not None, "HyenaBlock requires spatial_res for the SIREN kernel"
        self.n_spatial_dims = n_spatial_dims
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.hyena = _build_hyena(dim, n_spatial_dims, spatial_res, omega_0, siren_layers, siren_hidden_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Apply Hyena block: gated global conv -> residual -> FFN -> residual."""
        x = rearrange(x, _TO_CHANNELS_LAST[self.n_spatial_dims])

        # Hyena mixer
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        x = self.hyena(q, k, v)
        x = self.out_proj(x)
        x = residual + self.drop_path(x)

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = residual + self.drop_path(x)

        x = rearrange(x, _TO_CHANNELS_FIRST[self.n_spatial_dims])
        return x


# ─── Unified Stage & UNet ───────────────────────────────────────────────────


class _Stage(nn.Module):
    """UNet stage: sequence of blocks + optional resampling."""

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_res,
        block_cfg,
        depth=1,
        drop_path=0.0,
        mode="down",
        skip_project=False,
    ):
        super().__init__()

        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = _Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = _Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [instantiate(block_cfg, dim=dim_in, spatial_res=spatial_res) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.skip_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.resample(x)
        return x


class UNet(nn.Module):
    """Unified UNet with pluggable blocks — channels-first (NCHW / NCDHW) interface.

    Follows the same LazyConfig pattern as :class:`ResidualNetwork`: the caller
    provides a ``block_cfg`` LazyConfig and the UNet fills in ``dim`` and
    ``spatial_res`` per stage when instantiating blocks.

    Args:
        dim_in: Number of input channels.
        dim_out: Number of output channels.
        n_spatial_dims: 2 for images, 3 for volumes.
        spatial_resolution: Tuple of spatial sizes (used to compute per-stage
            resolutions for resolution-aware blocks like Hyena).
        stages: Number of encoder/decoder stages (default 4).
        blocks_per_stage: Blocks per stage (default 1).
        blocks_at_neck: Blocks at bottleneck (default 1).
        init_features: Feature map width at the first stage (default 32).
        block_cfg: LazyConfig targeting a block class (e.g. ``ConvNeXtBlock``,
            ``AttentionBlock``, ``HyenaBlock``).  The UNet fills in ``dim`` and
            ``spatial_res`` per stage; the config only needs block-specific
            params (e.g. ``num_heads``, ``omega_0``).
        gradient_checkpointing: Use activation checkpointing to save memory.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...] | None = None,
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        block_cfg=None,
        gradient_checkpointing: bool = False,
    ):
        """Build encoder-decoder UNet with the given block config and depth/width."""
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        features = init_features
        self.gradient_checkpointing = gradient_checkpointing

        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]

        # Per-stage spatial resolutions (assumes isotropic and halves each stage)
        base_res = spatial_resolution[0] if spatial_resolution else 0
        stage_resolutions = [base_res // (2**i) for i in range(stages)]
        neck_resolution = base_res // (2**stages) if base_res else 0

        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](dim_in, features, kernel_size=3, padding=1)
        self.out_proj = conv_modules[n_spatial_dims](features, dim_out, kernel_size=3, padding=1)
        for i in range(stages):
            encoder.append(
                _Stage(
                    encoder_dims[i],
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    spatial_res=stage_resolutions[i],
                    block_cfg=block_cfg,
                    depth=blocks_per_stage,
                    mode="down",
                )
            )
            dec_res = neck_resolution * (2**i) if i > 0 else neck_resolution
            decoder.append(
                _Stage(
                    decoder_dims[i],
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    spatial_res=dec_res,
                    block_cfg=block_cfg,
                    depth=blocks_per_stage,
                    mode="up",
                    skip_project=i != 0,
                )
            )
        self.encoder = nn.ModuleList(encoder)
        self.neck = _Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            n_spatial_dims,
            spatial_res=neck_resolution,
            block_cfg=block_cfg,
            depth=blocks_at_neck,
            mode="neck",
        )
        self.decoder = nn.ModuleList(decoder)

    def _optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with skip connections, process at neck, decode with skip fusion."""
        x = self.in_proj(x)
        skips = []
        for enc in self.encoder:
            skips.append(x)
            x = self._optional_checkpointing(enc, x)
        x = self.neck(x)
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)
            x = dec(x)
        x = self.out_proj(x)
        return x


# ─── Channels-last dict wrapper for the nvSubquadratic training infra ────────

_CHANNELS_LAST_TO_FIRST = {
    2: "B H W C -> B C H W",
    3: "B D H W C -> B C D H W",
}
_CHANNELS_FIRST_TO_LAST = {
    2: "B C H W -> B H W C",
    3: "B C D H W -> B D H W C",
}


class WellUNet(nn.Module):
    """Unified UNet with the dict-based channels-last interface expected by WELLRegressionWrapper.

    Input:  ``{"input": [B, *spatial, C_in], "condition": None}``
    Output: ``{"logits": [B, *spatial, C_out]}``

    Constructor args are forwarded to :class:`UNet`.
    """

    def __init__(self, **kwargs):
        """Initialize by forwarding all kwargs to :class:`UNet`."""
        super().__init__()
        self.net = UNet(**kwargs)
        self._n_spatial_dims = self.net.n_spatial_dims

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transpose to channels-first, run UNet, transpose back."""
        x = input_and_condition["input"]
        x = rearrange(x, _CHANNELS_LAST_TO_FIRST[self._n_spatial_dims])
        y = self.net(x)
        y = rearrange(y, _CHANNELS_FIRST_TO_LAST[self._n_spatial_dims])
        return {"logits": y}
