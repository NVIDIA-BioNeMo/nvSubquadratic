#!/usr/bin/env python3
"""Compare FLOPs and parameters for UNet with ConvNeXt, Attention, and Hyena blocks.

Usage:
    python examples/well/euler_multi_quadrants_periodicBC/compare_models.py
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet import (
    AttentionBlock,
    ConvNeXtBlock,
    HyenaBlock,
    UNet,
)


# ─── Shared constants (from _base.py) ────────────────────────────────────────
DATA_DIM = 2
SPATIAL_RESOLUTION = (512, 512)
IN_CHANNELS = 20
OUT_CHANNELS = 5
BATCH_SIZE = 1  # for counting

# ─── Shared UNet hyperparameters ──────────────────────────────────────────────
INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 4
BLOCKS_AT_NECK = 1


def count_params(model: torch.nn.Module) -> int:
    """Return total number of learnable parameters."""
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, x: torch.Tensor) -> int:
    """Count FLOPs using PyTorch's built-in FlopCounterMode."""
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(x)
    return flop_counter.get_total_flops()


def fmt_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.1f}K"


def fmt_flops(n: int) -> str:
    """Format FLOP count as human-readable string."""
    if n >= 1e12:
        return f"{n / 1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GFLOPs"
    return f"{n / 1e6:.1f} MFLOPs"


# ─── Model configs (block_cfg as LazyConfig) ─────────────────────────────────

MODELS = {
    "UNet-ConvNeXt": LazyConfig(ConvNeXtBlock)(
        n_spatial_dims=DATA_DIM,
    ),
    "UNet-Attention": LazyConfig(AttentionBlock)(
        n_spatial_dims=DATA_DIM,
        num_heads=6,
        mlp_ratio=4,
    ),
    "UNet-Hyena": LazyConfig(HyenaBlock)(
        n_spatial_dims=DATA_DIM,
        mlp_ratio=4,
        omega_0=30.0,
        siren_layers=3,
        siren_hidden_dim=64,
    ),
}


def build_model(block_cfg):
    """Build a UNet with the given block config and shared hyperparameters."""
    return UNet(
        dim_in=IN_CHANNELS,
        dim_out=OUT_CHANNELS,
        n_spatial_dims=DATA_DIM,
        spatial_resolution=SPATIAL_RESOLUTION,
        stages=STAGES,
        blocks_per_stage=BLOCKS_PER_STAGE,
        blocks_at_neck=BLOCKS_AT_NECK,
        init_features=INIT_FEATURES,
        block_cfg=block_cfg,
    )


def main():
    """Build all three UNet variants and print a FLOPs/params comparison table."""
    print("=" * 76)
    print("UNet Variant Comparison: euler_multi_quadrants_periodicBC (512x512)")
    print("=" * 76)
    print(
        f"  init_features={INIT_FEATURES}, stages={STAGES}, "
        f"blocks_per_stage={BLOCKS_PER_STAGE}, blocks_at_neck={BLOCKS_AT_NECK}"
    )
    print(f"  input: [{BATCH_SIZE}, {IN_CHANNELS}, {SPATIAL_RESOLUTION[0]}, {SPATIAL_RESOLUTION[1]}]")
    print()

    results = {}
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, *SPATIAL_RESOLUTION)

    for name, block_cfg in MODELS.items():
        print(f"Building {name}...")
        model = build_model(block_cfg)
        model.eval()
        params = count_params(model)

        with torch.no_grad():
            flops = count_flops(model, x)

        results[name] = {"params": params, "flops": flops}
        print(f"  Parameters: {fmt_params(params)}")
        print(f"  FLOPs:      {fmt_flops(flops)}")
        print()

    # ─── Comparison table ─────────────────────────────────────────────────
    ref = results["UNet-ConvNeXt"]
    print("-" * 76)
    print(f"{'Model':<20} {'Params':>12} {'vs ConvNeXt':>12} {'FLOPs':>16} {'vs ConvNeXt':>12}")
    print("-" * 76)
    for name, r in results.items():
        p_ratio = r["params"] / ref["params"]
        f_ratio = r["flops"] / ref["flops"] if ref["flops"] > 0 else float("inf")
        print(
            f"{name:<20} {fmt_params(r['params']):>12} {p_ratio:>11.2f}x {fmt_flops(r['flops']):>16} {f_ratio:>11.2f}x"
        )
    print("-" * 76)

    # ─── Per-stage resolution breakdown ───────────────────────────────────
    print()
    print("Per-stage spatial resolutions (encoder blocks operate BEFORE downsampling):")
    features = INIT_FEATURES
    for i in range(STAGES):
        res = SPATIAL_RESOLUTION[0] // (2**i)
        dim = features * 2**i
        tokens = res * res
        print(f"  Encoder stage {i}: {res}x{res} ({tokens:,} tokens), {dim} channels")
    neck_res = SPATIAL_RESOLUTION[0] // (2**STAGES)
    print(f"  Neck:           {neck_res}x{neck_res} ({neck_res**2:,} tokens), {features * 2**STAGES} channels")

    print()
    print("NOTE: Self-attention at 512x512 (262,144 tokens) is O(n^2) and")
    print("practically infeasible. Hyena's global conv is O(n log n) per stage.")


if __name__ == "__main__":
    main()
