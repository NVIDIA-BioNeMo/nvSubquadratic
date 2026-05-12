"""Forward-time vs patch-size benchmark for the 2D residual-net mixers.

Times the forward pass of a small 2D ``ResidualNetwork`` with the QKV
sequence mixer swapped between attention, Hyena, and Mamba2.  For each
patch size ``p`` in ``--patch-sizes`` we feed an input image of shape
``(batch, H//p, W//p, in_channels)`` — i.e. patching reduces the
per-axis token count by ``p``, so the total sequence length scales as
``p^-2``.

Mirrors the configuration of the EMNIST spatial-recall-2D mask-selection
XS configs:

    examples/spatial_recall_2d/emnist_regression_mask_selection/ccnn_attn_xs.py
    examples/spatial_recall_2d/emnist_regression_mask_selection/ccnn_hyena_xs.py
    examples/spatial_recall_2d/emnist_regression_mask_selection/ccnn_mamba_xs.py

Default canvas (H=W=64) is the same as those configs; ``hidden_dim``
is taken per-mixer to match (160 for attention/Hyena, 96 for Mamba).

Local sanity-check (RTX-class GPU)::

    python scripts/benchmark_patch_size_2d.py \\
        --batch-size 1 --no-compile --fft-backend torch_fft \\
        --patch-sizes 2 4 8 --num-warmup 2 --num-iters 5

H100 production run (matching the throughput script)::

    python scripts/benchmark_patch_size_2d.py
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Callable

import torch


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import (
    partial_wang_init_fn_with_num_layers,
    small_init,
)


# ─── Mirrored constants (spatial_recall_2d XS) ────────────────────────────────
DATA_DIM = 2
NUM_BLOCKS = 4
IN_CHANNELS = 2
OUT_CHANNELS = 1
CANVAS_SIZE = 64

# Per-mixer XS hidden_dim (matches the *_xs.py configs)
HIDDEN_DIM_ATTN = 160
HIDDEN_DIM_HYENA = 160
HIDDEN_DIM_MAMBA = 96

# SIREN kernel hyperparameters (mixer_defaults.get_hyena_mixer_cfg defaults)
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# Attention defaults (ccnn_attn_xs)
ATTN_NUM_HEADS = 8
# RoPE is disabled here: it would need ``rope_spatial_dims`` baked in at
# construction time, but the patch sweep changes spatial dims at every
# point.  Step-time impact of RoPE itself is small relative to the QKV
# matmul + attention cost, so this is a fair stand-in.
ATTN_USE_ROPE = False

# Mamba defaults (ccnn_mamba_xs)
MAMBA_HEADDIM = 32
MAMBA_EXPAND = 2
MAMBA_BIDIRECTIONAL = True


# ─── Mixer config builders (concrete — no OmegaConf interpolation) ────────────


def _hyena_mixer_cfg(hidden_dim: int, fft_backend: str) -> LazyConfig:
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=hidden_dim,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=DATA_DIM,
                hidden_dim=hidden_dim,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=DATA_DIM,
                    out_dim=hidden_dim,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=CANVAS_SIZE,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
                fft_backend=fft_backend,
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * hidden_dim,
                out_channels=3 * hidden_dim,
                kernel_size=3,
                groups=3 * hidden_dim,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            qk_norm_cfg=None,
            use_rope=False,
            rope_base=10000.0,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )


def _attention_mixer_cfg(hidden_dim: int) -> LazyConfig:
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=hidden_dim,
        mixer_cfg=LazyConfig(Attention)(
            hidden_dim=hidden_dim,
            num_heads=ATTN_NUM_HEADS,
            apply_qk_norm=True,
            use_rope=ATTN_USE_ROPE,
            is_causal=False,
            rope_base=10000.0,
            attn_dropout=0.0,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )


def _mamba_mixer_cfg(hidden_dim: int) -> LazyConfig:
    from mamba_ssm import Mamba2

    from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

    return LazyConfig(MambaNDMixer)(
        mamba_layer_cfg=LazyConfig(Mamba2)(
            d_model=hidden_dim,
            headdim=MAMBA_HEADDIM,
            expand=MAMBA_EXPAND,
        ),
        bidirectional=MAMBA_BIDIRECTIONAL,
    )


# ─── Network builder ──────────────────────────────────────────────────────────


def build_network(mixer_cfg: LazyConfig, hidden_dim: int) -> LazyConfig:
    """Replicates spatial_recall_2d/base_config.base_experiment_config network."""
    return LazyConfig(ResidualNetwork)(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=hidden_dim,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=IN_CHANNELS, out_features=hidden_dim),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=hidden_dim, out_features=OUT_CHANNELS),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=mixer_cfg,
            sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=hidden_dim,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=hidden_dim),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        target_size=None,
    )


# ─── Mixer registry ───────────────────────────────────────────────────────────

MIXERS: dict[str, tuple[int, Callable[[str], LazyConfig], bool]] = {
    # name → (hidden_dim, mixer_cfg_builder(fft_backend), needs_fft_backend)
    "attention": (HIDDEN_DIM_ATTN, lambda fft: _attention_mixer_cfg(HIDDEN_DIM_ATTN), False),
    "hyena": (HIDDEN_DIM_HYENA, lambda fft: _hyena_mixer_cfg(HIDDEN_DIM_HYENA, fft), True),
    "mamba": (HIDDEN_DIM_MAMBA, lambda fft: _mamba_mixer_cfg(HIDDEN_DIM_MAMBA), False),
}


# ─── Benchmarking ─────────────────────────────────────────────────────────────


def _input_shape_for(patch: int) -> tuple[int, int]:
    """Return (H, W) after patching by ``patch`` along every spatial axis."""
    if CANVAS_SIZE % patch != 0:
        raise ValueError(
            f"patch_size={patch} does not evenly divide canvas (H=W={CANVAS_SIZE})."
        )
    return (CANVAS_SIZE // patch, CANVAS_SIZE // patch)


def _make_inputs(batch: int, patch: int, device: torch.device) -> dict[str, torch.Tensor]:
    H, W = _input_shape_for(patch)
    x = torch.randn(batch, H, W, IN_CHANNELS, device=device, dtype=torch.float32)
    c = torch.zeros(batch, H, W, 0, device=device, dtype=torch.float32)
    return {"input": x, "condition": c}


def time_forward(
    mixer_name: str,
    patch: int,
    *,
    batch_size: int,
    dtype: torch.dtype,
    num_warmup: int,
    num_iters: int,
    compile_mode: str | None,
    fft_backend: str,
    device: torch.device,
) -> tuple[float, float]:
    """Returns ``(ms_per_batch, peak_mem_gb)``."""
    hidden_dim, build_mixer_cfg, has_fft = MIXERS[mixer_name]

    if has_fft and compile_mode is not None:
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True

    network = instantiate(build_network(build_mixer_cfg(fft_backend), hidden_dim)).to(device).eval()

    if compile_mode is not None:
        network = torch.compile(network, mode=compile_mode)

    inputs = _make_inputs(batch_size, patch, device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for _ in range(num_warmup):
            _ = network(inputs)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for _ in range(num_iters):
            _ = network(inputs)
    end.record()
    torch.cuda.synchronize(device)

    ms_per_batch = start.elapsed_time(end) / num_iters
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    del network, inputs
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    return ms_per_batch, peak_mem_gb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--patch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Patch sizes to sweep. Each axis is divided by the patch size, "
        "so total seq_len scales as p^-2 in 2D.",
    )
    parser.add_argument(
        "--mixers",
        nargs="+",
        choices=sorted(MIXERS.keys()),
        default=sorted(MIXERS.keys()),
        help="Subset of mixers to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the timed forward pass.")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations (also covers compile).")
    parser.add_argument("--num-iters", type=int, default=30, help="Timed iterations.")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Autocast dtype.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode (matches the training configs). Use --no-compile to disable.",
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile entirely.")
    parser.add_argument(
        "--fft-backend",
        choices=["torch_fft", "subq_ops"],
        default="subq_ops",
        help="FFT conv backend for Hyena. Use 'torch_fft' on hosts without the subq_ops CUDA kernels.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="patch_size_step_time.png",
        help="Output figure path (PNG).",
    )
    parser.add_argument(
        "--normalize",
        choices=["per-mixer-min", "global-min", "none"],
        default="per-mixer-min",
        help="How to normalize step time for the relative-time plot. "
        "'per-mixer-min' divides each mixer by its own min time (rightmost point = 1). "
        "'global-min' divides every series by the global min. "
        "'none' plots absolute ms.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for forward-time benchmark.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    compile_mode = None if args.no_compile else args.compile_mode

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Settings: batch_size={args.batch_size} dtype={args.dtype} compile={compile_mode} "
        f"fft_backend={args.fft_backend} warmup={args.num_warmup} timed={args.num_iters}"
    )
    print(f"Mixers: {args.mixers}    Patch sizes: {args.patch_sizes}")

    # Validate patch sizes against canvas
    for p in args.patch_sizes:
        try:
            _input_shape_for(p)
        except ValueError as e:
            raise SystemExit(str(e))

    results: dict[str, dict[int, dict[str, float]]] = {m: {} for m in args.mixers}

    for mixer in args.mixers:
        for patch in args.patch_sizes:
            H, W = _input_shape_for(patch)
            seq_len = H * W
            label = f"[{mixer:>9s}  patch={patch}  shape=({H},{W})  N={seq_len}]"
            print(f"\n{label}", flush=True)
            t0 = time.perf_counter()
            try:
                ms, mem = time_forward(
                    mixer,
                    patch,
                    batch_size=args.batch_size,
                    dtype=dtype,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                    compile_mode=compile_mode,
                    fft_backend=args.fft_backend,
                    device=device,
                )
                wall = time.perf_counter() - t0
                print(
                    f"   ms/batch = {ms:8.3f}  |  peak mem = {mem:5.2f} GB  |  wall = {wall:5.1f}s",
                    flush=True,
                )
                results[mixer][patch] = {"ms": ms, "mem_gb": mem}
            except Exception as exc:
                print(f"   [error] {exc!r}", flush=True)
                results[mixer][patch] = {"ms": float("nan"), "mem_gb": float("nan")}

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Forward-time summary (lower is better, ms per batch)")
    print(f"{'=' * 72}")
    header = f"{'Patch':>6s}  " + "  ".join(f"{m:>14s}" for m in args.mixers)
    print(header)
    print("-" * len(header))
    for p in args.patch_sizes:
        row = f"{p:>6d}  "
        for m in args.mixers:
            ms = results[m].get(p, {}).get("ms", float("nan"))
            row += f"{ms:>14.3f}  "
        print(row.rstrip())

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not available — skipping figure.", flush=True)
        return

    if args.normalize == "global-min":
        all_ms = [results[m][p]["ms"] for m in args.mixers for p in args.patch_sizes]
        finite = [v for v in all_ms if v == v]  # NaN-safe
        norm = min(finite) if finite else 1.0
        ylabel = "Step Time (relative to global min)"
    elif args.normalize == "per-mixer-min":
        norm = None  # per-curve
        ylabel = "Step Time (relative)"
    else:
        norm = 1.0
        ylabel = "Step Time (ms / batch)"

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    markers = {"attention": "s", "hyena": "o", "mamba": "^"}
    linestyles = {"attention": "--", "hyena": "-", "mamba": ":"}

    for m in args.mixers:
        xs = args.patch_sizes
        ys = [results[m][p]["ms"] for p in xs]
        if args.normalize == "per-mixer-min":
            finite = [v for v in ys if v == v]
            denom = min(finite) if finite else 1.0
            ys = [v / denom for v in ys]
        elif args.normalize == "global-min":
            ys = [v / norm for v in ys]
        ax.plot(
            xs,
            ys,
            marker=markers.get(m, "o"),
            linestyle=linestyles.get(m, "-"),
            label=m.capitalize(),
        )

    ax.set_xlabel("Patch Size")
    ax.set_ylabel(ylabel)
    ax.set_title("Step Time vs. Patch Size (2D residual-net, mixer ablation)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.patch_sizes)
    ax.set_xticklabels([str(p) for p in args.patch_sizes])
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=150)
    print(f"\n[plot] saved {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
