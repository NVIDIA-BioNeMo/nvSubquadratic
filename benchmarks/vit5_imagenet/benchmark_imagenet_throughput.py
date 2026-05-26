"""Inference throughput benchmark for ImageNet-1k ViT-5 configs.

Reports images/sec on a single GPU, comparable to the throughput numbers in
VMamba Table 1 (https://arxiv.org/pdf/2401.10166).

This script builds the *network* directly from ``nvsubquadratic`` modules — it
does **not** load the full training-side config (which pulls in apex, DALI,
lightning, etc.).  The four architectures benchmarked match the runs from
``dwromero/blockdiag-kernel`` reported in the paper:

    * full_hyena_learnable_omega_blockdiag (12H, learnable-ω₀ + block-diag)
    * hhha_blockdiag                       (3:1 H:A)
    * attention_pretrain                   (pure attn baseline)
    * ha_blockdiag                         (1:1 H:A)

Constants (HIDDEN_DIM=384, NUM_BLOCKS=12, image_size=224, patch_size=16,
kernel hyperparameters, block-diag schedule, learnable-ω₀ clamp) are mirrored
from:

    examples/vit5_imagenet/v5/_base.py
    examples/vit5_imagenet/vit5_hybrid/_base_config.py
    examples/vit5_imagenet/vit5_hybrid/_blockdiag.py
    examples/vit5_imagenet/vit5_hybrid/_learnable_omega.py
    examples/vit5_imagenet/v5/attention_pretrain.py

Protocol (matches Swin / VMamba): 224x224 mock inputs, BF16 autocast,
``torch.no_grad`` + ``eval()``, optional ``torch.compile``, N warmup +
M timed iterations using cuda events.

Usage::

    python scripts/benchmark_imagenet_throughput.py
    python scripts/benchmark_imagenet_throughput.py --batch-size 4 --no-compile  # smoketest
    python scripts/benchmark_imagenet_throughput.py --batch-size 256 --dtype fp16
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Callable

import torch


# Ensure project root is on sys.path so nvsubquadratic.* is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import (
    BlockDiagonalLearnableOmegaSIRENKernelND,
    BlockDiagonalMultiOmegaSIRENKernelND,
    SIRENKernelND,
)
from nvsubquadratic.modules.masks_nd import (
    BlockAlignedGaussianModulationND,
    GaussianModulationND,
)
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import (
    trunc_normal_init,
    trunc_normal_init_factory,
)
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Mirrored constants ───────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4
NUM_HEADS = 6
NUM_REGISTERS = 4
DROP_PATH_RATE = 0.05  # ignored at inference (we eval()), kept for parity

# SIREN kernel hyperparameters
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# Block-diagonal schedule defaults (from _blockdiag.py)
BD_NUM_BLOCKS = 8
BD_OMEGA_0_MIN = 1.0
BD_OMEGA_0_MAX = 12.0
BD_SCHEDULE = "linear"
BD_OFF_BLOCK_SCALE = 0.1

# Learnable-ω₀ clamp + LR-scale (from _learnable_omega.py)
LO_SCALE_MIN = 1e-2
LO_SCALE_MAX = 2.0
LO_APPLY_LR_SCALE = True

INIT_FN = trunc_normal_init(std=0.02)
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


def _grid_h(num_patches_w: int, num_registers: int) -> int:
    """Replicates the _GRID_H interpolation expression for L_cache.

    Mirrors:
        ((W**2 + 1 + R + (-(W**2 + 1 + R) % W)) // W)
    where ``W = image_size // patch_size``.
    """
    W = num_patches_w
    R = num_registers
    pad = (-(W**2) - 1 - R) % W
    return (W**2 + 1 + R + pad) // W


# ─── Block builders ───────────────────────────────────────────────────────────


def _attention_block(num_patches_w: int, num_registers: int) -> LazyConfig:
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(ViT5Attention)(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            num_patches_h=num_patches_w,
            num_patches_w=num_patches_w,
            num_registers=num_registers,
            qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
            rope_base=10000.0,
            reg_rope_base=100.0,
            attn_dropout=0.0,
            proj_dropout=0.0,
            qkv_bias=False,
            out_proj_bias=False,
            init_fn_qkv_proj=INIT_FN,
            init_fn_out_proj=INIT_FN,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
    )


def _hyena_block(
    num_patches_w: int,
    num_registers: int,
    *,
    kernel_variant: str,  # "siren" | "blockdiag" | "blockdiag_learnable"
    fft_backend: str = "subq_ops",
) -> LazyConfig:
    L_cache = _grid_h(num_patches_w, num_registers)

    if kernel_variant == "siren":
        kernel_cfg = LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=HIDDEN_DIM,
            mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_layers=KERNEL_NUM_LAYERS,
            embedding_dim=KERNEL_EMBEDDING_DIM,
            omega_0=KERNEL_OMEGA_0,
            L_cache=L_cache,
            use_bias=True,
            hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
        )
        mask_cfg = LazyConfig(GaussianModulationND)(
            data_dim=2,
            num_channels=HIDDEN_DIM,
            min_attenuation_at_step=0.1,
            max_attenuation_at_limit=0.95,
            init_extent=1.0,
            parametrization="direct",
        )
    elif kernel_variant == "blockdiag":
        kernel_cfg = LazyConfig(BlockDiagonalMultiOmegaSIRENKernelND)(
            data_dim=2,
            out_dim=HIDDEN_DIM,
            mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_layers=KERNEL_NUM_LAYERS,
            embedding_dim=KERNEL_EMBEDDING_DIM,
            L_cache=L_cache,
            use_bias=True,
            hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
            num_blocks=BD_NUM_BLOCKS,
            omega_0_min=BD_OMEGA_0_MIN,
            omega_0_max=BD_OMEGA_0_MAX,
            schedule=BD_SCHEDULE,
            off_block_scale=BD_OFF_BLOCK_SCALE,
        )
        mask_cfg = LazyConfig(BlockAlignedGaussianModulationND)(
            data_dim=2,
            num_channels=HIDDEN_DIM,
            min_attenuation_at_step=0.1,
            max_attenuation_at_limit=0.95,
            init_extent=1.0,
            parametrization="direct",
        )
    elif kernel_variant == "blockdiag_learnable":
        kernel_cfg = LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
            data_dim=2,
            out_dim=HIDDEN_DIM,
            mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_layers=KERNEL_NUM_LAYERS,
            embedding_dim=KERNEL_EMBEDDING_DIM,
            L_cache=L_cache,
            use_bias=True,
            hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
            num_blocks=BD_NUM_BLOCKS,
            omega_0_min=BD_OMEGA_0_MIN,
            omega_0_max=BD_OMEGA_0_MAX,
            schedule=BD_SCHEDULE,
            off_block_scale=BD_OFF_BLOCK_SCALE,
            omega_0_scale_min=LO_SCALE_MIN,
            omega_0_scale_max=LO_SCALE_MAX,
            apply_lr_scale=LO_APPLY_LR_SCALE,
        )
        mask_cfg = LazyConfig(BlockAlignedGaussianModulationND)(
            data_dim=2,
            num_channels=HIDDEN_DIM,
            min_attenuation_at_step=0.1,
            max_attenuation_at_limit=0.95,
            init_extent=1.0,
            parametrization="direct",
        )
    else:
        raise ValueError(f"unknown kernel_variant: {kernel_variant}")

    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=kernel_cfg,
                mask_cfg=mask_cfg,
                grid_type="double",
                fft_padding="zero",
                fft_backend=fft_backend,
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=mixer_cfg,
            grid_w=num_patches_w,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM),
    )


# ─── Network builders ─────────────────────────────────────────────────────────


def build_hybrid(layer_pattern: str, kernel_variant: str, fft_backend: str = "subq_ops") -> LazyConfig:
    num_patches_w = IMAGE_SIZE // PATCH_SIZE
    layer_types: dict[str, LazyConfig] = {}
    if "A" in layer_pattern:
        layer_types["A"] = _attention_block(num_patches_w, NUM_REGISTERS)
    if "H" in layer_pattern:
        layer_types["H"] = _hyena_block(
            num_patches_w, NUM_REGISTERS, kernel_variant=kernel_variant, fft_backend=fft_backend
        )
    return LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=len(layer_pattern),
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        layer_pattern=layer_pattern,
        layer_types=layer_types,
        max_drop_path_rate=DROP_PATH_RATE,
        drop_path_schedule="constant",
    )


def build_attention_pretrain() -> LazyConfig:
    """Replicates examples/vit5_imagenet/v5/attention_pretrain.py."""
    num_patches_w = IMAGE_SIZE // PATCH_SIZE
    return LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=num_patches_w,
                num_patches_w=num_patches_w,
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
                out_proj_bias=False,
                init_fn_qkv_proj=INIT_FN,
                init_fn_out_proj=INIT_FN,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                bias=False,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=INIT_FN_FACTORY,
                init_method_out=INIT_FN_FACTORY,
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )


# Names + builders + whether the model uses FFT conv (Hyena).
# Builders take a single ``fft_backend`` arg so the CLI can pick between the
# native ``torch_fft`` path (works anywhere) and the optimized ``subq_ops``
# CUDA kernels (production).
MODELS: list[tuple[str, Callable[[str], LazyConfig], bool]] = [
    (
        "full_hyena_learnable_omega_blockdiag",
        lambda fft: build_hybrid("H" * 12, "blockdiag_learnable", fft_backend=fft),
        True,
    ),
    (
        "hhha_blockdiag (3H:1A)",
        lambda fft: build_hybrid("HHHA" * 3, "blockdiag", fft_backend=fft),
        True,
    ),
    (
        "attention_pretrain (pure attn)",
        lambda fft: build_attention_pretrain(),
        False,
    ),
    (
        "ha_blockdiag (1H:1A)",
        lambda fft: build_hybrid("HA" * 6, "blockdiag", fft_backend=fft),
        True,
    ),
]


def _make_inputs(batch_size: int, image_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """ViT5ClassificationNet expects input as [B, H, W, C] channels-last."""
    x = torch.randn(batch_size, image_size, image_size, 3, device=device, dtype=torch.float32)
    c = torch.zeros(batch_size, image_size, image_size, 0, device=device, dtype=torch.float32)
    return {"input": x, "condition": c}


def benchmark(
    name: str,
    builder: Callable[[str], LazyConfig],
    has_fft: bool,
    *,
    batch_size: int,
    image_size: int,
    dtype: torch.dtype,
    num_warmup: int,
    num_iters: int,
    compile_mode: str | None,
    device: torch.device,
    fft_backend: str,
) -> dict[str, float]:
    print(f"\n{'=' * 72}")
    print(f"[{name}]")
    print(f"{'=' * 72}", flush=True)

    if has_fft and compile_mode is not None:
        # FFT conv needs the real-valued complex multiply path under torch.compile.
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True
        print("[setup] FFT compile-compat enabled", flush=True)

    network = instantiate(builder(fft_backend)).to(device).eval()
    n_params = sum(p.numel() for p in network.parameters())
    print(f"[setup] params: {n_params:,}", flush=True)

    if compile_mode is not None:
        print(f"[setup] torch.compile(mode={compile_mode!r}) ...", flush=True)
        network = torch.compile(network, mode=compile_mode)

    inputs = _make_inputs(batch_size, image_size, device)
    torch.cuda.reset_peak_memory_stats(device)

    print(f"[warmup] {num_warmup} iters ...", flush=True)
    t_compile_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for _ in range(num_warmup):
            _ = network(inputs)
    torch.cuda.synchronize(device)
    compile_secs = time.perf_counter() - t_compile_start

    print(f"[timed] {num_iters} iters ...", flush=True)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for _ in range(num_iters):
            _ = network(inputs)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000.0
    imgs_per_sec = (batch_size * num_iters) / elapsed_s
    ms_per_batch = elapsed_ms / num_iters
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    print(
        f"[result] imgs/s = {imgs_per_sec:>9.1f}  |  "
        f"{ms_per_batch:6.2f} ms/batch  |  "
        f"peak mem = {peak_mem_gb:5.2f} GB  |  "
        f"warmup+compile = {compile_secs:6.1f}s",
        flush=True,
    )

    del network, inputs
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    return {
        "name": name,
        "params": n_params,
        "imgs_per_sec": imgs_per_sec,
        "ms_per_batch": ms_per_batch,
        "peak_mem_gb": peak_mem_gb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size (VMamba uses 128).")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Input spatial resolution.")
    parser.add_argument("--num-warmup", type=int, default=30, help="Warmup iterations (also covers compile).")
    parser.add_argument("--num-iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Autocast dtype (bf16 is the H100 default).",
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
        help="FFT conv backend for Hyena blocks. Use 'torch_fft' on hosts without the subq_ops CUDA kernels.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names to run (substring match). Default: all 4.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for throughput benchmark.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    compile_mode = None if args.no_compile else args.compile_mode

    selected = MODELS
    if args.models:
        selected = [m for m in MODELS if any(s in m[0] for s in args.models)]
        if not selected:
            raise SystemExit(f"No models match {args.models!r}")

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"Settings: batch_size={args.batch_size} image_size={args.image_size} "
        f"dtype={args.dtype} compile={compile_mode} fft_backend={args.fft_backend} "
        f"warmup={args.num_warmup} timed={args.num_iters}"
    )

    results: list[dict[str, float]] = []
    for name, builder, has_fft in selected:
        try:
            results.append(
                benchmark(
                    name,
                    builder,
                    has_fft,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    dtype=dtype,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                    compile_mode=compile_mode,
                    device=device,
                    fft_backend=args.fft_backend,
                )
            )
        except Exception as exc:
            print(f"[error] {name} failed: {exc!r}", flush=True)
            results.append(
                {
                    "name": name,
                    "params": 0,
                    "imgs_per_sec": float("nan"),
                    "ms_per_batch": float("nan"),
                    "peak_mem_gb": float("nan"),
                }
            )

    print(f"\n{'=' * 72}")
    print("Throughput summary  (higher imgs/s is better)")
    print(f"{'=' * 72}")
    print(f"{'Model':<40s} {'Params':>10s} {'imgs/s':>10s} {'ms/batch':>10s} {'GB':>6s}")
    print("-" * 80)
    for r in results:
        params_m = r["params"] / 1e6 if r["params"] else float("nan")
        print(
            f"{r['name']:<40s} "
            f"{params_m:>9.1f}M "
            f"{r['imgs_per_sec']:>10.1f} "
            f"{r['ms_per_batch']:>10.2f} "
            f"{r['peak_mem_gb']:>6.2f}"
        )


if __name__ == "__main__":
    main()
