r"""End-to-end throughput benchmark: Attention vs Hyena vs Hyena-FiLM ViT-5-Small.

Measures forward+backward throughput (samples/sec, ms/step) and peak GPU
memory for three ViT-5-Small sequence-mixer variants:

  1. **Attention** — standard multi-head self-attention (reference baseline).
  2. **Hyena** — gated Hyena-2D with shared SIREN kernel (no FiLM).
  3. **Hyena-FiLM** — gated Hyena-2D with per-sample FiLM-conditioned SIREN kernel.

For the Hyena variants the FFT convolution backend can be swapped between
``torch.fft`` (the default) and ``subquadratic_ops_torch`` (custom CUDA
kernel) via monkey-patching — no production code is modified.

Each configuration is tested in *eager* and *torch.compile* modes, giving
10 rows in total (see ``BENCH_CONFIGS``).

Usage (requires GPU — run inside SLURM with the project container)::

    srun --gres=gpu:1 -c 16 --partition low \
        --container-image=/shared/images/nvsubquadratic_cuda129.sqsh \
        --container-mounts=/home/dwromero:/home/dwromero,/shared:/shared \
        bash -c "source /home/dwromero/miniconda3/etc/profile.d/conda.sh && \
                 conda activate nv-subq && \
                 cd /home/dwromero/projects/nvSubquadratic-private && \
                 python benchmarks/vit5_imagenet/bench_vit5_hyena.py"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, ".")

import nvsubquadratic.ops.fftconv as _fftconv
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ──────────────────────────────────────────────────────────────────────────────
# Architecture constants (ViT-5-Small, CLS-row layout)
# ──────────────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCHES_H = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_REGISTERS = NUM_PATCHES_W - 1  # 13 — fills CLS row: [CLS, regs, patches] -> 15x14

KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0
FILM_HIDDEN_DIM = 64

MLP_RATIO = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


# ──────────────────────────────────────────────────────────────────────────────
# subquadratic_ops_torch availability
# ──────────────────────────────────────────────────────────────────────────────
def _has_subq_ops() -> bool:
    try:
        from subquadratic_ops_torch.fft_conv2d import fft_conv2d  # noqa: F401

        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────


def _shared_block_kwargs() -> dict:
    """Return the kwargs shared by every ViT5ResidualBlock across all mixers."""
    return {
        "sequence_mixer_norm_cfg": LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        "mlp_cfg": LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        "mlp_norm_cfg": LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        "hidden_dim": HIDDEN_DIM,
        "layer_scale_init": LAYER_SCALE_INIT,
        "drop_path_rate": DROP_PATH_RATE,
    }


ATTN_NUM_REGISTERS = 4  # attention uses fewer registers (no CLS-row)


def build_attention() -> nn.Module:
    """Build ViT-5-Small with multi-head self-attention (production config)."""
    return ViT5ClassificationNet(
        in_channels=3,
        num_classes=1000,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=ATTN_NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=NUM_PATCHES_H,
                num_patches_w=NUM_PATCHES_W,
                num_registers=ATTN_NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
            ),
            **_shared_block_kwargs(),
        ),
    ).cuda()


def _hyena_mixer_cfg(*, film: bool) -> LazyConfig:
    """Return a QKVSequenceMixer config wrapping Hyena (optionally with FiLM)."""
    kernel_kw: dict = {
        "data_dim": 2,
        "out_dim": HIDDEN_DIM,
        "mlp_hidden_dim": KERNEL_MLP_HIDDEN_DIM,
        "num_layers": KERNEL_NUM_LAYERS,
        "embedding_dim": KERNEL_EMBEDDING_DIM,
        "omega_0": KERNEL_OMEGA_0,
        "L_cache": NUM_PATCHES_H + 1,
        "use_bias": True,
        "hidden_omega_0": KERNEL_HIDDEN_OMEGA_0,
    }
    if film:
        kernel_kw["film_cfg"] = LazyConfig(KernelFiLMGenerator)(
            cond_dim=HIDDEN_DIM,
            kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=KERNEL_NUM_LAYERS - 1,
            film_hidden_dim=FILM_HIDDEN_DIM,
        )

    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(**kernel_kw),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
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
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )


def build_hyena(*, film: bool = False) -> nn.Module:
    """Build ViT-5-Small with Hyena sequence mixer (optionally FiLM-conditioned)."""
    block_kw = _shared_block_kwargs()
    block_kw["sequence_mixer_cfg"] = LazyConfig(ViT5HyenaAdapter)(
        inner_mixer_cfg=_hyena_mixer_cfg(film=film),
        grid_w=NUM_PATCHES_W,
    )
    if film:
        block_kw["register_pooling_cfg"] = LazyConfig(RegisterPooling)(num_registers=NUM_REGISTERS)
        block_kw["num_registers"] = NUM_REGISTERS

    return ViT5ClassificationNet(
        in_channels=3,
        num_classes=1000,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(**block_kw),
    ).cuda()


# ──────────────────────────────────────────────────────────────────────────────
# Monkey-patching helpers
# ──────────────────────────────────────────────────────────────────────────────
def _align_dtypes(fn):
    """Wrap an fftconv function so that x, kernel, shortcut share the same dtype.

    Under ``torch.amp.autocast`` certain ops (SiLU, Sigmoid) promote tensors
    to fp32 while others (Linear) cast to bf16, causing a dtype mismatch that
    the fftconv assertions reject.  This wrapper aligns everything to ``x.dtype``
    before forwarding.
    """

    def wrapper(x, kernel, shortcut=None):
        kernel = kernel.to(x.dtype)
        if shortcut is not None:
            shortcut = shortcut.to(x.dtype)
        return fn(x, kernel, shortcut)

    return wrapper


def patch_model_align_dtypes(model: nn.Module) -> nn.Module:
    """Wrap CKConvND fftconv functions with dtype alignment (in-place)."""
    for module in model.modules():
        if isinstance(module, CKConvND):
            module.fftconv_fn_bhl_input = _align_dtypes(module.fftconv_fn_bhl_input)
            module.fftconv_fn = _align_dtypes(module.fftconv_fn)
    return model


def _make_subq_adapter():
    """Return a drop-in replacement for ``fftconv2d_fp32_bhl`` using subq_ops."""
    from subquadratic_ops_torch.fft_conv2d import fft_conv2d

    def adapter(x, kernel, shortcut=None):
        _B, H, _X, _Y = x.shape
        input_dtype = x.dtype

        # subq_ops only supports fp32
        x_fp32 = x.float()
        kernel_fp32 = kernel.float()

        # subq_ops expects [H, Kx, Ky] for shared, [B, H, Kx, Ky] for FiLM
        k = kernel_fp32.squeeze(0) if kernel_fp32.shape[0] == 1 else kernel_fp32
        y = fft_conv2d(x_fp32.contiguous(), k.contiguous())

        y = y.to(input_dtype)

        if shortcut is not None:
            y = y + shortcut.to(input_dtype).view(1, H, 1, 1) * x

        return y

    return adapter


def patch_model_subq(model: nn.Module) -> nn.Module:
    """Replace all CKConvND FFT functions with the subq_ops adapter (in-place)."""
    adapter = _make_subq_adapter()
    for module in model.modules():
        if isinstance(module, CKConvND):
            module.fftconv_fn_bhl_input = adapter
            module.fftconv_fn = adapter
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ──────────────────────────────────────────────────────────────────────────────
def benchmark(
    model: nn.Module,
    batch_size: int,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> tuple[float, float, float]:
    """Time forward + backward and measure peak memory.

    Returns:
        (ms_per_step, samples_per_sec, peak_memory_mb)
    """
    x = torch.randn(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3, device="cuda", dtype=torch.bfloat16)
    inp = {"input": x, "condition": None}
    target = torch.randint(0, 1000, (batch_size,), device="cuda")

    # Warmup
    for _ in range(num_warmup):
        out = model(inp)
        loss = F.cross_entropy(out["logits"], target)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Timed iterations using CUDA events
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        out = model(inp)
        loss = F.cross_entropy(out["logits"], target)
        loss.backward()
        model.zero_grad(set_to_none=True)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    ms_per_step = elapsed_ms / num_iters
    samples_per_sec = batch_size * num_iters / (elapsed_ms / 1000)

    return ms_per_step, samples_per_sec, peak_mem_mb


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark configurations
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BenchConfig:
    """One row in the benchmark matrix."""

    label: str
    model_name: str  # "attn", "hyena", "hyena-film"
    backend: str  # "torch.fft", "subq_ops", "-"
    compile_mode: str | None  # None (eager), "default", "max-autotune"
    needs_subq: bool = False
    compile_compat_fft: bool = False


BENCH_CONFIGS = [
    # ── Attention ──
    BenchConfig("attn-eager", "attn", "-", None),
    BenchConfig("attn-compiled", "attn", "-", "default"),
    # ── Hyena (shared kernel) ──
    BenchConfig("hyena-eager", "hyena", "torch.fft", None),
    BenchConfig("hyena-compiled", "hyena", "torch.fft", "default"),
    BenchConfig("hyena-subq", "hyena", "subq_ops", None, needs_subq=True),
    BenchConfig("hyena-subq-compiled", "hyena", "subq_ops", "default", needs_subq=True),
    # ── Hyena-FiLM (per-sample kernel) ──
    BenchConfig("hyena-film-eager", "hyena-film", "torch.fft", None),
    BenchConfig("hyena-film-compiled", "hyena-film", "torch.fft", "default", compile_compat_fft=True),
    BenchConfig("hyena-film-subq", "hyena-film", "subq_ops", None, needs_subq=True),
    BenchConfig("hyena-film-subq-compiled", "hyena-film", "subq_ops", "default", needs_subq=True),
]


def _build_model(cfg: BenchConfig) -> nn.Module:
    """Construct the model for a given benchmark configuration."""
    if cfg.model_name == "attn":
        return build_attention()
    film = cfg.model_name == "hyena-film"
    model = build_hyena(film=film)
    if cfg.needs_subq:
        patch_model_subq(model)
    else:
        # Under autocast, SiLU/Sigmoid promote x to fp32 while SIREN kernel
        # stays bf16 → align dtypes before the fftconv assertion.
        patch_model_align_dtypes(model)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Run all benchmark configurations and print a summary table."""
    parser = argparse.ArgumentParser(description="ViT-5-Small end-to-end throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Subset of config labels to run (default: all). E.g. --configs hyena-eager hyena-subq",
    )
    args = parser.parse_args()

    has_subq = _has_subq_ops()

    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"subq_ops: {'available' if has_subq else 'NOT INSTALLED'}")
    print(f"Batch:    {args.batch_size}  |  Warmup: {args.warmup}  |  Iters: {args.iters}")
    print()

    configs = BENCH_CONFIGS
    if args.configs:
        configs = [c for c in configs if c.label in args.configs]

    results: list[tuple[str, str, str, float, float, float]] = []

    for cfg in configs:
        if cfg.needs_subq and not has_subq:
            print(f"[SKIP] {cfg.label:<30s}  (subquadratic_ops_torch not installed)")
            continue

        print(f"[RUN]  {cfg.label:<30s}  ... ", end="", flush=True)

        # Only FiLM + compile + torch.fft needs the real-valued complex multiply
        _fftconv.COMPILE_COMPATIBLE = cfg.compile_compat_fft

        model = _build_model(cfg)
        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        if cfg.compile_mode is not None:
            compile_kw = {} if cfg.compile_mode == "default" else {"mode": cfg.compile_mode}
            model = torch.compile(model, **compile_kw)
            warmup = max(args.warmup, 20)  # compiled models need more warmup
        else:
            warmup = args.warmup

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ms, sps, mem = benchmark(model, args.batch_size, warmup, args.iters)

        compile_label = cfg.compile_mode or "eager"
        results.append((cfg.label, cfg.backend, compile_label, ms, sps, mem))
        print(f"{ms:>7.1f} ms/step  |  {sps:>7.0f} samples/s  |  {mem:>8.0f} MB  ({num_params:.1f}M params)")

        del model
        torch.cuda.empty_cache()

    # ── Summary table ──
    print()
    print("=" * 100)
    print(f"{'Label':<30s} {'Backend':<12s} {'Compile':<14s} {'ms/step':>8s} {'samples/s':>10s} {'peak_MB':>9s}")
    print("-" * 100)
    for label, backend, compile_label, ms, sps, mem in results:
        print(f"{label:<30s} {backend:<12s} {compile_label:<14s} {ms:>8.1f} {sps:>10.0f} {mem:>9.0f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
