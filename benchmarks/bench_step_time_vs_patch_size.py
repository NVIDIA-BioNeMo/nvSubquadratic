#!/usr/bin/env python3
"""Benchmark wall-clock step time vs. patch size for Hyena, Attention, and Mamba.

Runs two setups side-by-side:
  • ViT-5-Small  (hidden_dim=384, 12 blocks, ViT5ClassificationNet)
  • ResNet       (hidden_dim=256,  4 blocks, ResidualNetwork)

Attention is compiled with torch.compile (max-autotune-no-cudagraphs).
Hyena uses torch_fft (compile-compatible); Mamba runs eager (Triton kernels).
Each (model, patch_size) pair runs in an isolated subprocess to avoid Triton
memory corruption after OOM events.

Usage:
    python benchmarks/bench_step_time_vs_patch_size.py
    python benchmarks/bench_step_time_vs_patch_size.py --out paper/figures/step_time_vs_patch_size.pdf
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

PATCH_SIZES   = [1, 2, 4, 8, 16]
WARMUP_STEPS  = 15   # first warmup step triggers torch.compile; allow extra margin
MEASURE_STEPS = 20
COMPILE_MODE  = "max-autotune-no-cudagraphs"


# ═══════════════════════════════════════════════════════════════════════════════
# ViT-5-Small setup  (mirrors examples/vit5_imagenet/v5_patch)
# ═══════════════════════════════════════════════════════════════════════════════

VIT5_IMAGE_SIZE   = 256
VIT5_IN_CHANNELS  = 3
VIT5_NUM_CLASSES  = 1000
VIT5_HIDDEN       = 384
VIT5_BLOCKS       = 12
VIT5_HEADS        = 6
VIT5_HEAD_DIM     = VIT5_HIDDEN // VIT5_HEADS   # 64
VIT5_REGISTERS    = 4
VIT5_LAYER_SCALE  = 1e-4
VIT5_DROP_PATH    = 0.05
VIT5_MLP_RATIO    = 4

VIT5_KERNEL_HIDDEN  = 32
VIT5_KERNEL_LAYERS  = 3
VIT5_KERNEL_EMBED   = 32
VIT5_OMEGA_0        = 10.0
VIT5_HIDDEN_W0      = 1.0
VIT5_FILM_HIDDEN    = 64

VIT5_PATCH_BATCH = {1: 1, 2: 2, 4: 4, 8: 16, 16: 64}


def _vit5_init():
    from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory
    return trunc_normal_init(std=0.02), trunc_normal_init_factory(std=0.02)


def _vit5_block_cfg(mixer_cfg, **extra):
    import torch
    from nvsubquadratic.lazy_config import LazyConfig
    from nvsubquadratic.modules.mlp import MLP
    from nvsubquadratic.modules.rms_norm import RMSNorm
    from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
    _, INIT_FF = _vit5_init()
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=mixer_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=VIT5_HIDDEN, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=VIT5_HIDDEN, activation="gelu",
            expansion_factor=float(VIT5_MLP_RATIO), bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FF, init_method_out=INIT_FF,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=VIT5_HIDDEN, eps=1e-6),
        hidden_dim=VIT5_HIDDEN, layer_scale_init=VIT5_LAYER_SCALE,
        drop_path_rate=VIT5_DROP_PATH, **extra,
    )


def _vit5_net_cfg(block_cfg, patch_size, **extra):
    import torch
    from nvsubquadratic.lazy_config import LazyConfig
    from nvsubquadratic.modules.rms_norm import RMSNorm
    from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
    return LazyConfig(ViT5ClassificationNet)(
        in_channels=VIT5_IN_CHANNELS, num_classes=VIT5_NUM_CLASSES,
        hidden_dim=VIT5_HIDDEN, num_blocks=VIT5_BLOCKS,
        patch_size=patch_size, image_size=VIT5_IMAGE_SIZE,
        num_registers=VIT5_REGISTERS, dropout_rate=0.0, readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim=VIT5_HIDDEN, eps=1e-6),
        block_cfg=block_cfg, **extra,
    )


def build_vit5_hyena(patch_size: int) -> torch.nn.Module:
    import torch
    from nvsubquadratic.lazy_config import LazyConfig, instantiate
    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
    from nvsubquadratic.modules.grn import GlobalResponseNorm
    from nvsubquadratic.modules.hyena_nd import Hyena
    from nvsubquadratic.modules.kernels_nd import SIRENKernelND
    from nvsubquadratic.modules.rms_norm import RMSNorm
    from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
    from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
    from nvsubquadratic.utils.qk_norm import L2Norm
    INIT_FN, INIT_FF = _vit5_init()

    grid = VIT5_IMAGE_SIZE // patch_size
    film_cfg = LazyConfig(KernelFiLMGenerator)(
        cond_dim=VIT5_HIDDEN, kernel_hidden_dim=VIT5_KERNEL_HIDDEN,
        num_film_layers=VIT5_KERNEL_LAYERS - 1, film_hidden_dim=VIT5_FILM_HIDDEN,
    )
    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=VIT5_HIDDEN,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2, hidden_dim=VIT5_HIDDEN,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2, out_dim=VIT5_HIDDEN,
                    mlp_hidden_dim=VIT5_KERNEL_HIDDEN, num_layers=VIT5_KERNEL_LAYERS,
                    embedding_dim=VIT5_KERNEL_EMBED, omega_0=VIT5_OMEGA_0,
                    L_cache=grid + 1, use_bias=True,
                    hidden_omega_0=VIT5_HIDDEN_W0, film_cfg=film_cfg,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double", fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * VIT5_HIDDEN, out_channels=3 * VIT5_HIDDEN,
                kernel_size=3, groups=3 * VIT5_HIDDEN, padding=1, bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=VIT5_HIDDEN, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=VIT5_HIDDEN, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False, out_proj_bias=False,
        init_method_in=INIT_FF, init_method_out=INIT_FF,
    )
    block = _vit5_block_cfg(
        LazyConfig(ViT5HyenaAdapter)(inner_mixer_cfg=mixer_cfg, grid_w=grid),
        register_pooling_cfg=LazyConfig(RegisterPooling)(num_registers=VIT5_REGISTERS),
        num_registers=VIT5_REGISTERS, register_start_idx=1,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=VIT5_HIDDEN),
    )
    return instantiate(_vit5_net_cfg(block, patch_size, prepend_registers=True))


def build_vit5_attention(patch_size: int) -> torch.nn.Module:
    import torch
    from nvsubquadratic.lazy_config import LazyConfig, instantiate
    from nvsubquadratic.modules.rms_norm import RMSNorm
    from nvsubquadratic.modules.vit5_attention import ViT5Attention
    INIT_FN, _ = _vit5_init()

    grid = VIT5_IMAGE_SIZE // patch_size
    mixer_cfg = LazyConfig(ViT5Attention)(
        hidden_dim=VIT5_HIDDEN, num_heads=VIT5_HEADS,
        num_patches_h=grid, num_patches_w=grid,
        num_registers=VIT5_REGISTERS,
        qk_norm=LazyConfig(RMSNorm)(dim=VIT5_HEAD_DIM, eps=1e-6),
        rope_base=10000.0, reg_rope_base=100.0,
        attn_dropout=0.0, proj_dropout=0.0,
        qkv_bias=False, out_proj_bias=False,
        init_fn_qkv_proj=INIT_FN, init_fn_out_proj=INIT_FN,
    )
    return instantiate(_vit5_net_cfg(_vit5_block_cfg(mixer_cfg), patch_size))


def build_vit5_mamba(patch_size: int) -> torch.nn.Module:
    import torch
    from mamba_ssm import Mamba2
    from nvsubquadratic.lazy_config import LazyConfig, instantiate
    from nvsubquadratic.modules.grn import GlobalResponseNorm
    from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer
    from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter

    grid = VIT5_IMAGE_SIZE // patch_size
    mamba_cfg = LazyConfig(MambaNDMixer)(
        mamba_layer_cfg=LazyConfig(Mamba2)(
            d_model=VIT5_HIDDEN, headdim=VIT5_HEAD_DIM, expand=2,
        ),
        bidirectional=True,
    )
    block = _vit5_block_cfg(
        LazyConfig(ViT5HyenaAdapter)(inner_mixer_cfg=mamba_cfg, grid_w=grid),
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=VIT5_HIDDEN),
    )
    return instantiate(_vit5_net_cfg(block, patch_size, prepend_registers=True))


# ═══════════════════════════════════════════════════════════════════════════════
# ResNet setup  (mirrors examples/spatial_recall_2d)
# ═══════════════════════════════════════════════════════════════════════════════

RESNET_CANVAS     = 256
RESNET_IN_CH      = 3
RESNET_OUT_CH     = 3
RESNET_HIDDEN     = 256
RESNET_BLOCKS     = 4
RESNET_HEADS      = 8
RESNET_DATA_DIM   = 2

RESNET_PATCH_BATCH = {1: 1, 2: 2, 4: 8, 8: 32, 16: 128}


def _resnet_net_cfg(mixer_cfg, patch_size: int):
    import torch
    from nvsubquadratic.lazy_config import LazyConfig
    from nvsubquadratic.modules.mlp import MLP
    from nvsubquadratic.modules.patchify import Patchify, Unpatchify
    from nvsubquadratic.modules.residual_block import ResidualBlock
    from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
    from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init

    norm = LazyConfig(torch.nn.LayerNorm)(normalized_shape=RESNET_HIDDEN)
    block_cfg = LazyConfig(ResidualBlock)(
        sequence_mixer_cfg=mixer_cfg,
        sequence_mixer_norm_cfg=norm,
        condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
        condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
        mlp_cfg=LazyConfig(MLP)(
            dim=RESNET_HIDDEN, activation="glu", expansion_factor=1.0,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=small_init,
            init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=RESNET_BLOCKS),
        ),
        mlp_norm_cfg=norm,
        dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )

    if patch_size == 1:
        in_proj  = LazyConfig(torch.nn.Linear)(in_features=RESNET_IN_CH,  out_features=RESNET_HIDDEN)
        out_proj = LazyConfig(torch.nn.Linear)(in_features=RESNET_HIDDEN, out_features=RESNET_OUT_CH)
    else:
        in_proj  = LazyConfig(Patchify)(
            in_features=RESNET_IN_CH, out_features=RESNET_HIDDEN,
            data_dim=RESNET_DATA_DIM, patch_size=patch_size, stride=patch_size,
        )
        out_proj = LazyConfig(Unpatchify)(
            in_features=RESNET_HIDDEN, out_features=RESNET_OUT_CH,
            data_dim=RESNET_DATA_DIM, patch_size=patch_size, stride=patch_size,
        )

    return LazyConfig(ResidualNetwork)(
        in_channels=RESNET_IN_CH, out_channels=RESNET_OUT_CH,
        num_blocks=RESNET_BLOCKS, hidden_dim=RESNET_HIDDEN,
        data_dim=RESNET_DATA_DIM, in_proj_cfg=in_proj, out_proj_cfg=out_proj,
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=RESNET_HIDDEN),
        block_cfg=block_cfg,
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        target_size=None,
    )


def build_resnet_hyena(patch_size: int) -> torch.nn.Module:
    import torch
    from nvsubquadratic.lazy_config import LazyConfig, instantiate
    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.hyena_nd import Hyena
    from nvsubquadratic.modules.kernels_nd import SIRENKernelND
    from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
    from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init

    grid = RESNET_CANVAS // patch_size
    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=RESNET_HIDDEN,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2, hidden_dim=RESNET_HIDDEN,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2, out_dim=RESNET_HIDDEN,
                    mlp_hidden_dim=32, num_layers=3, embedding_dim=32,
                    omega_0=10.0, L_cache=grid, use_bias=True, hidden_omega_0=1.0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double", fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * RESNET_HIDDEN, out_channels=3 * RESNET_HIDDEN,
                kernel_size=3, groups=3 * RESNET_HIDDEN, padding=1, bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
            pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=RESNET_HIDDEN),
            qk_norm_cfg=None, use_rope=False,
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=RESNET_BLOCKS),
    )
    return instantiate(_resnet_net_cfg(mixer_cfg, patch_size))


def build_resnet_attention(patch_size: int) -> torch.nn.Module:
    import torch
    from nvsubquadratic.lazy_config import LazyConfig, instantiate
    from nvsubquadratic.modules.attention import Attention
    from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
    from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init

    grid = RESNET_CANVAS // patch_size
    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=RESNET_HIDDEN,
        mixer_cfg=LazyConfig(Attention)(
            hidden_dim=RESNET_HIDDEN, num_heads=RESNET_HEADS,
            apply_qk_norm=True, use_rope=True, is_causal=False,
            rope_base=10000.0, rope_spatial_dims=(grid, grid),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=RESNET_BLOCKS),
    )
    return instantiate(_resnet_net_cfg(mixer_cfg, patch_size))


def build_resnet_mamba(patch_size: int) -> torch.nn.Module:
    import torch
    from mamba_ssm import Mamba2
    from nvsubquadratic.lazy_config import LazyConfig, instantiate
    from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer

    mixer_cfg = LazyConfig(MambaNDMixer)(
        mamba_layer_cfg=LazyConfig(Mamba2)(
            d_model=RESNET_HIDDEN, headdim=RESNET_HIDDEN // RESNET_HEADS, expand=2,
        ),
        bidirectional=True,
    )
    return instantiate(_resnet_net_cfg(mixer_cfg, patch_size))


# ═══════════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════════

def _make_batch(setup: str, patch_size: int, batch_size: int, device: torch.device):
    if setup == "vit5":
        x = torch.randn(batch_size, VIT5_IMAGE_SIZE, VIT5_IMAGE_SIZE, VIT5_IN_CHANNELS, device=device)
    else:
        x = torch.randn(batch_size, RESNET_CANVAS, RESNET_CANVAS, RESNET_IN_CH, device=device)
    return {"input": x, "condition": None}


def measure(model, batch, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    for _ in range(WARMUP_STEPS):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)["logits"].mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    times = []
    for _ in range(MEASURE_STEPS):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)["logits"].mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


# ═══════════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════════

STYLE = {
    "Hyena":         {"color": "#2563EB", "marker": "o"},
    "Attention":     {"color": "#DC2626", "marker": "s"},
    "Mamba (bidir)": {"color": "#16A34A", "marker": "^"},
}

SETUP_LABELS = {
    "vit5":   r"ViT-5-Small  (384-dim, 12 blocks, 256×256)",
    "resnet": r"ResNet  (256-dim, 4 blocks, 256×256)",
}


def make_plot(all_results: dict, out_path: Path):
    import matplotlib.pyplot as plt

    setups = list(all_results.keys())
    fig, axes = plt.subplots(1, len(setups), figsize=(5.5 * len(setups), 4.2), sharey=False)
    if len(setups) == 1:
        axes = [axes]

    for ax, setup in zip(axes, setups):
        results = all_results[setup]
        for name, style in STYLE.items():
            if name not in results:
                continue
            xs, ys, errs = [], [], []
            for p in PATCH_SIZES:
                vals = np.array(results[name].get(p, [float("nan")]))
                if np.isnan(vals).all():
                    continue
                xs.append(p)
                ys.append(float(np.mean(vals)))
                errs.append(float(np.std(vals)))
            if not xs:
                continue
            ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                    linewidth=2.0, markersize=7, markeredgecolor="white",
                    markeredgewidth=1.2, label=name, zorder=3)
            ax.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            color=style["color"], alpha=0.10, zorder=2)

        ax.set_xlabel("Patch Size", fontsize=12, labelpad=6)
        ax.set_ylabel("Step Time  (ms)", fontsize=12, labelpad=6)
        ax.set_title(SETUP_LABELS[setup], fontsize=10, fontweight="bold", pad=10)
        ax.set_xticks(PATCH_SIZES)
        ax.tick_params(axis="both", labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.grid(axis="y", color="#EEEEEE", linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)

        # Secondary x-axis: token count (grid²)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(PATCH_SIZES)
        canvas = VIT5_IMAGE_SIZE if setup == "vit5" else RESNET_CANVAS
        labels = [f"{(canvas // p) ** 2:,}" for p in PATCH_SIZES]
        ax2.set_xticklabels(labels, fontsize=8, color="#555555")
        ax2.set_xlabel("Tokens  (N)", fontsize=8, color="#555555", labelpad=4)
        ax2.tick_params(axis="x", length=0)
        ax2.spines["top"].set_color("#CCCCCC")

        ax.legend(fontsize=10, frameon=True, loc="upper right",
                  framealpha=0.92, edgecolor="#DDDDDD")

    fig.suptitle("Step Time vs. Patch Size  ·  bf16  ·  Attention: compiled  ·  Hyena: torch_fft (eager)  ·  Mamba: eager",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Subprocess worker
# ═══════════════════════════════════════════════════════════════════════════════

_BUILDERS = {
    "vit5": {
        "Hyena":         build_vit5_hyena,
        "Attention":     build_vit5_attention,
        "Mamba (bidir)": build_vit5_mamba,
    },
    "resnet": {
        "Hyena":         build_resnet_hyena,
        "Attention":     build_resnet_attention,
        "Mamba (bidir)": build_resnet_mamba,
    },
}

_PATCH_BATCH = {
    "vit5":   VIT5_PATCH_BATCH,
    "resnet": RESNET_PATCH_BATCH,
}


def _worker(setup: str, model_name: str, patch_size: int, result_path: str):
    sys.path.insert(0, str(_ROOT))
    device = torch.device("cuda")
    try:
        model = _BUILDERS[setup][model_name](patch_size).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        # Compile only Attention — subq_ops FFT kernel and Mamba Triton ops are
        # not compatible with torch.compile (graph breaks / shape assertions).
        if model_name == "Attention":
            model = torch.compile(model, mode=COMPILE_MODE)

        batch_size = _PATCH_BATCH[setup][patch_size]
        batch = _make_batch(setup, patch_size, batch_size, device)
        times = measure(model, batch, device)
        result = {"times": times, "n_params": n_params, "status": "ok",
                  "compiled": model_name != "Mamba (bidir)"}
    except torch.cuda.OutOfMemoryError:
        result = {"times": [float("nan")], "n_params": 0, "status": "oom", "compiled": False}
    except Exception as exc:
        result = {"times": [float("nan")], "n_params": 0,
                  "status": f"error: {exc}", "compiled": False}
    with open(result_path, "w") as f:
        json.dump(result, f)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "paper/figures/step_time_vs_patch_size.pdf")
    parser.add_argument("--setups", nargs="+", default=["vit5", "resnet"],
                        choices=["vit5", "resnet"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available — run on a GPU node.")

    ctx = mp.get_context("spawn")
    all_results: dict[str, dict[str, dict[int, list[float]]]] = {}

    for setup in args.setups:
        print(f"\n{'═' * 62}")
        print(f"  Setup: {SETUP_LABELS[setup]}")
        print(f"{'═' * 62}")
        all_results[setup] = {}

        for model_name in ("Hyena", "Attention", "Mamba (bidir)"):
            mode_label = {"Hyena": "eager + subq_ops", "Attention": "compiled",
                          "Mamba (bidir)": "eager"}[model_name]
            print(f"\n  ── {model_name} ({mode_label})")
            all_results[setup][model_name] = {}

            for p in PATCH_SIZES:
                canvas = VIT5_IMAGE_SIZE if setup == "vit5" else RESNET_CANVAS
                grid = canvas // p
                bs = _PATCH_BATCH[setup][p]
                print(f"    patch={p:2d}  {grid:3d}×{grid:<3d}  N={grid**2:6,}  "
                      f"batch={bs:3d} ...", end=" ", flush=True)

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                    result_path = tf.name

                proc = ctx.Process(target=_worker,
                                   args=(setup, model_name, p, result_path))
                proc.start()
                proc.join()

                with open(result_path) as f:
                    result = json.load(f)
                os.unlink(result_path)

                times, status = result["times"], result["status"]
                if status == "ok":
                    print(f"{np.mean(times):6.1f} ms ±{np.std(times):.1f}  "
                          f"({result['n_params']:.1f}M params)")
                else:
                    print(status.upper())
                all_results[setup][model_name][p] = times

    make_plot(all_results, args.out)


if __name__ == "__main__":
    main()
