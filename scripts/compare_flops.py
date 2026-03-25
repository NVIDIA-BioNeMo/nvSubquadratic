#!/usr/bin/env python3
"""Compare FLOPs across ViT-5-Small model variants: Attention, Hyena, Hyena+FiLM.

Instantiates the three ViT-5-Small network variants using the same parameters
as the production configs (examples/vit5_imagenet/v2 and v3), calls
``model.flop_count(inference=...)`` for both training and inference modes,
and prints a side-by-side comparison table.

With ``--scaling``, sweeps resolutions from 7x7 to 112x112 patches, saves
the raw data as CSV, and plots a scaling comparison (PNG).

Usage:
    conda run -n nv-subq python scripts/compare_flops.py
    conda run -n nv-subq python scripts/compare_flops.py --scaling
"""

import csv
import sys
from pathlib import Path

import torch


# Ensure the project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.layer_scale import LayerScale
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Shared model constants (ViT-5-Small, ImageNet-1k) ──────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = IMAGE_SIZE // PATCH_SIZE  # 14

INIT_FN = trunc_normal_init(std=0.02)
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)

# Hyena / SIREN kernel hyperparameters
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# FiLM conditioning
FILM_HIDDEN_DIM = 64


def _make_mlp_cfg() -> LazyConfig:
    """Shared MLP config across all variants."""
    return LazyConfig(MLP)(
        dim=HIDDEN_DIM,
        activation="gelu",
        expansion_factor=float(MLP_RATIO),
        bias=False,
        dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )


def _make_block_cfg(sequence_mixer_cfg: LazyConfig, **kwargs) -> LazyConfig:
    """Shared block config (same as v3/_pretrain_base.py make_block_cfg)."""
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=sequence_mixer_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=_make_mlp_cfg(),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
        **kwargs,
    )


def build_attention_model() -> ViT5ClassificationNet:
    """Build the ViT-5-Small Attention variant (matches v2 config)."""
    num_heads = 6
    num_registers = 4

    attention_cfg = LazyConfig(ViT5Attention)(
        hidden_dim=HIDDEN_DIM,
        num_heads=num_heads,
        num_patches_h=NUM_PATCHES_H,
        num_patches_w=NUM_PATCHES_W,
        num_registers=num_registers,
        qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // num_heads, eps=1e-6),
        rope_base=10000.0,
        reg_rope_base=100.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=False,
        out_proj_bias=False,
        init_fn_qkv_proj=INIT_FN,
        init_fn_out_proj=INIT_FN,
    )

    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=attention_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=_make_mlp_cfg(),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
    )

    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=num_registers,
        dropout_rate=0.0,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=block_cfg,
    )
    return instantiate(net_cfg)


def _make_hyena_mixer_cfg(film_cfg=None) -> LazyConfig:
    """Build the QKVSequenceMixer wrapping Hyena (shared between Hyena/FiLM)."""
    kernel_kwargs = {
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
    if film_cfg is not None:
        kernel_kwargs["film_cfg"] = film_cfg

    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(**kernel_kwargs),
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
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )


def build_hyena_model() -> ViT5ClassificationNet:
    """Build the ViT-5-Small Hyena variant (matches v3 gated config, no FiLM)."""
    num_registers = NUM_PATCHES_W - 1  # 13

    hyena_mixer_cfg = _make_hyena_mixer_cfg(film_cfg=None)

    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=num_registers,
        dropout_rate=0.0,
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=_make_block_cfg(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w=NUM_PATCHES_W,
            ),
        ),
    )
    return instantiate(net_cfg)


def build_hyena_film_model() -> ViT5ClassificationNet:
    """Build the ViT-5-Small Hyena+FiLM variant (matches v3 gated+FiLM config)."""
    num_registers = NUM_PATCHES_W - 1  # 13

    film_cfg = LazyConfig(KernelFiLMGenerator)(
        cond_dim=HIDDEN_DIM,
        kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_film_layers=KERNEL_NUM_LAYERS - 1,
        film_hidden_dim=FILM_HIDDEN_DIM,
    )

    hyena_mixer_cfg = _make_hyena_mixer_cfg(film_cfg=film_cfg)
    register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=num_registers)

    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=num_registers,
        dropout_rate=0.0,
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=_make_block_cfg(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w=NUM_PATCHES_W,
            ),
            register_pooling_cfg=register_pooling_cfg,
            num_registers=num_registers,
        ),
    )
    return instantiate(net_cfg)


def _build_attention_at_resolution(grid_side: int) -> ViT5ClassificationNet:
    """Build Attention variant for an arbitrary square patch grid."""
    num_heads = 6
    num_registers = 4
    image_size = grid_side * PATCH_SIZE

    attention_cfg = LazyConfig(ViT5Attention)(
        hidden_dim=HIDDEN_DIM,
        num_heads=num_heads,
        num_patches_h=grid_side,
        num_patches_w=grid_side,
        num_registers=num_registers,
        qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // num_heads, eps=1e-6),
        rope_base=10000.0,
        reg_rope_base=100.0,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=False,
        out_proj_bias=False,
        init_fn_qkv_proj=INIT_FN,
        init_fn_out_proj=INIT_FN,
    )

    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=attention_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=_make_mlp_cfg(),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
    )

    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=image_size,
        num_registers=num_registers,
        dropout_rate=0.0,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=block_cfg,
    )
    return instantiate(net_cfg)


def _make_hyena_mixer_cfg_at_resolution(grid_side: int, film_cfg=None) -> LazyConfig:
    """Build the QKVSequenceMixer wrapping Hyena for a given grid side."""
    kernel_kwargs = {
        "data_dim": 2,
        "out_dim": HIDDEN_DIM,
        "mlp_hidden_dim": KERNEL_MLP_HIDDEN_DIM,
        "num_layers": KERNEL_NUM_LAYERS,
        "embedding_dim": KERNEL_EMBEDDING_DIM,
        "omega_0": KERNEL_OMEGA_0,
        "L_cache": grid_side + 1,
        "use_bias": True,
        "hidden_omega_0": KERNEL_HIDDEN_OMEGA_0,
    }
    if film_cfg is not None:
        kernel_kwargs["film_cfg"] = film_cfg

    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(**kernel_kwargs),
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
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )


def _build_hyena_at_resolution(grid_side: int, film: bool = False) -> ViT5ClassificationNet:
    """Build Hyena (or Hyena+FiLM) variant for an arbitrary square patch grid."""
    num_registers = grid_side - 1
    image_size = grid_side * PATCH_SIZE

    film_cfg = None
    if film:
        film_cfg = LazyConfig(KernelFiLMGenerator)(
            cond_dim=HIDDEN_DIM,
            kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
            num_film_layers=KERNEL_NUM_LAYERS - 1,
            film_hidden_dim=FILM_HIDDEN_DIM,
        )

    hyena_mixer_cfg = _make_hyena_mixer_cfg_at_resolution(grid_side, film_cfg=film_cfg)

    block_kwargs = {}
    if film:
        block_kwargs["register_pooling_cfg"] = LazyConfig(RegisterPooling)(num_registers=num_registers)
        block_kwargs["num_registers"] = num_registers

    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=image_size,
        num_registers=num_registers,
        dropout_rate=0.0,
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=_make_block_cfg(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w=grid_side,
            ),
            **block_kwargs,
        ),
    )
    return instantiate(net_cfg)


# ─── Resolution grid for scaling analysis ────────────────────────────────────
GRID_SIDES = [7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112]


def scaling_analysis():
    """Sweep resolutions, save CSV + PNG scaling plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _PROJECT_ROOT / "scripts"
    csv_path = out_dir / "flop_scaling.csv"
    png_path = out_dir / "flop_scaling.png"

    variants = ["Attention", "Hyena", "Hyena+FiLM"]

    rows = []
    print(
        f"Sweeping {len(GRID_SIDES)} resolutions: {GRID_SIDES[0]}x{GRID_SIDES[0]} to {GRID_SIDES[-1]}x{GRID_SIDES[-1]} ..."
    )

    for gs in GRID_SIDES:
        img_size = gs * PATCH_SIZE
        num_patches = gs * gs
        print(f"  grid {gs}x{gs}  (image {img_size}x{img_size}, {num_patches} patches) ... ", end="", flush=True)

        attn = _build_attention_at_resolution(gs)
        hyena = _build_hyena_at_resolution(gs, film=False)
        film = _build_hyena_at_resolution(gs, film=True)

        row = {
            "grid_side": gs,
            "image_size": img_size,
            "num_patches": num_patches,
            "Attention_train": attn.flop_count(inference=False),
            "Attention_infer": attn.flop_count(inference=True),
            "Hyena_train": hyena.flop_count(inference=False),
            "Hyena_infer": hyena.flop_count(inference=True),
            "Hyena+FiLM_train": film.flop_count(inference=False),
            "Hyena+FiLM_infer": film.flop_count(inference=True),
        }
        rows.append(row)
        print(
            f"Attn={row['Attention_train'] / 1e9:.1f}G  Hyena={row['Hyena_train'] / 1e9:.1f}G  FiLM={row['Hyena+FiLM_train'] / 1e9:.1f}G"
        )

        del attn, hyena, film

    # ── Save CSV ─────────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    num_patches_list = [r["num_patches"] for r in rows]

    style_map = {
        "Attention": {"color": "#2563eb", "marker": "o"},
        "Hyena": {"color": "#16a34a", "marker": "s"},
        "Hyena+FiLM": {"color": "#dc2626", "marker": "^"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (mode_label, mode_key) in zip(axes, [("Training", "train"), ("Inference", "infer")]):
        for variant in variants:
            col = f"{variant}_{mode_key}"
            gflops = [r[col] / 1e9 for r in rows]
            st = style_map[variant]
            ax.plot(
                num_patches_list,
                gflops,
                label=variant,
                color=st["color"],
                marker=st["marker"],
                markersize=5,
                linewidth=2,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")

        # Mark current resolution (14x14 = 196 patches)
        ax.axvline(x=196, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        y_lo, y_hi = ax.get_ylim()
        ax.text(196 * 1.12, y_lo * 1.5, "14x14\n(224px)", color="gray", fontsize=8, va="bottom", ha="left")

        ax.set_xlabel("Number of patches (grid_side²)", fontsize=11)
        ax.set_title(mode_label, fontsize=13, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, loc="upper left")

    axes[0].set_ylabel("GFLOPs (log scale)", fontsize=11)

    fig.suptitle(
        f"ViT-5-Small FLOP Scaling  (D={HIDDEN_DIM}, {NUM_BLOCKS} blocks, patch={PATCH_SIZE})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.90, wspace=0.08)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {png_path}")


def _fmt(n: int) -> str:
    """Format a FLOP count as a human-readable string (e.g. '4.61 GFLOPs')."""
    if n >= 1e12:
        return f"{n / 1e12:.2f} TFLOPs"
    if n >= 1e9:
        return f"{n / 1e9:.2f} GFLOPs"
    if n >= 1e6:
        return f"{n / 1e6:.2f} MFLOPs"
    return f"{n:,} FLOPs"


def _per_block_breakdown(model: ViT5ClassificationNet, inference: bool) -> dict[str, int]:
    """Return a dict with per-component FLOPs for a single block and the full model."""
    D = model.hidden_dim
    P = model.patch_size
    num_patches = model.num_patches
    in_channels = model.patch_embed.in_channels

    result = {}

    # Patch embedding
    result["patch_embed"] = 2 * in_channels * D * P * P * num_patches

    # Pos embed add
    result["pos_embed_add"] = num_patches * D

    # Token count (same logic as flop_count)
    T = num_patches
    if model.readout == "cls":
        T += 1
    if model.num_registers > 0:
        if model.prepend_registers and model.readout != "cls":
            T += model._register_row_width
        else:
            T += model.num_registers
    result["num_tokens"] = T

    # Per-block breakdown (use first block as representative)
    block = model.blocks[0]
    result["block_total"] = block.flop_count(T, inference=inference)

    # Sub-components of the block
    result["block_input_norm"] = block.input_norm.flop_count(T)
    if block.register_pooling is not None:
        result["block_reg_pooling"] = block.register_pooling.flop_count(D)
    else:
        result["block_reg_pooling"] = 0
    result["block_mixer"] = block.sequence_mixer.flop_count(T, inference=inference)
    result["block_mlp_norm"] = block.mlp_norm.flop_count(T)
    result["block_mlp"] = block.mlp.flop_count(T)
    if isinstance(block.ls_attn, LayerScale):
        result["block_ls"] = 2 * block.ls_attn.flop_count(T)
    else:
        result["block_ls"] = 0

    # Output norm + head
    result["out_norm"] = model.out_norm.flop_count(1)
    if model.readout == "gap":
        result["gap"] = num_patches * D
    else:
        result["gap"] = 0
    if model.readout == "register_concat":
        head_dim = model.num_registers * model.neck_dim
    else:
        head_dim = D
    result["head"] = 2 * head_dim * model.num_classes

    # Full model total
    result["total"] = model.flop_count(inference=inference)

    return result


def main():
    print("=" * 80)
    print("ViT-5-Small FLOP Comparison: Attention vs Hyena vs Hyena+FiLM")
    print("=" * 80)
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, Patch size: {PATCH_SIZE}")
    print(f"  Hidden dim: {HIDDEN_DIM}, Blocks: {NUM_BLOCKS}, MLP ratio: {MLP_RATIO}")
    print(f"  Patches: {NUM_PATCHES_H}x{NUM_PATCHES_W} = {NUM_PATCHES_H * NUM_PATCHES_W}")
    print()

    models = {
        "Attention": build_attention_model(),
        "Hyena": build_hyena_model(),
        "Hyena+FiLM": build_hyena_film_model(),
    }

    for mode_label, inference in [("TRAINING", False), ("INFERENCE", True)]:
        print("-" * 80)
        print(f"  Mode: {mode_label}")
        print("-" * 80)

        breakdowns = {}
        for name, model in models.items():
            breakdowns[name] = _per_block_breakdown(model, inference=inference)

        # Header
        col_w = 18
        label_w = 24
        header = f"{'Component':<{label_w}}"
        for name in models:
            header += f"{name:>{col_w}}"
        print(header)
        print("─" * (label_w + col_w * len(models)))

        # Rows
        rows = [
            ("Patch embed", "patch_embed"),
            ("Pos embed add", "pos_embed_add"),
            ("Num tokens (T)", "num_tokens"),
            ("", None),  # separator
            ("Block total (x1)", "block_total"),
            ("  Input norm", "block_input_norm"),
            ("  Reg pooling", "block_reg_pooling"),
            ("  Mixer", "block_mixer"),
            ("  MLP norm", "block_mlp_norm"),
            ("  MLP", "block_mlp"),
            ("  LayerScale (x2)", "block_ls"),
            ("", None),
            ("Output norm", "out_norm"),
            ("GAP", "gap"),
            ("Head", "head"),
            ("", None),
            ("TOTAL", "total"),
        ]

        for label, key in rows:
            if key is None:
                print()
                continue
            row = f"{label:<{label_w}}"
            for name in models:
                val = breakdowns[name][key]
                if key == "num_tokens":
                    row += f"{val:>{col_w},}"
                else:
                    row += f"{_fmt(val):>{col_w}}"
            print(row)

        print()

    # Inference savings for Hyena (frozen kernels)
    print("-" * 80)
    print("  Inference Savings (Hyena: frozen kernel vs training)")
    print("-" * 80)
    for name in ["Hyena", "Hyena+FiLM"]:
        model = models[name]
        train_flops = model.flop_count(inference=False)
        infer_flops = model.flop_count(inference=True)
        saved = train_flops - infer_flops
        pct = 100.0 * saved / train_flops if train_flops > 0 else 0
        print(
            f"  {name:15s}: train={_fmt(train_flops)}, inference={_fmt(infer_flops)}, saved={_fmt(saved)} ({pct:.1f}%)"
        )
    print()

    # Relative to Attention
    print("-" * 80)
    print("  Relative to Attention (training)")
    print("-" * 80)
    attn_flops = models["Attention"].flop_count(inference=False)
    for name, model in models.items():
        total = model.flop_count(inference=False)
        ratio = total / attn_flops
        print(f"  {name:15s}: {_fmt(total):>15s}  ({ratio:.3f}x Attention)")
    print()


if __name__ == "__main__":
    if "--scaling" in sys.argv:
        scaling_analysis()
    else:
        main()
        print("Tip: run with --scaling to generate a resolution scaling plot.")
