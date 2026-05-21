r"""Hyena config for rayleigh_benard (v2) — **mixed boundary conditions**.

Dataset boundary conditions (from ``_base.BOUNDARY_CONDITIONS``):

- x axis: PERIODIC
- y axis: WALL  (treated as zero-padded "linear" at the conv level)

These map to ``fft_padding=(True, False)``: per-axis ``True`` ⇒ periodic
on that axis, ``False`` ⇒ zero-padded. This is the first config that
exercises the per-axis BC path added in `nvsubquadratic/ops/mixed_fftconv.py`
and wired into ``CKConvND`` in `nvsubquadratic/modules/ckconv_nd.py`.

Notes:
-----
- ``grid_type`` is ``None`` because the per-axis SIREN kernel grid is
  auto-derived from ``fft_padding``: "single" (kernel ≈ input) on the
  periodic x-axis, "double" (kernel ≈ 2·input − 1) on the non-periodic
  y-axis.
- The dataset is anisotropic (512×128), so ``L_cache`` is supplied as a
  **per-axis list** ``[L_x, L_y]`` interpolated from the patch size. The
  SIREN positional grid is built independently per axis and each axis
  spans ``[-1, 1]`` (verified by ``_tmp/check_grid_spans.py``).
- WALL ≠ OPEN physically, but for the convolutional operator both are
  treated as zero-padded linear (the physical distinction is handled by
  data normalisation / the loss, not by the conv kernel).

Submit (1 GPU):

    sbatch --job-name=well_rb_hyena scripts/slurm/submit_1gpu.sh \\
        examples/well/v2/rayleigh_benard/hyena.py
"""

import torch

from examples.well.v2.rayleigh_benard._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL_RESOLUTION,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Model hyperparameters ────────────────────────────────────────────────────
NUM_HIDDEN_CHANNELS = 384
NUM_BLOCKS = 12
PATCH_SIZE = 8  # 512 / 8 = 64 patches on x, 128 / 8 = 16 patches on y.

DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0

# ── Boundary conditions ──────────────────────────────────────────────────────
# rayleigh_benard: periodic on x, wall on y. Wall is treated as zero-padded
# linear at the conv level (see this file's docstring).
FFT_PADDING = (True, False)
# ``grid_type=None`` ⇒ per-axis grid auto-derived from FFT_PADDING:
# "single" on periodic axes, "double" on non-periodic axes.
GRID_TYPE = None

OMEGA_0 = 30.0

# ── Dataset-specific spatial sizes ───────────────────────────────────────────
SPATIAL_X, SPATIAL_Y = SPATIAL_RESOLUTION  # (512, 128)


def get_config() -> ExperimentConfig:
    """Build Hyena experiment config for rayleigh_benard (mixed BC)."""
    config = get_base_config()

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    norm_cfg = LazyConfig(RMSNorm)(dim=NUM_HIDDEN_CHANNELS)

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=IN_CHANNELS,
            out_features=NUM_HIDDEN_CHANNELS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride="${net.in_proj_cfg.patch_size}",
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=NUM_HIDDEN_CHANNELS,
            out_features=OUT_CHANNELS,
            data_dim=DATA_DIM,
            patch_size="${net.in_proj_cfg.patch_size}",
            stride="${net.in_proj_cfg.patch_size}",
        ),
        norm_cfg=norm_cfg,
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=NUM_HIDDEN_CHANNELS,
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim=NUM_HIDDEN_CHANNELS,
                        # Mixed boundary conditions: per-axis tuple.
                        fft_padding=FFT_PADDING,
                        use_fp16_fft=False,
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim=DATA_DIM,
                            out_dim=NUM_HIDDEN_CHANNELS,
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=OMEGA_0,
                            # Anisotropic per-axis L_cache: x is 512, y is 128.
                            # The SIREN positional grid is built per axis at
                            # the corresponding resolution; each axis still
                            # spans [-1, 1].
                            L_cache=[
                                f"${{eval:'{SPATIAL_X} // ${{net.in_proj_cfg.patch_size}}'}}",
                                f"${{eval:'{SPATIAL_Y} // ${{net.in_proj_cfg.patch_size}}'}}",
                            ],
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),
                        # grid_type must be None when fft_padding is a tuple
                        # (per-axis grid is auto-derived).
                        grid_type=GRID_TYPE,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels=3 * NUM_HIDDEN_CHANNELS,
                        out_channels=3 * NUM_HIDDEN_CHANNELS,
                        kernel_size=3,
                        groups=3 * NUM_HIDDEN_CHANNELS,
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                    gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
                    pixelhyena_norm_cfg=LazyConfig(RMSNorm)(
                        dim=NUM_HIDDEN_CHANNELS,
                    ),
                    output_norm_cfg=LazyConfig(RMSNorm)(
                        dim=NUM_HIDDEN_CHANNELS,
                    ),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=norm_cfg,
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=NUM_HIDDEN_CHANNELS,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=norm_cfg,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    return config
