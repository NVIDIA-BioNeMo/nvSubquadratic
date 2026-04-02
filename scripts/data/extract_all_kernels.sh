#!/usr/bin/env bash
# Extract kernel data from all gaussian-mask ablation runs for the
# euler_multi_quadrants_periodicBC experiment.
# Creates JSON files in tmp/kernel_data/
#
# Run inside the container via srun:
#   srun --gres=gpu:1 --cpus-per-task=16 --partition=batch \
#     --container-image=/shared/images/nvsubquadratic_cuda129.sqsh \
#     --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared,/scratch:/scratch" \
#     --container-workdir=/home/dwromero/projects/nvSubquadratic-private \
#     bash scripts/extract_all_kernels.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq

OUTDIR="tmp/kernel_data"
mkdir -p "$OUTDIR"

GMASK_CFG="examples/well/euler_multi_quadrants_periodicBC/cfg_hyena_gaussian_mask.py"
SPATIAL="--spatial-dims 32 32"
NCHAN="--num-channels 16"

extract() {
    local label="$1"
    local ckpt="$2"
    shift 2
    local overrides=("$@")

    echo "=== $label ==="
    if [ ! -f "$ckpt" ]; then
        echo "  [SKIP] checkpoint not found: $ckpt"
        return
    fi
    PYTHONPATH=. python scripts/extract_kernel_data.py \
        --config "$GMASK_CFG" \
        --checkpoint "$ckpt" \
        --output "$OUTDIR/${label}.npz" $SPATIAL $NCHAN \
        "${overrides[@]+"${overrides[@]}"}"
    echo ""
}

# ── Init-extent ablation (lr=1e-3, wd=1e-5, w0=30) ──────────────────────
extract "gmask-e025" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.25_num_workers_2_2026-03-27-23-54-37/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.25

extract "gmask-e050" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_2_2026-03-27-23-54-39/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

extract "gmask-e075" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.75_num_workers_2_2026-03-27-23-54-37/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.75

extract "gmask-e100" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_1_num_workers_2_2026-03-27-23-54-39/checkpoints/last.ckpt"

# ── omega_0 ablation (init_extent=0.5, lr=1e-3, wd=1e-5) ────────────────
extract "gmask-w0-1" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_2_w0_1_2026-03-27-23-54-37/checkpoints/last.ckpt" \
    --overrides \
        net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5 \
        net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=1.0

extract "gmask-w0-10" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_2_w0_10_2026-03-27-23-54-37/checkpoints/last.ckpt" \
    --overrides \
        net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5 \
        net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=10.0

extract "gmask-w0-100" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_2_w0_100_2026-03-27-23-54-36/checkpoints/last.ckpt" \
    --overrides \
        net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5 \
        net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=100.0

# ── LR ablation (init_extent=0.5, wd=1e-5, w0=30) ──────────────────────
extract "gmask-lr3e4" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_lr_0.0003_num_workers_12_2026-03-28-19-28-23/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

extract "gmask-lr3e3" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_lr_0.003_num_workers_12_2026-03-28-19-28-23/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

extract "gmask-lr4e3" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_lr_0.004_num_workers_12_2026-03-28-07-50-25/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

# ── WD ablation (init_extent=0.5, lr=1e-3, w0=30) ──────────────────────
extract "gmask-wd0" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_12_wd_0_2026-03-28-19-28-23/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

extract "gmask-wd1e4" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_12_wd_0.0001_2026-03-28-19-28-23/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

extract "gmask-wd1e6" \
    "runs/DW_examples_well_euler_multi_quadrants_periodicBC_cfg_hyena_gaussian_mask_init_extent_0.5_num_workers_12_wd_1e-06_2026-03-28-22-49-21/checkpoints/last.ckpt" \
    --overrides net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg.init_extent=0.5

echo ""
echo "All done! Output files:"
ls -lh "$OUTDIR"/*.npz
