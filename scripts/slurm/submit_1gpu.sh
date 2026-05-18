#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Portable SLURM 1-GPU submission script for nvSubquadratic training.
#
# Usage (run directly — do NOT prefix with sbatch):
#
#   scripts/slurm/submit_1gpu.sh [--job-name=NAME] [sbatch opts...] <config.py> [training overrides...]
#
# Example:
#   scripts/slurm/submit_1gpu.sh --job-name=sn64-hyena \
#       examples/well/v2/supernova_explosion_64/hyena.py
#
# The script auto-detects the project root and cluster layout.
# Cluster-specific overrides are shared with submit.sh via
# scripts/slurm/cluster.env (gitignored).
# ─────────────────────────────────────────────────────────────────────────────

# Static SBATCH defaults — overridden by the sbatch CLI flags we generate below.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest

# In submission mode BASH_SOURCE resolves to the real path; in job mode SLURM
# copies the script to /var/spool so BASH_SOURCE is wrong — use the value
# passed via --export instead.
: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# ── Per-cluster overrides (optional, gitignored) ────────────────────────────
CLUSTER_ENV="$PROJECT_ROOT/scripts/slurm/cluster.env"
[ -f "$CLUSTER_ENV" ] && source "$CLUSTER_ENV"

# Defaults — override any of these in cluster.env or the environment.
# Note: CPUS_PER_TASK_1GPU / MEM_1GPU are 1-GPU-specific so they don't collide
# with submit.sh's CPUS_PER_TASK (intended for the 8-GPU node-fill case).
: "${SLURM_PARTITION:=all}"
: "${CONTAINER_IMAGE:=/shared/images/nvsubquadratic_cuda129.sqsh}"
: "${CPUS_PER_TASK_1GPU:=16}"
: "${MEM_1GPU:=250000M}"
: "${CONDA_ENV:=nv-subq}"
: "${IMAGENET_PATH:=/shared/data/image_datasets/imagenet}"
: "${IMAGENET_FOLDER_PATH:=/shared/data/image_datasets/imagenet_folder}"
: "${WELL_DATA_PATH:=/shared/data/image_datasets/the_well/datasets}"

# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION MODE — no SLURM_JOB_ID means we're on the login node.
# Separate sbatch flags from training args, then re-submit this script.
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    SBATCH_ARGS=()
    TRAIN_ARGS=()
    for arg in "$@"; do
        if [ ${#TRAIN_ARGS[@]} -eq 0 ] && [[ "$arg" == --* ]]; then
            SBATCH_ARGS+=("$arg")
        else
            TRAIN_ARGS+=("$arg")
        fi
    done

    if [ ${#TRAIN_ARGS[@]} -eq 0 ]; then
        echo "Usage: $0 [--job-name=NAME] [sbatch opts...] <config.py> [training overrides...]"
        echo ""
        echo "  e.g. $0 --job-name=sn64-hyena examples/well/v2/supernova_explosion_64/hyena.py"
        echo ""
        echo "Cluster-specific settings can be placed in: scripts/slurm/cluster.env"
        exit 1
    fi

    mkdir -p "$PROJECT_ROOT/logs"

    exec sbatch \
        --partition="$SLURM_PARTITION" \
        --cpus-per-task="$CPUS_PER_TASK_1GPU" \
        --mem="$MEM_1GPU" \
        --container-image="$CONTAINER_IMAGE" \
        --container-mounts="$HOME:$HOME,/shared:/shared,/scratch:/scratch" \
        --container-workdir="$PROJECT_ROOT" \
        --output="$PROJECT_ROOT/logs/%x_%j.out" \
        --error="$PROJECT_ROOT/logs/%x_%j.err" \
        "${SBATCH_ARGS[@]}" \
        --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONDA_ENV=$CONDA_ENV,IMAGENET_PATH=$IMAGENET_PATH,IMAGENET_FOLDER_PATH=$IMAGENET_FOLDER_PATH,WELL_DATA_PATH=$WELL_DATA_PATH" \
        "$0" "${TRAIN_ARGS[@]}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# JOB MODE — running inside the SLURM allocation (worker node / container).
# ─────────────────────────────────────────────────────────────────────────────
set -eo pipefail

if [ -z "$1" ]; then
    echo "Error: no config file provided."
    exit 1
fi

CONFIG="$1"
shift

# Source secrets (.env is gitignored)
set -a
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env"
set +a
export IMAGENET_PATH
export IMAGENET_FOLDER_PATH
export WELL_DATA_PATH

# Activate environment (supports both conda envs and venvs)
ENV_DIR="$HOME/miniconda3/envs/$CONDA_ENV"
if [ -f "$ENV_DIR/bin/activate" ]; then
    source "$ENV_DIR/bin/activate"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "ERROR: Cannot find environment $CONDA_ENV"
    exit 1
fi

export SLURM_JOB_NAME=bash

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export DALI_NO_MMAP=1

cd "$PROJECT_ROOT"

# Triton calls /sbin/ldconfig to find libcuda — bypass entirely via env knob
if [ -z "$TRITON_LIBCUDA_PATH" ]; then
    _libcuda=$(find /usr/lib /usr/local/lib /usr/lib64 /lib /lib64 -name "libcuda.so.1" 2>/dev/null | head -n1) || true
    if [ -n "$_libcuda" ]; then
        export TRITON_LIBCUDA_PATH="$(dirname "$_libcuda")"
    fi
fi

PYTHONPATH=. python3 experiments/run.py --config "$CONFIG" "$@"
