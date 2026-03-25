#!/bin/bash
#SBATCH --account=all6000users
#SBATCH --partition=all6000
#SBATCH --gpus=1
#SBATCH --job-name=well-bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/well_bench_%A.out

# Benchmark WELL dataloader + training step throughput.
#
# Usage:
#   sbatch benchmarks/well/submit_bench_ivi.sh <config.py> [bench script args...]
#
# Examples:
#   # Dataloader benchmark
#   sbatch benchmarks/well/submit_bench_ivi.sh examples/well/supernova_explosion_64/cfg_vit5_attention.py --bench dataloader
#
#   # Training step benchmark (with compile)
#   sbatch benchmarks/well/submit_bench_ivi.sh examples/well/supernova_explosion_64/cfg_vit5_attention.py --bench training --compile
#
#   # Full A/B comparison (baseline vs optimized)
#   sbatch benchmarks/well/submit_bench_ivi.sh examples/well/supernova_explosion_64/cfg_vit5_attention.py --bench ab

echo "This job is running on node: $SLURM_NODELIST"
set -eo pipefail

CONFIG="$1"
shift

# Parse --bench argument (default: ab)
BENCH_TYPE="ab"
COMPILE_FLAG=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench) BENCH_TYPE="$2"; shift 2 ;;
        --compile) COMPILE_FLAG="--compile"; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: sbatch benchmarks/well/submit_bench_ivi.sh <config.py> [--bench dataloader|training|ab] [--compile]"
    exit 1
fi

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export WELL_DATA_PATH="${WELL_DATA_PATH:-/ivi/zfs/s0/original_homes/dwessel/data/the_well/datasets}"
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_bench_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/dwessel/code/nvSubquadratic-private

case "$BENCH_TYPE" in
    dataloader)
        echo "=== Dataloader benchmark ==="
        PYTHONPATH=. python benchmarks/well/bench_dataloader.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
        ;;
    training)
        echo "=== Training step benchmark ==="
        PYTHONPATH=. python benchmarks/well/bench_training_step.py --config "$CONFIG" $COMPILE_FLAG "${EXTRA_ARGS[@]}"
        ;;
    ab)
        echo "=== A/B comparison: baseline vs optimized ==="
        PYTHONPATH=. python benchmarks/well/bench_ab_comparison.py --config "$CONFIG" $COMPILE_FLAG "${EXTRA_ARGS[@]}"
        ;;
    *)
        echo "Unknown bench type: $BENCH_TYPE (use: dataloader, training, ab)"
        exit 1
        ;;
esac
