#!/bin/bash
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=1
#SBATCH --job-name=arc_ttt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/arc_ttt_%A.out

echo "This job is running on node: $SLURM_NODELIST"

set -eo pipefail

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch examples/arc/submit_arc_ttt_1gpu_geodude.sh <config.py> <checkpoint.ckpt> [extra args...]"
    echo "  e.g. sbatch examples/arc/submit_arc_ttt_1gpu_geodude.sh examples/arc/cfg_vit_rearc.py /path/to/epoch=469.ckpt"
    exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
shift 2

# ─── Environment ──────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

# ─── Run ──────────────────────────────────────────────────────────────────────
cd /home/dwessel/code/nvSubquadratic-private
mkdir -p logs

PYTHONPATH=. python scripts/evaluation/eval_arc_ttt.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    "$@"
