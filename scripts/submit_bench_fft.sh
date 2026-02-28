#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=low
#SBATCH --gpu-bind=closest
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq-bench
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared,/scratch:/scratch"
#SBATCH --container-workdir=/home/dwromero/projects/nvSubquadratic-private
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/%x_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --job-name=fft-bench

set -eo pipefail
source /home/dwromero/miniconda3/etc/profile.d/conda.sh
conda activate nv-subq
cd /home/dwromero/projects/nvSubquadratic-private

python scripts/bench_fftconv_compile.py
