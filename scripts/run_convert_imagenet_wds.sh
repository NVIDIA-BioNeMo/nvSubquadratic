#!/bin/bash
#SBATCH --job-name=convert_imagenet_wds
#SBATCH --account=ceesusers
#SBATCH --partition=cees
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out

# Convert HuggingFace Arrow ImageNet cache to WebDataset TAR shards.
# This is a one-time operation. Output goes to data/imagenet-wds/.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dknigge/code/nvSubquadratic-private
export PYTHONPATH=.
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

# Install webdataset if not already installed
pip install webdataset

# Run conversion
python scripts/convert_imagenet_to_webdataset.py \
    --src data/imagenet \
    --dst data/imagenet-wds \
    --splits train validation \
    --max-shard-size 500000000

echo "Conversion complete!"
