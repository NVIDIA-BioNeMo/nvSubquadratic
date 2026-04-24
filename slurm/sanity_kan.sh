#!/bin/bash
#SBATCH --account=healthcareeng_bionemo
#SBATCH --nodes=1
#SBATCH --partition=polar,polar3,polar4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=healthcareeng_research-nvsubq.sanity_kan

set -x

WORKDIR="${PWD}"
WARPKAN_DIR="$(dirname ${WORKDIR})/warpKAN"
DATA_DIR="/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_bionemo/amoradzadeh/hyena"

BASE_SQSH="${DATA_DIR}/nvsubquadratic-slurm-x86_64-04-17-2026.sqsh"
SAVE_SQSH="${DATA_DIR}/nvsubquadratic-slurm-x86_64-04-22-2026-warpkan.sqsh"

MOUNTS="${DATA_DIR}:/workspace/data"
MOUNTS="${MOUNTS},${WORKDIR}:/workspaces/nvSubquadratic-private"
MOUNTS="${MOUNTS},${WARPKAN_DIR}:/workspaces/warpKAN"
MOUNTS="${MOUNTS},$HOME/.cache:/root/.cache"

WORK_DIR="/workspaces/nvSubquadratic-private"

srun \
    --output "${WORKDIR}/sanity_kan-%j.out" \
    --error  "${WORKDIR}/sanity_kan-%j.err" \
    --export=ALL \
    --container-image="${BASE_SQSH}" \
    --container-mounts="${MOUNTS}" \
    --container-writable \
    --container-save="${SAVE_SQSH}" \
    bash -c "
set -e
cd ${WORK_DIR}
export PYTHONPATH='.'

echo '=== pip install warpkan (from /workspaces/warpKAN into container site-packages) ==='
pip install --no-deps /workspaces/warpKAN
pip show warpkan | head -10

echo '=== import checks ==='
python -c \"
import torch
import warpkan
from warpkan.torch_ext.kanlinear import KANLinear
print('warpKAN imported OK')

from nvsubquadratic.modules.kan_kernels_nd import KANKernelND
print('KANKernelND imported OK')
\"

echo '=== config build check ==='
python -c \"
import sys
sys.path.insert(0, '/workspaces/nvSubquadratic-private')
from examples.vit5_imagenet.vit5_hybrid.full_hyena_kan import get_config
config = get_config()
print(f'get_config() OK')
\"

echo '=== model instantiation + forward pass ==='
python -c \"
import warp as wp
wp.init()
import torch
from omegaconf import OmegaConf
from examples.vit5_imagenet.vit5_hybrid.full_hyena_kan import get_config
from nvsubquadratic.lazy_config import instantiate

# Register the eval resolver used by the training framework
OmegaConf.register_new_resolver('eval', eval, replace=True)

config = get_config()
root = OmegaConf.create({'net': config.net})
OmegaConf.resolve(root)
net = instantiate(root.net).cuda().eval()
n_params = sum(p.numel() for p in net.parameters())
print(f'Model instantiated OK — {n_params/1e6:.1f}M params')

with torch.no_grad():
    x = torch.randn(1, 224, 224, 3, device='cuda')  # [B, H, W, C] channels-last
    out = net({'input': x, 'condition': None})
print(f'Forward pass OK — output: {out}')
\"

echo '=== all checks passed ==='
"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Sanity check passed. New sqsh saved to: ${SAVE_SQSH}"
else
    echo "Sanity check FAILED with exit code ${EXIT_CODE}"
fi
exit ${EXIT_CODE}
