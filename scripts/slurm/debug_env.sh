#!/bin/bash
#SBATCH --job-name=debug-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=low
#SBATCH --gpu-bind=closest
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-name=nv-subq
#SBATCH --container-writable
#SBATCH --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared"
#SBATCH --container-workdir=/home/dwromero
#SBATCH --output=/home/dwromero/projects/nvSubquadratic-private/logs/debug_env_%j.out
#SBATCH --error=/home/dwromero/projects/nvSubquadratic-private/logs/debug_env_%j.err

echo "=== PATH ==="
echo $PATH
echo "=== which python ==="
which python 2>&1 || true
echo "=== which python3 ==="
which python3 2>&1 || true
echo "=== which conda ==="
which conda 2>&1 || true
echo "=== conda envs ==="
conda env list 2>&1 || true
echo "=== find conda ==="
find / -maxdepth 4 -name "conda" -type f 2>/dev/null | head -5
echo "=== find python ==="
find / -maxdepth 4 -name "python*" -type f 2>/dev/null | head -10
echo "=== ls /opt ==="
ls -la /opt/ 2>&1 || true
echo "=== cat /etc/bash.bashrc ==="
cat /etc/bash.bashrc 2>/dev/null | tail -5
echo "=== cat ~/.bashrc ==="
cat ~/.bashrc 2>/dev/null | tail -10
