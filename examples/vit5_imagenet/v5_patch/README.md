# v5_patch — Patch-Size Ablation: Hyena vs Attention

Ablates ViT-5-Small on ImageNet-1k across patch sizes **16, 8, 4, 2, 1**
for both **Hyena (CLS-row + FiLM + GRN)** and **standard multi-head attention**.

All runs use 8 H100 GPUs on a single node with an effective batch size of 2048.

## Sequence lengths

All configs use **4 register tokens** + 1 CLS token, regardless of patch size.
For Hyena (CLS-row), the first grid row is zero-padded to width `num_patches_w`
when 1 + 4 registers \< `num_patches_w`.

| Patch | Patches | Registers | Total tokens |
| ----: | ------: | --------: | -----------: |
|    16 |   14x14 |         4 |          201 |
|     8 |   28x28 |         4 |          789 |
|     4 |   56x56 |         4 |        3,141 |
|     2 | 112x112 |         4 |       12,549 |
|     1 | 224x224 |         4 |       50,181 |

For Hyena with CLS-row layout, the actual grid includes zero-padding:
T = `num_patches_w` (first row: CLS + regs + pad) + `num_patches` = `(H+1) × W`.
Attention does not use prepend_registers, so T = 1 + 4 + `num_patches` (no padding).

> **Note:** Attention is O(n^2). Patch sizes 2 and 1 will be very expensive or
> infeasible for attention. Hyena should handle all patch sizes.

## Setup

### 1. Clone and install nvSubquadratic

```bash
git clone <repo-url> && cd nvSubquadratic-private

# PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install -r requirements-dev.txt
pip install --no-build-isolation -e .
```

### 2. Install NVIDIA DALI (GPU-accelerated data loading)

DALI provides fused decode + crop + augmentations on the GPU, significantly
speeding up data loading for ImageNet.

```bash
pip install nvidia-dali-cuda120
```

### 3. Install Apex (FusedLAMB optimizer)

Apex provides the `FusedLAMB` optimizer which batches all parameter updates
into 1-2 kernel launches. It must be built from source with CUDA extensions.

```bash
# Apex is included as a subdirectory in this repo
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..
```

### 4. Install QuACK kernels (fused RMSNorm — optional but recommended)

QuACK provides a fused Triton RMSNorm kernel that replaces the manual
float32-upcast-then-downcast path. **This is optional** — the code falls
back to pure PyTorch if QuACK is not installed.

```bash
pip install quack-kernels
```

### 5. Verify environment

```bash
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
from apex.optimizers import FusedLAMB; print('Apex FusedLAMB: OK')
try:
    from nvidia.dali import pipeline; print('DALI: OK')
except ImportError:
    print('DALI: NOT INSTALLED')
try:
    from quack import rmsnorm; print(f'QuACK RMSNorm: {\"OK\" if rmsnorm is not None else \"fallback\"}')
except ImportError:
    print('QuACK: NOT INSTALLED (will use PyTorch fallback)')
"
```

## Running experiments

### Environment variables

Set these before running (also set in the submit script):

```bash
export IMAGENET_PATH=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageNet/
export IMAGENET_FOLDER_PATH=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder
export LOCAL_STAGING_DIR=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder
```

### Submit via SLURM (recommended)

Each config is self-contained — just point the submit script at it:

```bash
# Hyena runs
sbatch --job-name=hyena-p16 examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/hyena_patch16.py
sbatch --job-name=hyena-p8  examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/hyena_patch8.py
sbatch --job-name=hyena-p4  examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/hyena_patch4.py
sbatch --job-name=hyena-p2  examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/hyena_patch2.py
sbatch --job-name=hyena-p1  examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/hyena_patch1.py

# Attention runs
sbatch --job-name=attn-p16  examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/attention_patch16.py
sbatch --job-name=attn-p8   examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/attention_patch8.py
sbatch --job-name=attn-p4   examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/attention_patch4.py
sbatch --job-name=attn-p2   examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/attention_patch2.py
sbatch --job-name=attn-p1   examples/vit5_imagenet/v5_patch/submit_8gpu.sh examples/vit5_imagenet/v5_patch/attention_patch1.py
```

### Run locally (single node, 8 GPUs)

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 experiments/run.py \
    --config examples/vit5_imagenet/v5_patch/hyena_patch16.py \
    num_nodes=1
```

### Resume a run (autoresume)

To resume from a W&B checkpoint, pass the experiment directory and run ID:

```bash
sbatch --job-name=hyena-p8-resume examples/vit5_imagenet/v5_patch/submit_8gpu.sh \
    examples/vit5_imagenet/v5_patch/hyena_patch8.py \
    experiment_dir=runs/<run_directory_name> \
    autoresume.enabled=true \
    wandb.run_id=<wandb_run_id>
```

### Override batch size if OOM

If a config OOMs, reduce `dataset.batch_size` and increase `train.accumulate_grad_steps`
to keep effective batch = 2048:

```bash
sbatch --job-name=attn-p4 examples/vit5_imagenet/v5_patch/submit_8gpu.sh \
    examples/vit5_imagenet/v5_patch/attention_patch4.py \
    dataset.batch_size=8 train.accumulate_grad_steps=32
```

## Batch configurations

Each config has batch_per_gpu and accumulate_grad_steps pre-set to target
effective_batch = 8 GPUs x batch_per_gpu x accum_steps = 2048:

| Patch | batch/gpu | accum | effective | compile_mode               |
| ----: | --------: | ----: | --------: | :------------------------- |
|    16 |       256 |     1 |     2,048 | max-autotune               |
|     8 |        64 |     4 |     2,048 | max-autotune-no-cudagraphs |
|     4 |        16 |    16 |     2,048 | max-autotune-no-cudagraphs |
|     2 |         4 |    64 |     2,048 | max-autotune-no-cudagraphs |
|     1 |         1 |   256 |     2,048 | max-autotune-no-cudagraphs |

## Training recipe

Identical across all configs (same as v3 pretrain base):

- **Optimizer:** Apex FusedLAMB, lr=4e-3, wd=0.05
- **Schedule:** Cosine, 800 epochs, 5-epoch warmup
- **Precision:** bf16-mixed
- **EMA:** decay=0.99996
- **Augmentation:** ThreeAugment + Mixup(0.8) + CutMix(1.0)
- **Gradient clip:** 1.0

## Architecture notes

- **Hyena** uses CLS-row layout: \[CLS, registers, zero-pad, patches\] reshaped to a 2D grid.
  The first row is `[CLS, 4 regs, (W-5) zeros]` and the remaining rows are patches.
  FiLM conditioning modulates the SIREN kernel via register pooling.
  GRN (Global Response Normalization) adds inter-channel competition.

- **Attention** uses standard multi-head attention (6 heads) with RoPE
  and 4 register tokens (appended, no 2D reshape). QK normalization via RMSNorm.

## W&B

All runs log to project `nvsubquadratic`, entity `implicit-long-convs`,
group `v5_patch_ablation`.
