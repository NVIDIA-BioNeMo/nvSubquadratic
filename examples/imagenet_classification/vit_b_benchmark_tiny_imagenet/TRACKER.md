# TinyImageNet Classification - ViT-B Benchmark

Benchmark configs comparing Hyena vs Attention sequence mixers at ViT-B scale, with and without patchification.

## Task Description

- **Dataset**: TinyImageNet (200 classes, 64×64 RGB images)
- **Task**: Multi-class classification
- **Objective**: Compare Hyena vs Attention at ViT-B scale, ablating over patchification

## Model Architecture (ViT-B Scale)

All models match ViT-B architecture size:

- **Hidden dimension**: 768
- **Number of blocks**: 12
- **MLP expansion**: 2.0 (GLU activation)
- **Precision**: bf16-mixed

| Config                  | Mixer     | Patchify | Hidden | Blocks | Heads | Patch Size | Seq Length |
| ----------------------- | --------- | -------- | ------ | ------ | ----- | ---------- | ---------- |
| `hyena.py`              | Hyena     | No       | 768    | 12     | -     | -          | 4,096      |
| `hyena_patchify.py`     | Hyena     | Yes      | 768    | 12     | -     | 4          | 256        |
| `attention.py`          | Attention | No       | 768    | 12     | 12    | -          | 4,096      |
| `attention_patchify.py` | Attention | Yes      | 768    | 12     | 12    | 4          | 256        |

### Sequence Lengths

With 64×64 images:

- **No patchify**: 64×64 = 4,096 tokens
- **Patchify (p=4)**: 16×16 = 256 tokens

### Data Augmentation

All configs use:

- **Mixup** (α=0.8) + **Cutmix** (α=1.0)
- **RandAugment**: rand-m9-n3-mstd0.5
- Random crop (64×64 with padding=4) + Horizontal flip

### Design Notes

- **Weight decay**: Attention uses 0.05; Hyena uses 0.0 (known to work better without weight decay).
- **Modulation mask**: Hyena uses `GaussianModulationND` on the SIREN kernel to prevent ringing artifacts.
- **L_cache**: Set to match spatial resolution seen by the kernel — 64 for non-patchified, 16 for patchified (p=4).
- **Kernel**: Uses `SIRENKernelND` (implicit neural representation) for continuous convolution kernels.
- **RoPE**: Enabled for Attention, disabled for Hyena (Hyena uses its own positional encoding via the SIREN kernel).

______________________________________________________________________

## Running Experiments

```bash
# Activate environment
conda activate nvsubq
source .env  # For WandB API key and HF_TOKEN
export PYTHONPATH=.

# Run individual configs
srun --gres=gpu:1 -c 16 --partition capacity \
    python experiments/run.py --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/hyena.py

# Or submit all via SLURM
sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_hyena.sh
sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_hyena_patchify.sh
sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_attention.sh
sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_attention_patchify.sh
```

______________________________________________________________________

## 🏆 Results

### Leaderboard (Best Results Per Architecture)

| Rank | Architecture | Config | Val Acc | Val Loss | WandB Link |
| ---- | ------------ | ------ | ------- | -------- | ---------- |
| -    | -            | -      | -       | -        | -          |

### Patchification Impact

| Architecture | No Patch | With Patch (p=4) | Notes |
| ------------ | -------- | ---------------- | ----- |
| Hyena        | -        | -                | -     |
| Attention    | -        | -                | -     |

______________________________________________________________________

## Job Submission Log

| Date | Job ID | Config | Status | Val Acc | Notes |
| ---- | ------ | ------ | ------ | ------- | ----- |
| -    | -      | -      | -      | -       | -     |

______________________________________________________________________

## Notes

- All models use TinyImageNet (200 classes, 64×64 images)
- Training uses AdamW optimizer with cosine LR schedule and 5% warmup
- Default training: 600,000 iterations
- Batch size: 32 (consistent across all configs)

______________________________________________________________________

**Last Updated**: 2026-02-13
**Status**: ⏳ Configs reviewed and fixed, ready for training
