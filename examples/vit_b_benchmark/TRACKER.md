# ImageNet-1k Classification - ViT-B Benchmark

Benchmark configs comparing Hyena vs Attention sequence mixers at ViT-B scale, with and without patchification.

## Task Description

- **Dataset**: ImageNet-1k (1000 classes, 224×224 RGB images)
- **Task**: Multi-class classification
- **Objective**: Compare Hyena vs Attention at ViT-B scale

## Model Architecture (ViT-B Scale)

All models match ViT-B architecture size:

- **Hidden dimension**: 768
- **Number of blocks**: 12
- **MLP expansion**: 2.0 (GLU activation)
- **Precision**: bf16-mixed

| Config                  | Mixer     | Patchify | Hidden | Blocks | Heads | Patch Size | Head Dim |
| ----------------------- | --------- | -------- | ------ | ------ | ----- | ---------- | -------- |
| `hyena.py`              | Hyena     | No       | 768    | 12     | -     | -          | -        |
| `hyena_patchify.py`     | Hyena     | Yes      | 768    | 12     | -     | 16         | -        |
| `attention.py`          | Attention | No       | 768    | 12     | 12    | -          | 64       |
| `attention_patchify.py` | Attention | Yes      | 768    | 12     | 12    | 16         | 64       |

### Sequence Lengths

With 224×224 images:

- **No patchify**: 224×224 = 50,176 tokens
- **Patchify (p=16)**: 14×14 = 196 tokens

### Data Augmentation

All configs use standard ViT augmentations:

- **Mixup** (α=0.8) + **Cutmix** (α=1.0)
- **3-Augment**: Grayscale, Solarization, Gaussian Blur
- **Color jitter** (0.4)
- Center crop

______________________________________________________________________

## Running Experiments

```bash
# Activate environment
conda activate nvsubq
source .env  # For WandB API key and HF_TOKEN
export PYTHONPATH=.

# Run on GPU node
srun --gres=gpu:1 -c 16 --partition low \
    python experiments/run.py --config examples/imagenet_classification/vit_b_benchmark/hyena_patchify.py
```

______________________________________________________________________

## 🏆 Results

### Leaderboard (Best Results Per Architecture)

| Rank | Architecture | Config | Val Acc | Val Loss | WandB Link |
| ---- | ------------ | ------ | ------- | -------- | ---------- |
| -    | -            | -      | -       | -        | -          |

### Patchification Impact

| Architecture | No Patch | With Patch (p=16) | Notes |
| ------------ | -------- | ----------------- | ----- |
| Hyena        | -        | -                 | -     |
| Attention    | -        | -                 | -     |

______________________________________________________________________

## Job Submission Log

| Date | Job ID | Config | Status | Val Acc | Notes |
| ---- | ------ | ------ | ------ | ------- | ----- |
| -    | -      | -      | -      | -       | -     |

______________________________________________________________________

## Notes

- All models use ImageNet-1k (1000 classes)
- Training uses AdamW optimizer with cosine LR schedule and 5% warmup
- Default training: 600,000 iterations
- Batch size: 32

______________________________________________________________________

**Last Updated**: 2026-01-30
**Status**: ⏳ Configs updated to ImageNet, ready for training
