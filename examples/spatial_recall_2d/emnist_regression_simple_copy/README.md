# EMNIST Spatial Recall 2D - Model Configurations

Configs for the EMNIST Spatial Recall 2D regression task with different sequence mixers.

## Task Description

- **Input**: 64×64 grayscale canvas with EMNIST digit
- **Target**: 16×16 region containing the digit
- **Objective**: Regress the target region from the full canvas

## Model Sizes

All models use **4 blocks** with **GLU MLP** (expansion_factor=1.0).

### XS (Extra-Small) ~0.7-0.8M params

| Config                  | Architecture | Hidden | Heads | Params |
| ----------------------- | ------------ | ------ | ----- | ------ |
| `ccnn_hyena_xs`         | Hyena        | 160    | -     | 0.77M  |
| `ccnn_attn_xs`          | Attention    | 160    | 8     | 0.72M  |
| `ccnn_mamba_xs`         | Mamba        | 96     | 6     | 0.78M  |
| `ccnn_attn_patchify_xs` | Attn+Patch   | 160    | 8     | 0.74M  |

### S (Small) ~1.8-2.0M params

| Config                 | Architecture | Hidden | Heads | Params |
| ---------------------- | ------------ | ------ | ----- | ------ |
| `ccnn_hyena_s`         | Hyena        | 256    | -     | 1.91M  |
| `ccnn_attn_s`          | Attention    | 256    | 8     | 1.84M  |
| `ccnn_mamba_s`         | Mamba        | 160    | 10    | 1.91M  |
| `ccnn_attn_patchify_s` | Attn+Patch   | 256    | 8     | 1.87M  |

## Architecture Details

### Hyena

- Uses CKConvND with SIREN kernel for global convolution
- QKV projections via QKVSequenceMixer
- No explicit heads (continuous kernel)

### Attention

- Multi-head self-attention
- QK normalization + RoPE enabled
- **Design**: Fix `num_heads=8`, head_dim scales with model size

| Size | hidden_dim | num_heads | head_dim |
| ---- | ---------- | --------- | -------- |
| XS   | 160        | 8         | 20       |
| S    | 256        | 8         | 32       |

### Mamba

- Mamba2 with bidirectional processing
- More parameter-efficient → needs smaller hidden_dim to match params
- **Design**: Fix `headdim=32`, heads scale with model size

| Size | hidden_dim | expand | inner_dim | headdim | heads |
| ---- | ---------- | ------ | --------- | ------- | ----- |
| XS   | 96         | 2      | 192       | 32      | 6     |
| S    | 160        | 2      | 320       | 32      | 10    |

> **Note**: "Heads" in Mamba vs Attention are different concepts. Attention heads split hidden_dim for parallel attention patterns. Mamba heads partition the expanded SSM state space. We keep `headdim` constant (like head_dim in some ViT variants) rather than `num_heads`.

### Attention + Patchify

- ViT-style patchification (patch_size=8, non-overlapping)
- Reduces sequence from 64×64=4096 to 8×8=64 tokens
- Same attention as above, but operates on patches

## Running Experiments

```bash
# Activate environment
conda activate nvsubq
source .env  # For WandB API key
export PYTHONPATH=.

# Run on GPU node
srun --gres=gpu:1 -c 16 --partition low --nodelist cxis-[0-35] \
    python experiments/run.py --config examples/spatial_recall_2d/emnist_regression_simple_copy/ccnn_hyena_xs.py
```

## Notes

- **Mamba requires GPU** due to triton dependency (configs won't load on CPU)
- All models trained for 20K iterations (~2 epochs @ batch_size=64)
- Parameter counts are closely matched for fair comparison
