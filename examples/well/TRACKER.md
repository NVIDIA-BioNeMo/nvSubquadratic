# The WELL Experiments Tracker

W&B project: [nvsubquadratic-well](https://wandb.ai/dafidofff/nvsubquadratic-well)

## Goal

We evaluate **Hyena** (subquadratic, global convolution-based) and **attention** (quadratic) as sequence mixers on [The Well](https://polymathic-ai.org/the_well/) PDE benchmark. The Well contains 23 datasets spanning diverse physical systems (fluid dynamics, MHD, astrophysics, reaction-diffusion, etc.) in 2D and 3D, making it a broad testbed for neural PDE solvers.

Both models share the same overall architecture (ViT5): a patchified encoder, a stack of residual blocks (each containing a sequence mixer + MLP), and an unpatchified decoder. The only difference is the mixer inside each block — Hyena uses a gated long convolution with a learned implicit kernel (SIREN), while attention uses standard scaled dot-product attention with QK-norm. By keeping everything else identical, we isolate the effect of the mixing mechanism on forecasting accuracy across physical domains.

The primary metric is **test/VRMSE** (variable-weighted root mean squared error), which normalizes across fields with different scales within each dataset.

### Experiment 1: Hyena vs Attention across datasets

Run both models on all Well datasets with matched hyperparameters. The leaderboard below tracks the best result per dataset.

### Experiment 2: Patch-size scaling (TBA)

Smaller patch sizes yield longer token sequences, which is where Hyena's subquadratic complexity should shine relative to attention's O(n^2) cost. Ablate both models across decreasing patch sizes, measuring test/VRMSE, throughput (samples/sec), and peak GPU memory.

______________________________________________________________________

## Ablation: `supernova_explosion_64`

Supernova serves as the initial ablation dataset before scaling to other Well datasets. It is a 3D dataset (64^3) with 5 physical fields — small enough for fast iteration but representative of 3D PDE dynamics.

**Ablation axes:**

1. **Backbone**: ResNet (`ResidualNetwork`) vs ViT5 (`ViT5GeneralPurposeNet`) — tests whether the ViT5 architecture (registers, layer scale, drop path) improves over the simpler ResNet.
1. **Sequence mixer**: Hyena vs Attention — the core comparison. Hyena uses a gated long convolution with a learned SIREN kernel; Attention uses scaled dot-product attention with QK-norm and RoPE.
1. **FiLM conditioning** (ViT5 + Hyena only): Tests whether conditioning the SIREN kernel via FiLM (register-pooled context) improves over the unconditioned variant.

**Shared hyperparameters (all configs):** patch_size=8, depth=10, LR=1e-3, 260k iterations, cosine schedule, 10% warmup, AdamW, bf16-mixed, weight_decay=1e-5, grad_clip=1.0. Patch size and depth chosen based on active_matter scaling results (see README).

| Config group | Hidden | MLP        | Batch | Notes                                  |
| ------------ | ------ | ---------- | ----- | -------------------------------------- |
| ResNet       | 512    | GLU exp=1  | 4     |                                        |
| ViT5         | 384    | GELU exp=4 | 2     | + 14 registers, layer scale, drop path |

> **TODO:** Align all supernova configs to patch_size=8, depth=10, LR=1e-3 before running.

### Ablation Results

| #   | Config                               | Model               | Params | FLOPs | W&B ID                                                                          | Epochs | val/VRMSE | test/VRMSE | test/NRMSE | Notes                              |
| --- | ------------------------------------ | ------------------- | ------ | ----- | ------------------------------------------------------------------------------- | ------ | --------- | ---------- | ---------- | ---------------------------------- |
| A1  | `cfg_hyena.py`                       | ResNet + Hyena      | 26.0M  | 26.4G | [`vlzxxwp9`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vlzxxwp9) | —      | —         | —          | —          | SLURM 147963, ivi-h1 geodude 2×GPU |
| A2  | `cfg_attention.py`                   | ResNet + Attention  | 24.9M  | 25.5G | [`f6ud6ris`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/f6ud6ris) | —      | —         | —          | —          | SLURM 147964, ivi-h1 geodude 2×GPU |
| A3  | `cfg_vit5_hyena.py`                  | ViT5 + Hyena        | 23.0M  | 23.8G | [`lycw7w1j`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lycw7w1j) | —      | —         | —          | —          | SLURM 147968, ivi-h1 all6000 1×GPU |
| A4  | `cfg_vit5_attention.py`              | ViT5 + Attention    | 22.8M  | 23.6G | [`8ygl9vn1`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/8ygl9vn1) | —      | —         | —          | —          | SLURM 147961, ivi-h1 all6000 1×GPU |
| A5  | `cfg_vit5_hyena_film_conditioned.py` | ViT5 + Hyena + FiLM | 23.4M  | 23.8G | —                                                                               | —      | —         | —          | —          |                                    |
| A6  | `cfg_vit5_hyena_3d.py`               | ViT5 + Hyena 3D     | 23.3M  | 23.9G | —                                                                               | —      | —         | —          | —          |                                    |

______________________________________________________________________

## Leaderboard (Best per Dataset)

Each row tracks the **best-scoring run** for a dataset. Only datasets with configs are listed.

| Dataset                  | Dim | Resolution | Hyena test/VRMSE | Hyena W&B | Attn test/VRMSE | Attn W&B | Notes |
| ------------------------ | --- | ---------- | ---------------- | --------- | --------------- | -------- | ----- |
| `supernova_explosion_64` | 3D  | 64^3       | —                | —         | —               | —        |       |
| `MHD_64`                 | 3D  | 64^3       | —                | —         | —               | —        |       |
| `active_matter`          | 2D  | 256x256    | —                | —         | —               | —        |       |

______________________________________________________________________

## Job Log

All completed runs. Failed, cancelled, and incomplete runs should be purged periodically.

| #   | Dataset                | Config                  | Model              | SLURM ID | Cluster              | W&B ID                                                                          | Epochs | val/VRMSE | test/VRMSE | test/NRMSE | Who | Notes   |
| --- | ---------------------- | ----------------------- | ------------------ | -------- | -------------------- | ------------------------------------------------------------------------------- | ------ | --------- | ---------- | ---------- | --- | ------- |
| A1  | supernova_explosion_64 | `cfg_hyena.py`          | ResNet + Hyena     | 147963   | ivi-h1 geodude 2×GPU | [`vlzxxwp9`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vlzxxwp9) | —      | —         | —          | —          | DW  | running |
| A2  | supernova_explosion_64 | `cfg_attention.py`      | ResNet + Attention | 147964   | ivi-h1 geodude 2×GPU | [`f6ud6ris`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/f6ud6ris) | —      | —         | —          | —          | DW  | running |
| A3  | supernova_explosion_64 | `cfg_vit5_hyena.py`     | ViT5 + Hyena       | 147968   | ivi-h1 all6000 1×GPU | [`lycw7w1j`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lycw7w1j) | —      | —         | —          | —          | DW  | running |
| A4  | supernova_explosion_64 | `cfg_vit5_attention.py` | ViT5 + Attention   | 147961   | ivi-h1 all6000 1×GPU | [`8ygl9vn1`](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/8ygl9vn1) | —      | —         | —          | —          | DW  | running |

______________________________________________________________________

## Observations

(none yet)

______________________________________________________________________

## How to download datasets

Use `scripts/download_well.sh` which wraps the `the-well-download` CLI:

```bash
# Download a full dataset (all splits)
bash scripts/download_well.sh <dataset_name>

# Download a specific split only
bash scripts/download_well.sh <dataset_name> --split train

# Custom destination
WELL_DATA_PATH=/scratch/data bash scripts/download_well.sh MHD_64
```

Run `bash scripts/download_well.sh` with no arguments to see all available datasets.

For **SLURM clusters**, submit via `slurm/download_well.sh` to avoid tying up a login node:

```bash
sbatch slurm/download_well.sh supernova_explosion_64
```

**NFS/shared filesystem note:** On some shared mounts (e.g. IVI), `curl --create-dirs` fails with permission errors when trying to `mkdir` parent directories. The SLURM download script works around this by **pre-creating the dataset directories** (`train/`, `valid/`, `test/`) before invoking the download. If you hit similar `mkdir`-related permission warnings when downloading interactively, manually create the target directories first:

```bash
DEST=/ivi/zfs/s0/original_homes/$USER/data/the_well/datasets/<dataset_name>/data
mkdir -p "$DEST"/{train,valid,test}
```

After downloading, verify integrity with:

```bash
python scripts/check_well_download.py /path/to/the_well/datasets/<dataset_name>
```

______________________________________________________________________
