# The WELL Experiments Tracker

W&B project:

## Goal

We evaluate **Hyena** (subquadratic, global convolution-based) and **attention** (quadratic) as sequence mixers on [The Well](https://polymathic-ai.org/the_well/) PDE benchmark. The Well contains 23 datasets spanning diverse physical systems (fluid dynamics, MHD, astrophysics, reaction-diffusion, etc.) in 2D and 3D, making it a broad testbed for neural PDE solvers.

Both models share the same overall architecture: a patchified encoder, a stack of residual blocks (each containing a sequence mixer + MLP), and an unpatchified decoder. The only difference is the mixer inside each block — Hyena uses a gated long convolution with a learned implicit kernel (SIREN), while attention uses standard scaled dot-product attention with QK-norm. By keeping everything else identical, we isolate the effect of the mixing mechanism on forecasting accuracy across physical domains.

The primary metric is **test/VRMSE** (variable-weighted root mean squared error), which normalizes across fields with different scales within each dataset.

### Experiment 1: Hyena vs Attention across datasets

Run both models on all 23 Well datasets with matched hyperparameters. The leaderboard below tracks the best result per dataset.

### Experiment 2: Patch-size scaling (TBA)

Smaller patch sizes yield longer token sequences, which is where Hyena's subquadratic complexity should shine relative to attention's O(n^2) cost. We ablate both models across decreasing patch sizes on one 2D dataset (TBA) and one 3D dataset (TBA), measuring test/VRMSE, throughput (samples/sec), and peak GPU memory. This tests the hypothesis that Hyena's advantage grows as sequence length increases.

______________________________________________________________________

## Leaderboard (Best per Dataset)

Each row tracks the **best-scoring model** for a dataset. Two runs per row: one Hyena, one Attention.

| Dataset                             | Dim | Resolution  | Hyena test/VRMSE | Hyena W&B                                                                  | Attn test/VRMSE | Attn W&B | Notes                       |
| ----------------------------------- | --- | ----------- | ---------------- | -------------------------------------------------------------------------- | --------------- | -------- | --------------------------- |
| `acoustic_scattering_discontinuous` | 2D  | 256x256     | —                | —                                                                          | —               | —        |                             |
| `acoustic_scattering_inclusions`    | 2D  | 256x256     | —                | —                                                                          | —               | —        |                             |
| `acoustic_scattering_maze`          | 2D  | 256x256     | —                | —                                                                          | —               | —        |                             |
| `active_matter`                     | 2D  | 256x256     | —                | —                                                                          | —               | —        |                             |
| `convective_envelope_rsg`           | 3D  | 256x128x256 | —                | —                                                                          | —               | —        | Spherical coords            |
| `euler_multi_quadrants_openBC`      | 2D  | 512x512     | —                | —                                                                          | —               | —        |                             |
| `euler_multi_quadrants_periodicBC`  | 2D  | 512x512     | —                | —                                                                          | —               | —        |                             |
| `gray_scott_reaction_diffusion`     | 2D  | 128x128     | —                | —                                                                          | —               | —        |                             |
| `helmholtz_staircase`               | 2D  | 1024x256    | —                | —                                                                          | —               | —        | Non-square                  |
| `MHD_64`                            | 3D  | 64^3        | —                | —                                                                          | —               | —        | Periodic BC                 |
| `MHD_256`                           | 3D  | 256^3       | —                | —                                                                          | —               | —        | Periodic BC, large          |
| `planetswe`                         | 2D  | 256x512     | —                | —                                                                          | —               | —        | Angular coords              |
| `post_neutron_star_merger`          | 3D  | 192x128x66  | —                | —                                                                          | —               | —        | Log-spherical               |
| `rayleigh_benard`                   | 2D  | 512x128     | —                | —                                                                          | —               | —        |                             |
| `rayleigh_benard_uniform`           | 2D  | —           | —                | —                                                                          | —               | —        |                             |
| `rayleigh_taylor_instability`       | 3D  | 128^3       | —                | —                                                                          | —               | —        |                             |
| `shear_flow`                        | 2D  | 128x256     | —                | —                                                                          | —               | —        |                             |
| `supernova_explosion_64`            | 3D  | 64^3        | 0.3674           | [`p7te253r`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/p7te253r) | —               | —        | Hyena (no FiLM) best so far |
| `supernova_explosion_128`           | 3D  | 128^3       | —                | —                                                                          | —               | —        |                             |
| `turbulence_gravity_cooling`        | 3D  | 64^3        | —                | —                                                                          | —               | —        |                             |
| `turbulent_radiative_layer_2D`      | 2D  | 128x384     | —                | —                                                                          | —               | —        |                             |
| `turbulent_radiative_layer_3D`      | 3D  | 128x128x256 | —                | —                                                                          | —               | —        |                             |
| `viscoelastic_instability`          | 2D  | 512x512     | —                | —                                                                          | —               | —        |                             |

______________________________________________________________________

## Job Log

All runs (including ablations, reruns, and failed jobs). Tracks every experiment launched.

| #   | Dataset                  | Config                               | Model               | W&B ID                                                                     | SLURM Job ID | Iterations | Epochs | val/VRMSE | test/VRMSE | test/NRMSE | Status    | Who   | Notes                               |
| --- | ------------------------ | ------------------------------------ | ------------------- | -------------------------------------------------------------------------- | ------------ | ---------- | ------ | --------- | ---------- | ---------- | --------- | ----- | ----------------------------------- |
| 1   | `supernova_explosion_64` | `cfg_hyena.py`                       | ResNet + Hyena      | [`z6o20go9`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/z6o20go9) | —            | 130k       | ~15    | 0.3266    | —          | —          | Completed | David | Historical baseline, only 15 epochs |
| 2   | `supernova_explosion_64` | `cfg_vit5_hyena.py`                  | ViT5 + Hyena        | [`p7te253r`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/p7te253r) | —            | 260k       | ~30    | 0.3561    | 0.3674     | 0.3643     | Completed | David | Best Hyena result on supernova      |
| 3   | `supernova_explosion_64` | `cfg_vit5_hyena_film_conditioned.py` | ViT5 + Hyena + FiLM | [`77n2mp0a`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/77n2mp0a) | —            | 260k       | ~30    | 0.3615    | 0.3727     | 0.3696     | Completed | David | FiLM did not help on this dataset   |
| 4   | `supernova_explosion_64` | `cfg_vit5_attention.py`              | ViT5 + Attention    | —                                                                          | —            | 260k       | ~30    | —         | —          | —          | Resuming  | David | Resuming z6o20go9 to 30 epochs      |

______________________________________________________________________

## Observations

1. **Plain Hyena > FiLM-conditioned Hyena on supernova_explosion_64.** ViT5 Hyena ([`p7te253r`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/p7te253r)) test/VRMSE `0.3674` vs FiLM ([`77n2mp0a`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/77n2mp0a)) `0.3727`. FiLM conditioning on the kernel does not appear to help here.

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
