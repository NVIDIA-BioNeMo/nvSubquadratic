# Experiment Overview — Paper Results

Which numbers are already in the paper, and which runs are still needed to fill the remaining gaps.

Status: ✅ in paper · 🔄 running · 📝 needed · — not applicable / expected OOM

______________________________________________________________________

## 1. ImageNet-1k Classification

### Table 3 — Top-1 at standard p=16 (comparison to external baselines)

| Model                                       | Type        | Patch Merge | Params | Top-1 (%) |
| ------------------------------------------- | ----------- | ----------- | ------ | --------- |
| DeiT-S                                      | Attention   | ✗           | 22M    | 79.8      |
| Swin-T                                      | Attention   | ✓           | 29M    | 81.3      |
| ConvNeXt-T                                  | Convolution | ✓           | 29M    | 82.1      |
| Vim-S                                       | GRM (1D)    | ✗           | 26M    | 80.3      |
| VMamba-T                                    | GRM (1D)    | ✓           | 30M    | 82.6      |
| ViT-5-Small (Attention)†                    | Attention   | ✗           | 22M    | 81.8 ✅   |
| HyenaND-S (pure)                            | GCM (2D)    | ✗           | 22M    | 81.5 ✅   |
| HyenaND-S (pure) + FiLM                     | GCM (2D)    | ✗           | TBD    | 📝 needed |
| HyenaND-S (pure, FLOP-matched to Attention) | GCM (2D)    | ✗           | TBD    | 📝 needed |
| HyenaND-S (pure) + patch merging            | GCM (2D)    | ✓           | TBD    | 📝 needed |
| HyenaND-S (pure) + FiLM + patch merging     | GCM (2D)    | ✓           | TBD    | 📝 needed |
| HyenaND-S (HA)×6                            | Hybrid      | ✗           | 22M    | 82.1 ✅   |
| HyenaND-S (HA)×6 + FiLM                     | Hybrid      | ✗           | TBD    | 📝 needed |
| HyenaND-S (HHHA)×3                          | Hybrid      | ✗           | 22M    | 82.0 ✅   |
| HyenaND-S (HHHA)×3 + FiLM                   | Hybrid      | ✗           | TBD    | 📝 needed |

______________________________________________________________________

4x4 attention, hyena...

> HyenaND-S (pure) + FiLM at p=16 = 81.8% ✅ (W&B `peeaqdkq`, v3 LAMB run). Hybrid + FiLM not yet run.
> Patch-merging configs do not yet exist for HyenaND.

______________________________________________________________________

### Table 13 — Patch-size ablation (Top-1 / GFLOPs)

GFLOPs shown in parentheses.

| Model                     | p=16           | p=8            | p=4           |
| ------------------------- | -------------- | -------------- | ------------- |
| Attention                 | 81.8 ✅ (9.4)  | 84.3 ✅ (45.5) | 85.1 ✅ (317) |
| HyenaND-S (pure)          | 81.5 ✅ (10.0) | 83.7 ✅ (39.0) | 84.0 ✅ (155) |
| HyenaND-S (pure) + FiLM   | 📝             | 📝             | 📝            |
| HyenaND-S (HA)×6          | 82.1 ✅ (9.7)  | 84.2 ✅ (42.3) | 85.0 ✅ (236) |
| HyenaND-S (HA)×6 + FiLM   | 📝             | 📝             | 📝            |
| HyenaND-S (HHHA)×3        | 82.0 ✅ (9.8)  | 84.0 ✅ (40.7) | 84.4 ✅ (196) |
| HyenaND-S (HHHA)×3 + FiLM | 📝             | 📝             | 📝            |

> FiLM GFLOPs ≈ same as non-FiLM (small SIREN-MLP overhead, negligible vs attention savings).
> v5_patch configs for Hyena+FiLM at all patch sizes are set up and ready in [examples/vit5_imagenet/v5_patch/](examples/vit5_imagenet/v5_patch/). Hybrid+FiLM configs are in [examples/vit5_imagenet/vit5_hybrid/](examples/vit5_imagenet/vit5_hybrid/).
> FLOP-matched and patch-merging variants live in Table 3 only — they're targeted comparisons against external baselines at p=16, not patch-sweep ablations.

______________________________________________________________________

## 3. The Well — PDE Surrogate Modeling

### Table 14 — Full-table view across all Cartesian Well datasets (val VRMSE, single H100, 24h / 110k iters)

Bold = best per (dataset, patch-size) column. 5 datasets in paper ✅; 9 datasets 📝 still to run (Cartesian scope from T1–T14 in [examples/well/v2/TRACKER.md](examples/well/v2/TRACKER.md)).

| Dataset                       | Res            | Model      | p=8           | p=4           | p=2           | full res      |
| ----------------------------- | -------------- | ---------- | ------------- | ------------- | ------------- | ------------- |
| acoustic_scattering_maze      | 2D 256²        | CNextU-net | N/A           | N/A           | N/A           | **0.0082** ✅ |
| acoustic_scattering_maze      | 2D 256²        | Attention  | 0.0456 ✅     | 0.0569 ✅     | 0.1057 ✅     | —             |
| acoustic_scattering_maze      | 2D 256²        | HyenaND    | 0.0086 ✅     | **0.0068** ✅ | **0.0062** ✅ | —             |
| active_matter                 | 2D 256²        | CNextU-net | N/A           | N/A           | N/A           | 0.0347 ✅     |
| active_matter                 | 2D 256²        | Attention  | 0.0586 ✅     | 0.0616 ✅     | 0.0914 ✅     | —             |
| active_matter                 | 2D 256²        | HyenaND    | **0.0073** ✅ | **0.0080** ✅ | **0.0070** ✅ | —             |
| euler_multi_quadrants         | 2D 512²        | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| euler_multi_quadrants         | 2D 512²        | Attention  | 📝            | 📝            | 📝            | —             |
| euler_multi_quadrants         | 2D 512²        | HyenaND    | 📝            | 📝            | 📝            | —             |
| gray_scott_reaction_diffusion | 2D 128²        | CNextU-net | N/A           | N/A           | N/A           | 0.2319 ✅     |
| gray_scott_reaction_diffusion | 2D 128²        | Attention  | 0.0520 ✅     | 0.0538 ✅     | 0.0974 ✅     | —             |
| gray_scott_reaction_diffusion | 2D 128²        | HyenaND    | **0.0092** ✅ | **0.0090** ✅ | **0.0091** ✅ | —             |
| helmholtz_staircase           | 2D 1024×256    | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| helmholtz_staircase           | 2D 1024×256    | Attention  | 📝            | 📝            | 📝            | —             |
| helmholtz_staircase           | 2D 1024×256    | HyenaND    | 📝            | 📝            | 📝            | —             |
| MHD_64                        | 3D 64³         | CNextU-net | N/A           | N/A           | N/A           | **0.2108** ✅ |
| MHD_64                        | 3D 64³         | Attention  | 0.3044 ✅     | 0.2164 ✅     | 0.3037 ✅     | —             |
| MHD_64                        | 3D 64³         | HyenaND    | 0.2810 ✅     | **0.1088** ✅ | **0.0543** ✅ | —             |
| rayleigh_benard               | 2D 512×128     | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| rayleigh_benard               | 2D 512×128     | Attention  | 📝            | 📝            | 📝            | —             |
| rayleigh_benard               | 2D 512×128     | HyenaND    | 📝            | 📝            | 📝            | —             |
| rayleigh_taylor_instability   | 2D 128×512     | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| rayleigh_taylor_instability   | 2D 128×512     | Attention  | 📝            | 📝            | 📝            | —             |
| rayleigh_taylor_instability   | 2D 128×512     | HyenaND    | 📝            | 📝            | 📝            | —             |
| shear_flow                    | 2D 128×256     | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| shear_flow                    | 2D 128×256     | Attention  | 📝            | 📝            | 📝            | —             |
| shear_flow                    | 2D 128×256     | HyenaND    | 📝            | 📝            | 📝            | —             |
| supernova_explosion_64        | 3D 64³         | CNextU-net | N/A           | N/A           | N/A           | 0.7400 ✅     |
| supernova_explosion_64        | 3D 64³         | Attention  | **0.6117** ✅ | 0.3879 ✅     | 0.3000 ✅     | —             |
| supernova_explosion_64        | 3D 64³         | HyenaND    | 0.6151 ✅     | **0.3578** ✅ | **0.1943** ✅ | —             |
| turbulence_gravity_cooling    | 3D 64³         | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| turbulence_gravity_cooling    | 3D 64³         | Attention  | 📝            | 📝            | 📝            | —             |
| turbulence_gravity_cooling    | 3D 64³         | HyenaND    | 📝            | 📝            | 📝            | —             |
| turbulent_radiative_layer_2D  | 2D 128×384     | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| turbulent_radiative_layer_2D  | 2D 128×384     | Attention  | 📝            | 📝            | 📝            | —             |
| turbulent_radiative_layer_2D  | 2D 128×384     | HyenaND    | 📝            | 📝            | 📝            | —             |
| turbulent_radiative_layer_3D  | 3D 128×128×256 | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| turbulent_radiative_layer_3D  | 3D 128×128×256 | Attention  | 📝            | 📝            | 📝            | —             |
| turbulent_radiative_layer_3D  | 3D 128×128×256 | HyenaND    | 📝            | 📝            | 📝            | —             |
| viscoelastic_instability      | 2D 512²        | CNextU-net | N/A           | N/A           | N/A           | 📝            |
| viscoelastic_instability      | 2D 512²        | Attention  | 📝            | 📝            | 📝            | —             |
| viscoelastic_instability      | 2D 512²        | HyenaND    | 📝            | 📝            | 📝            | —             |

**In paper (✅, 5 datasets):** acoustic_scattering_maze, active_matter, gray_scott_reaction_diffusion, MHD_64, supernova_explosion_64.
**Still to run (📝, 9 datasets):** euler_multi_quadrants, helmholtz_staircase, rayleigh_benard, rayleigh_taylor_instability, shear_flow, turbulence_gravity_cooling, turbulent_radiative_layer_2D, turbulent_radiative_layer_3D, viscoelastic_instability.

> Excluded from this table (non-Cartesian, per the v2 tracker): `convective_envelope_rsg`, `planetswe`, `post_neutron_star_merger`, `MHD_256`. Per-dataset LRs and trajectory counts: see T1–T14 in [examples/well/v2/TRACKER.md](examples/well/v2/TRACKER.md).

### 4. - The Well non-cartesian

### Table 15 — Full-table view across all non-Cartesian Well datasets (val VRMSE, single H100, 24h / 110k iters)

Bold = best per (dataset, patch-size) column. 0 datasets in paper ✅; 4 datasets 📝 still to run. These were excluded from the v2 Cartesian scope; CNextU-net values are paper baselines (The Well, Ohana et al. 2024).

| Dataset                  | Res            | Grid          | Model      | p=8 | p=4 | p=2 | full res  |
| ------------------------ | -------------- | ------------- | ---------- | --- | --- | --- | --------- |
| convective_envelope_rsg  | 3D 256×128×256 | Spherical     | CNextU-net | N/A | N/A | N/A | 0.0799 ✅ |
| convective_envelope_rsg  | 3D 256×128×256 | Spherical     | Attention  | 📝  | 📝  | 📝  | —         |
| convective_envelope_rsg  | 3D 256×128×256 | Spherical     | HyenaND    | 📝  | 📝  | 📝  | —         |
| MHD_256                  | 3D 256³        | Cartesian     | CNextU-net | N/A | N/A | N/A | —         |
| MHD_256                  | 3D 256³        | Cartesian     | Attention  | 📝  | 📝  | 📝  | —         |
| MHD_256                  | 3D 256³        | Cartesian     | HyenaND    | 📝  | 📝  | 📝  | —         |
| planetswe                | 2D 256×512     | Equiangular   | CNextU-net | N/A | N/A | N/A | 0.3724 ✅ |
| planetswe                | 2D 256×512     | Equiangular   | Attention  | 📝  | 📝  | 📝  | —         |
| planetswe                | 2D 256×512     | Equiangular   | HyenaND    | 📝  | 📝  | 📝  | —         |
| post_neutron_star_merger | 3D 192×128×66  | Log-spherical | CNextU-net | N/A | N/A | N/A | —         |
| post_neutron_star_merger | 3D 192×128×66  | Log-spherical | Attention  | 📝  | 📝  | 📝  | —         |
| post_neutron_star_merger | 3D 192×128×66  | Log-spherical | HyenaND    | 📝  | 📝  | 📝  | —         |

**In paper (✅, 0 datasets):** none.
**Still to run (📝, 4 datasets):** convective_envelope_rsg, MHD_256, planetswe, post_neutron_star_merger.

> CNextU-net baselines (FNO/TFNO/U-net/CNextU-net column) from [examples/well/BASELINES.md](examples/well/BASELINES.md). `MHD_256` and `post_neutron_star_merger` U-net/CNextU-net entries were not reported in the original paper (likely OOM on 256³ / incomplete runs).
> Non-Cartesian geometries require coordinate-aware mixers; HyenaND/Attention at full resolution likely OOMs at these sizes — patch-based variants are the realistic targets.
> Scope rationale: these were explicitly excluded from the v2 Cartesian scope (see [examples/well/v2/TRACKER.md](examples/well/v2/TRACKER.md)) because spherical / equiangular / log-spherical grids break the implicit Cartesian assumption in our positional encodings and FFT kernels.
