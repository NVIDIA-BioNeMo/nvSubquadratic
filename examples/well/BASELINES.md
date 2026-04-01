# The Well — Paper Baselines

Reference results from the [original Well paper](https://arxiv.org/abs/2412.00568) (Appendix E.1).

## Setup

- **Compute budget**: 12 hours on a single NVIDIA H100 per run
- **Input**: 4 timesteps (history)
- **Target**: next-step prediction
- **Model size**: all scaled to ~15–20M parameters (2D datasets)
- **Primary metric**: VRMSE (variance-scaled root mean squared error, lower is better)
  - A score of 1.0 means the model predicts the mean field value (no skill)

## Model Architectures & Hyperparameters

### FNO (Fourier Neural Operator)

| Hyperparameter               | Value  |
| ---------------------------- | ------ |
| Spectral filter size (modes) | 16     |
| Hidden dimension             | 128    |
| Blocks                       | 4      |
| ~Parameters                  | 15–20M |

### TFNO (Tucker-Factorized FNO)

| Hyperparameter               | Value  |
| ---------------------------- | ------ |
| Spectral filter size (modes) | 16     |
| Hidden dimension             | 128    |
| Blocks                       | 4      |
| ~Parameters                  | 15–20M |

### U-net (Classic)

| Hyperparameter      | Value  |
| ------------------- | ------ |
| Spatial filter size | 3      |
| Initial dimension   | 48     |
| Blocks per stage    | 1      |
| Up/Down blocks      | 4      |
| Bottleneck blocks   | 1      |
| ~Parameters         | 15–20M |

### CNextU-net (ConvNeXt U-net)

| Hyperparameter      | Value  |
| ------------------- | ------ |
| Spatial filter size | 7      |
| Initial dimension   | 42     |
| Blocks per stage    | 2      |
| Up/Down blocks      | 4      |
| Bottleneck blocks   | 1      |
| ~Parameters         | 15–20M |

## One-Step VRMSE (Table 2)

| Dataset                               | FNO         | TFNO       | U-net      | CNextU-net |
| ------------------------------------- | ----------- | ---------- | ---------- | ---------- |
| acoustic_scattering (maze)            | 0.5062      | 0.5057     | 0.0351     | **0.0153** |
| active_matter                         | 0.3691      | 0.3598     | 0.2489     | **0.1034** |
| convective_envelope_rsg               | **0.0269**  | 0.0283     | 0.0555     | 0.0799     |
| euler_multi_quadrants (periodic BC)   | 0.4081      | 0.4163     | 0.1834     | **0.1531** |
| gray_scott_reaction_diffusion         | **0.1365**  | 0.3633     | 0.2252     | 0.1761     |
| helmholtz_staircase                   | **0.00046** | 0.00346    | 0.01931    | 0.02758    |
| MHD_64                                | 0.3605      | 0.3561     | 0.1798     | **0.1633** |
| MHD_256                               | —           | —          | —          | —          |
| planetswe                             | 0.1727      | **0.0853** | 0.3620     | 0.3724     |
| post_neutron_star_merger              | 0.3866      | **0.3793** | —          | —          |
| rayleigh_benard                       | 0.8395      | **0.6566** | 1.4860     | 0.6699     |
| rayleigh_taylor_instability (At=0.25) | >10         | >10        | >10        | >10        |
| shear_flow                            | 1.189       | 1.472      | 3.447      | **0.8080** |
| supernova_explosion_64                | 0.3783      | 0.3785     | **0.3063** | 0.3181     |
| turbulence_gravity_cooling            | 0.2429      | 0.2673     | 0.6753     | **0.2096** |
| turbulent_radiative_layer_2D          | 0.5001      | 0.5016     | 0.2418     | **0.1956** |
| turbulent_radiative_layer_3D          | 0.5278      | 0.5187     | 0.3728     | **0.3667** |
| viscoelastic_instability              | 0.7212      | 0.7102     | 0.4185     | **0.2499** |

> MHD_256 and post_neutron_star_merger U-net/CNextU-net entries were not reported in the paper (likely OOM on 256³ or incomplete runs).

## Rollout VRMSE (Table 3)

Time-averaged VRMSE over two windows: steps 6–12 and steps 13–30.

| Dataset                       | FNO 6:12  | FNO 13:30 | TFNO 6:12 | TFNO 13:30 | U-net 6:12 | U-net 13:30 | CNextU 6:12 | CNextU 13:30 |
| ----------------------------- | --------- | --------- | --------- | ---------- | ---------- | ----------- | ----------- | ------------ |
| acoustic_scattering (maze)    | 1.06      | 1.72      | 1.13      | 1.23       | **0.56**   | 0.92        | 0.78        | 1.13         |
| active_matter                 | >10       | >10       | 7.52      | **4.72**   | 2.53       | 2.62        | **2.11**    | 2.71         |
| convective_envelope_rsg       | **0.28**  | **0.47**  | 0.32      | 0.65       | 0.76       | 2.16        | 1.15        | 1.59         |
| euler_multi_quadrants         | **1.13**  | **1.37**  | 1.23      | 1.52       | 1.02       | 1.63        | 4.98        | >10          |
| gray_scott_reaction_diffusion | 0.89      | >10       | 1.54      | >10        | 0.57       | >10         | **0.29**    | **7.62**     |
| helmholtz_staircase           | **0.002** | **0.003** | 0.011     | 0.019      | 0.057      | 0.097       | 0.110       | 0.194        |
| MHD_64                        | 1.24      | 1.61      | 1.25      | 1.81       | 1.65       | 4.66        | **1.30**    | **2.23**     |
| planetswe                     | 0.81      | 2.96      | **0.29**  | **0.55**   | 1.18       | 1.92        | 0.42        | 0.52         |
| post_neutron_star_merger      | 0.76      | 1.05      | **0.70**  | **1.05**   | —          | —           | —           | —            |
| rayleigh_benard               | >10       | >10       | >10       | >10        | >10        | >10         | >10         | >10          |
| rayleigh_taylor_instability   | >10       | >10       | **6.72**  | >10        | >10        | **2.84**    | >10         | 7.43         |
| shear_flow                    | >10       | >10       | >10       | >10        | >10        | >10         | **2.33**    | >10          |
| supernova_explosion_64        | 2.41      | >10       | 1.86      | >10        | **0.94**   | **1.69**    | 1.12        | 4.55         |
| turbulence_gravity_cooling    | 3.55      | 5.63      | 4.49      | 6.95       | 7.14       | 4.15        | **1.30**    | **2.09**     |
| turbulent_radiative_layer_2D  | 1.79      | 3.54      | 6.01      | >10        | 0.66       | 1.04        | **0.54**    | **1.01**     |
| turbulent_radiative_layer_3D  | **0.81**  | 0.94      | >10       | >10        | 0.95       | 1.09        | 0.77        | **0.86**     |
| viscoelastic_instability      | 4.11      | —         | 0.93      | —          | 0.89       | —           | **0.52**    | —            |

## Training Hyperparameters

### Shared across all models

| Hyperparameter    | Value                                                               |
| ----------------- | ------------------------------------------------------------------- |
| Optimizer         | AdamW                                                               |
| Weight decay      | 0.01 (PyTorch default)                                              |
| Loss              | MSE averaged over fields and space                                  |
| Precision         | fp32 (single precision — mixed caused instabilities)                |
| Compute budget    | 12 hours on 1× NVIDIA H100                                          |
| Batch size        | Maximized to fill GPU memory per dataset                            |
| LR search         | Coarse grid: {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}                         |
| Input timesteps   | 4                                                                   |
| Boundary handling | Naive per architecture (periodic for FNO/TFNO, zero-pad for U-nets) |

### Best learning rate and epochs per dataset (Table 6)

Format: LR (epochs completed in 12h)

| Dataset                               | FNO        | TFNO       | U-net      | CNextU-net |
| ------------------------------------- | ---------- | ---------- | ---------- | ---------- |
| acoustic_scattering (maze)            | 1e-3 (27)  | 1e-3 (27)  | 1e-2 (26)  | 1e-3 (10)  |
| active_matter                         | 5e-3 (239) | 1e-3 (243) | 5e-3 (239) | 5e-3 (156) |
| convective_envelope_rsg               | 1e-4 (14)  | 1e-3 (13)  | 5e-4 (19)  | 1e-4 (5)   |
| euler_multi_quadrants (periodic BC)   | 5e-4 (4)   | 5e-4 (4)   | 1e-3 (4)   | 5e-3 (1)   |
| gray_scott_reaction_diffusion         | 1e-3 (46)  | 5e-3 (45)  | 1e-2 (44)  | 1e-4 (15)  |
| helmholtz_staircase                   | 5e-4 (132) | 5e-4 (131) | 1e-3 (120) | 5e-4 (47)  |
| MHD_64                                | 5e-3 (170) | 1e-3 (155) | 5e-4 (165) | 5e-3 (59)  |
| planetswe                             | 5e-4 (49)  | 5e-4 (49)  | 1e-2 (49)  | 1e-2 (18)  |
| post_neutron_star_merger              | 5e-4 (104) | 5e-4 (99)  | —          | —          |
| rayleigh_benard                       | 1e-4 (32)  | 1e-4 (31)  | 1e-4 (29)  | 5e-4 (12)  |
| rayleigh_taylor_instability (At=0.25) | 5e-3 (177) | 1e-4 (175) | 5e-4 (193) | 5e-3 (56)  |
| shear_flow                            | 1e-3 (24)  | 1e-3 (24)  | 5e-4 (29)  | 5e-4 (9)   |
| supernova_explosion_64                | 1e-4 (40)  | 1e-4 (35)  | 5e-4 (46)  | 5e-4 (13)  |
| turbulence_gravity_cooling            | 1e-4 (13)  | 5e-4 (10)  | 1e-3 (14)  | 1e-3 (3)   |
| turbulent_radiative_layer_2D          | 5e-3 (500) | 1e-3 (500) | 5e-3 (500) | 5e-3 (495) |
| turbulent_radiative_layer_3D          | 1e-3 (12)  | 5e-4 (12)  | 5e-4 (13)  | 5e-3 (3)   |
| viscoelastic_instability              | 5e-3 (205) | 5e-3 (199) | 5e-4 (198) | 5e-4 (114) |

> Datasets with very few epochs (euler_multi_quadrants, turbulence_gravity_cooling, turbulent_radiative_layer_3D, convective_envelope_rsg) were data-I/O bound — the 12h budget was insufficient to complete more than ~5 epochs. Non-time-limited training could improve results on these.

## Key Observations from the Paper

1. **No single model dominates**: CNextU-net wins 8/17 one-step, but spectral models (FNO/TFNO) win 8/17 — there's a genuine split depending on the physics.
1. **Rollout stability is hard**: even short rollouts (13–30 steps) degrade significantly from one-step performance for all models.
1. **Boundary conditions matter**: the performance split may relate to how each model family handles boundaries (spectral vs spatial), though the paper notes no clear trend.
1. **These are not SOTA**: the authors explicitly state these are "off-the-shelf" baselines, not tuned for peak performance. They are meant to establish a floor.

## Dataset Overview

| Dataset                       | Dim | Resolution  | Steps    | Trajectories |
| ----------------------------- | --- | ----------- | -------- | ------------ |
| acoustic_scattering (maze)    | 2D  | 256×256     | 200      | 2400         |
| active_matter                 | 2D  | 256×256     | 200      | 2250         |
| convective_envelope_rsg       | 2D  | 128×128     | 500      | 5            |
| euler_multi_quadrants         | 2D  | 512×512     | 101      | 3000         |
| gray_scott_reaction_diffusion | 2D  | 128×128     | 1001     | 1200         |
| helmholtz_staircase           | 2D  | 256×256     | 200      | 2500         |
| MHD_64                        | 3D  | 64³         | 20       | 40           |
| MHD_256                       | 3D  | 256³        | 20       | 40           |
| planetswe                     | 2D  | 256×512     | 200      | 1560         |
| post_neutron_star_merger      | 2D  | 512×512     | 200      | 200          |
| rayleigh_benard               | 2D  | 512×128     | 200      | 1020         |
| rayleigh_taylor_instability   | 2D  | 128×512     | 501      | 4680         |
| shear_flow                    | 2D  | 256×256     | 200      | 1080         |
| supernova_explosion_64        | 3D  | 64³         | 101      | 1000         |
| turbulence_gravity_cooling    | 2D  | 256×256     | 200      | 2500         |
| turbulent_radiative_layer_2D  | 2D  | 128×384     | 101      | 90           |
| turbulent_radiative_layer_3D  | 3D  | 128×128×256 | 101      | 90           |
| viscoelastic_instability      | 2D  | 512×512     | variable | 260          |
