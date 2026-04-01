# The Well Experiments — v2 Tracker

W&B project: [nvsubquadratic-well](https://wandb.ai/implicit-long-convs/nvsubquadratic)

> **Status**: Planning phase. v1 runs are ongoing (MHD_64). This tracker supersedes v1 with a more rigorous experimental design for a NeurIPS submission.

______________________________________________________________________

## Research Question

Can a **subquadratic global sequence mixer** (Hyena) match or outperform the baselines on The Well PDE benchmark across diverse physical systems, while scaling more favourably with sequence length than attention?

______________________________________________________________________

## Hypotheses

| #   | Hypothesis                                                                                                                                                    | How We Test It                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| H1  | Hyena ≥ CNextU-net (VRMSE) on the majority of Cartesian Well datasets                                                                                         | Phase 4: full 14-dataset table, 3 seeds                                        |
| H2  | At equal full resolution (same UNet backbone), UNet-Hyena outperforms UNet-ConvNeXt and UNet-Attention                                                        | Phase 2: UNet-{ConvNeXt, Hyena, Attention} on helmholtz, gray_scott, supernova |
| H3  | The VRMSE gap between full-resolution UNet and patch-based Hyena narrows as patch size decreases — confirming the gap is resolution access, not mixer quality | Phase 3: patch sizes 16→1 on active_matter and supernova                       |
| H4  | Hyena's advantage over Attention in VRMSE and memory efficiency grows with sequence length (smaller patch → more tokens)                                      | Phase 3: VRMSE + peak memory + throughput vs token count                       |

> Phases 0 and 1 (precision and LR ablations) are **prerequisite experiments**, not hypothesis-testing phases. They establish the correct training setup before testing H1–H4.

## Dataset Selection

### Ablation Datasets (run all architecture variants here)

These datasets are used across Phases 0–3. Criteria: Cartesian, sufficient data for low-variance estimates, representative of diverse PDE regimes (high-freq, stiff, 3D, 2D turbulence).

| Dataset                         | Dim | Resolution | Train Trajs | Size  | CNextU-net VRMSE | Phases     | Rationale                                                                            |
| ------------------------------- | --- | ---------- | ----------- | ----- | ---------------- | ---------- | ------------------------------------------------------------------------------------ |
| `helmholtz_staircase`           | 2D  | 1024×256   | ~410        | 52GB  | 0.02758          | 0, 1, 2    | High-frequency PDE — primary stress test for bf16 precision and long-sequence mixing |
| `gray_scott_reaction_diffusion` | 2D  | 128×128    | ~960        | 154GB | 0.1761           | 0, 1, 2    | Precision-sensitive stiff PDE; 1001 steps → long rollout; fast to train              |
| `supernova_explosion_64`        | 3D  | 64³        | ~800        | 268GB | 0.3181           | 0, 1, 2, 3 | Representative 3D; close race with U-net (0.3063); tests 3D scaling                  |
| `active_matter`                 | 2D  | 256×256    | ~288        | 51GB  | 0.1034           | 3          | Phase 3 scaling study only; fast 2D iteration, CNextU-net wins                       |

> **Development / sanity-check dataset**: `turbulent_radiative_layer_2D` (6.9GB, ~72 train trajs). Use for testing new configs and debugging before committing to ablation runs. Do not report results from this dataset as primary evidence.

### Full Table Datasets (run only best model + CNextU-net)

All Cartesian datasets from The Well. FNO/TFNO/U-net numbers are taken directly from the paper. We run CNextU-net (reproduction check) and our best model.

| #   | Dataset                         | Dim | Resolution  | CNextU VRMSE | LR (CNextU) | Our LR | Notes                                            |
| --- | ------------------------------- | --- | ----------- | ------------ | ----------- | ------ | ------------------------------------------------ |
| T1  | `acoustic_scattering_maze`      | 2D  | 256×256     | **0.0153**   | 1e-3        | TODO   | FNO/TFNO competitive here                        |
| T2  | `active_matter`                 | 2D  | 256×256     | **0.1034**   | 5e-3        | TODO   | Ablation dataset                                 |
| T3  | `euler_multi_quadrants`         | 2D  | 512×512     | **0.1531**   | 5e-3        | TODO   | 5.1TB, I/O bound                                 |
| T4  | `gray_scott_reaction_diffusion` | 2D  | 128×128     | 0.1761       | 1e-4        | TODO   | FNO wins here; 1001 steps → long rollout         |
| T5  | `helmholtz_staircase`           | 2D  | 1024×256    | 0.02758      | 5e-4        | TODO   | FNO dominates (0.00046); low priority            |
| T6  | `MHD_64`                        | 3D  | 64³         | **0.1633**   | 5e-3        | TODO   | Only 40 total trajs — noisy                      |
| T7  | `rayleigh_benard`               | 2D  | 512×128     | 0.6699       | 5e-4        | TODO   | All models poor; TFNO wins                       |
| T8  | `rayleigh_taylor_instability`   | 2D  | 128×512     | >10          | 5e-3        | TODO   | All models fail; skip or include as open problem |
| T9  | `shear_flow`                    | 2D  | 128×256     | **0.8080**   | 5e-4        | TODO   | All models poor                                  |
| T10 | `supernova_explosion_64`        | 3D  | 64³         | 0.3181       | 5e-4        | TODO   | Ablation dataset; U-net wins (0.3063)            |
| T11 | `turbulence_gravity_cooling`    | 3D  | 64³         | **0.2096**   | 1e-3        | TODO   | 2700 trajs — ideal for 3D                        |
| T12 | `turbulent_radiative_layer_2D`  | 2D  | 128×384     | **0.1956**   | 5e-3        | TODO   | 6.9GB, fast                                      |
| T13 | `turbulent_radiative_layer_3D`  | 3D  | 128×128×256 | **0.3667**   | 5e-3        | TODO   | Asymmetric 3D resolution                         |
| T14 | `viscoelastic_instability`      | 2D  | 512×512     | **0.2499**   | 5e-4        | TODO   | Variable-length sequences                        |

> **Excluded (non-Cartesian)**: `convective_envelope_rsg` (spherical), `planetswe` (angular), `post_neutron_star_merger` (log-spherical), `MHD_256` (likely OOM at 256³). Explicitly state scope in the paper.

______________________________________________________________________

## Experimental Design

### Phase 0 — Precision Ablation ← **Run first, blocks everything else**

**Question**: Does bf16-mixed degrade accuracy or cause training instability relative to fp32 on high-frequency PDE dynamics? Can we safely use bf16-mixed (faster, less memory) or must we match the paper's fp32?

**Datasets**: two complementary regimes, each stressing a different failure mode of bf16.

| Dataset                         | Resolution | Why it stresses precision                                                                                                                                                                                                                                                                                                                    |
| ------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `helmholtz_staircase`           | 1024×256   | Canonical high-frequency PDE (∇²u + k²u = f). Oscillatory solutions require accurate phase and amplitude representation — exactly where bf16's 7-bit mantissa hurts most. FNO dominates (0.00046) because it is spectrally aligned; our models won't match that, but divergence or large VRMSE increase under bf16 would be a clear signal.  |
| `gray_scott_reaction_diffusion` | 128×128    | Precision-sensitive via a different mechanism: numerical stiffness. Reaction-diffusion systems require precise cancellation between diffusion and reaction terms to maintain sharp Turing pattern interfaces. Small mantissa errors accumulate across the 1001-step trajectory. Fast to train (44–46 epochs in 12h), giving rapid iteration. |

**Models**: UNet-ConvNeXt (baseline reference), ResNet-Hyena, ResNet-Attention. UNet-ConvNeXt anchors against the paper; the ResNet models test precision sensitivity without the confound of UNet architecture changes.

**Design**: 18 runs — 3 datasets × 3 models × 2 precisions. All other hyperparameters identical (same LR, same seed).

#### helmholtz_staircase (paper CNextU-net: 0.02758, best LR: 5e-4)

| ID   | Model            | Precision  | LR   | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | ---------------- | ---------- | ---- | ------ | --------- | ---------- | --- |
| P0-A | UNet-ConvNeXt    | fp32       | 5e-4 | —      | —         | —          | —   |
| P0-B | UNet-ConvNeXt    | bf16-mixed | 5e-4 | —      | —         | —          | —   |
| P0-C | ResNet-Hyena     | fp32       | 5e-4 | —      | —         | —          | —   |
| P0-D | ResNet-Hyena     | bf16-mixed | 5e-4 | —      | —         | —          | —   |
| P0-E | ResNet-Attention | fp32       | 5e-4 | —      | —         | —          | —   |
| P0-F | ResNet-Attention | bf16-mixed | 5e-4 | —      | —         | —          | —   |

#### gray_scott_reaction_diffusion (paper CNextU-net: 0.1761, best LR: 1e-4)

| ID   | Model            | Precision  | LR   | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | ---------------- | ---------- | ---- | ------ | --------- | ---------- | --- |
| P0-G | UNet-ConvNeXt    | fp32       | 1e-4 | —      | —         | —          | —   |
| P0-H | UNet-ConvNeXt    | bf16-mixed | 1e-4 | —      | —         | —          | —   |
| P0-I | ResNet-Hyena     | fp32       | 1e-4 | —      | —         | —          | —   |
| P0-J | ResNet-Hyena     | bf16-mixed | 1e-4 | —      | —         | —          | —   |
| P0-K | ResNet-Attention | fp32       | 1e-4 | —      | —         | —          | —   |
| P0-L | ResNet-Attention | bf16-mixed | 1e-4 | —      | —         | —          | —   |

#### supernova_explosion_64 — 3D (paper CNextU-net: 0.3181, best LR: 5e-4)

| ID   | Model            | Precision  | LR   | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | ---------------- | ---------- | ---- | ------ | --------- | ---------- | --- |
| P0-M | UNet-ConvNeXt    | fp32       | 5e-4 | —      | —         | —          | —   |
| P0-N | UNet-ConvNeXt    | bf16-mixed | 5e-4 | —      | —         | —          | —   |
| P0-O | ResNet-Hyena     | fp32       | 5e-4 | —      | —         | —          | —   |
| P0-P | ResNet-Hyena     | bf16-mixed | 5e-4 | —      | —         | —          | —   |
| P0-Q | ResNet-Attention | fp32       | 5e-4 | —      | —         | —          | —   |
| P0-R | ResNet-Attention | bf16-mixed | 5e-4 | —      | —         | —          | —   |

> We train beyond the paper's 12h budget, so we expect all models to reach lower VRMSE than the published baselines. The relevant comparison is fp32 vs bf16 **within** each model and dataset, not against the paper's absolute numbers.

**Decision rule**:

- If |VRMSE(fp32) − VRMSE(bf16)| \< 2% relative across all models on both datasets **and** all train stably → use **bf16-mixed** for all subsequent phases.
- If bf16 degrades on helmholtz but not gray_scott (or vice versa) → use **fp32** to be safe; note the result as a finding in the paper.
- If bf16 is >2% worse or diverges on any model → use **fp32** for all subsequent phases and document the deviation from The Well paper.

> ⚠ Do not start Phase 1 until P0 is resolved. Using the wrong precision for 100+ runs would require repeating everything.

### Phase 1 — Learning Rate Ablation

**Question**: What are the optimal learning rates for ResNet-Hyena and ResNet-Attention? The Phase 0 LRs are inherited from CNextU-net — they may not transfer to a different architecture.

**Datasets**: `helmholtz_staircase` and `gray_scott_reaction_diffusion` (same as Phase 0, so the Phase 0 fp32 runs at the paper-default LR serve as reference points at no extra cost).
**Seeds**: 1 seed (this is a hyperparameter search, not a final result).

#### helmholtz_staircase — LR sweep (paper CNextU-net best: 5e-4)

| ID   | Model            | LR   | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | ---------------- | ---- | ------ | --------- | ---------- | --- |
| P1-A | ResNet-Hyena     | 1e-4 | —      | —         | —          | —   |
| P1-B | ResNet-Hyena     | 5e-4 | —      | —         | —          | —   |
| P1-C | ResNet-Hyena     | 1e-3 | —      | —         | —          | —   |
| P1-D | ResNet-Hyena     | 5e-3 | —      | —         | —          | —   |
| P1-E | ResNet-Attention | 1e-4 | —      | —         | —          | —   |
| P1-F | ResNet-Attention | 5e-4 | —      | —         | —          | —   |
| P1-G | ResNet-Attention | 1e-3 | —      | —         | —          | —   |
| P1-H | ResNet-Attention | 5e-3 | —      | —         | —          | —   |

#### gray_scott_reaction_diffusion — LR sweep (paper CNextU-net best: 1e-4)

| ID   | Model            | LR   | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | ---------------- | ---- | ------ | --------- | ---------- | --- |
| P1-I | ResNet-Hyena     | 1e-4 | —      | —         | —          | —   |
| P1-J | ResNet-Hyena     | 5e-4 | —      | —         | —          | —   |
| P1-K | ResNet-Hyena     | 1e-3 | —      | —         | —          | —   |
| P1-L | ResNet-Hyena     | 5e-3 | —      | —         | —          | —   |
| P1-M | ResNet-Attention | 1e-4 | —      | —         | —          | —   |
| P1-N | ResNet-Attention | 5e-4 | —      | —         | —          | —   |
| P1-O | ResNet-Attention | 1e-3 | —      | —         | —          | —   |
| P1-P | ResNet-Attention | 5e-3 | —      | —         | —          | —   |

#### supernova_explosion_64 — 3D LR sweep (paper CNextU-net best: 5e-4)

| ID   | Model            | LR   | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | ---------------- | ---- | ------ | --------- | ---------- | --- |
| P1-Q | ResNet-Hyena     | 1e-4 | —      | —         | —          | —   |
| P1-R | ResNet-Hyena     | 5e-4 | —      | —         | —          | —   |
| P1-S | ResNet-Hyena     | 1e-3 | —      | —         | —          | —   |
| P1-T | ResNet-Hyena     | 5e-3 | —      | —         | —          | —   |
| P1-U | ResNet-Attention | 1e-4 | —      | —         | —          | —   |
| P1-V | ResNet-Attention | 5e-4 | —      | —         | —          | —   |
| P1-W | ResNet-Attention | 1e-3 | —      | —         | —          | —   |
| P1-X | ResNet-Attention | 5e-3 | —      | —         | —          | —   |

> Note: UNet-ConvNeXt does not need a sweep here — its optimal LR is already established from the paper. The Phase 0 fp32 ConvNeXt runs (P0-A, P0-G, P0-M) double as the fixed reference.

**Output**: Best LR per model per dataset. These are carried forward into Phase 2 and all subsequent phases.

### Phase 2 — Fair UNet Comparison

**Question**: Does Hyena add value over ConvNeXt and Attention when they all operate at the same full resolution inside a UNet?

**Datasets**: `helmholtz_staircase`, `gray_scott_reaction_diffusion`, `supernova_explosion_64` — same three as Phase 0/1, enabling direct comparison between ResNet and UNet mixers on identical datasets.
**Seeds**: 1 seed (ablation experiment; 3 seeds reserved for the final table)
**LR**: Best LR from Phase 1 for Hyena and Attention; paper best LR for ConvNeXt.

> P2-A reproduces the CNextU-net baseline using our codebase. If P2-A does not match BASELINES.md within ~5% VRMSE, we have an implementation or LR issue that must be fixed before proceeding.

**Expected result**: P2-C ≥ P2-A > P2-B (Hyena best, ConvNeXt second, Attention worst at full resolution due to quadratic cost vs image-scale tokens). If P2-A > P2-C, the UNet structure helps but not the mixer, and we need to reconsider the narrative.

#### helmholtz_staircase (paper: FNO **0.00046**, CNextU-net 0.02758, LR: 5e-4)

| ID   | Model          | LR           | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | -------------- | ------------ | ------ | --------- | ---------- | --- |
| P2-A | UNet-ConvNeXt  | 5e-4         | —      | —         | —          | —   |
| P2-B | UNet-Attention | Phase 1 best | —      | —         | —          | —   |
| P2-C | UNet-Hyena     | Phase 1 best | —      | —         | —          | —   |

#### gray_scott_reaction_diffusion (paper: FNO **0.1365**, CNextU-net 0.1761, LR: 1e-4)

| ID   | Model          | LR           | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | -------------- | ------------ | ------ | --------- | ---------- | --- |
| P2-D | UNet-ConvNeXt  | 1e-4         | —      | —         | —          | —   |
| P2-E | UNet-Attention | Phase 1 best | —      | —         | —          | —   |
| P2-F | UNet-Hyena     | Phase 1 best | —      | —         | —          | —   |

#### supernova_explosion_64 — 3D (paper: U-net **0.3063**, CNextU-net 0.3181, LR: 5e-4)

| ID   | Model          | LR           | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---- | -------------- | ------------ | ------ | --------- | ---------- | --- |
| P2-G | UNet-ConvNeXt  | 5e-4         | —      | —         | —          | —   |
| P2-H | UNet-Attention | Phase 1 best | —      | —         | —          | —   |
| P2-I | UNet-Hyena     | Phase 1 best | —      | —         | —          | —   |

### Phase 3 — Resolution & Scaling

**Questions**:

1. Does the UNet advantage over patch-based models shrink as patch size decreases (i.e. is it purely resolution)?
1. Does Hyena's advantage over Attention grow with sequence length (subquadratic scaling)?

**Datasets**: `active_matter` (2D, 256×256 — primary), `supernova_explosion_64` (3D, 64³ — secondary)
**Seeds**: 1 seed (scaling study; primary interest is the trend, not absolute numbers)
**LR**: Best from Phase 1

UNet models serve as the full-resolution reference (no patch size). ResNet models sweep patch sizes 16→1, doubling sequence length at each step.

#### active_matter (256×256)

| ID   | Model            | Patch    | Tokens | Epochs | val/VRMSE | test/VRMSE | Peak Mem | Throughput | W&B |
| ---- | ---------------- | -------- | ------ | ------ | --------- | ---------- | -------- | ---------- | --- |
| P3-A | UNet-ConvNeXt    | full res | 65,536 | —      | —         | —          | —        | —          | —   |
| P3-B | UNet-Hyena       | full res | 65,536 | —      | —         | —          | —        | —          | —   |
| P3-C | UNet-Attention   | full res | 65,536 | —      | —         | —          | —        | —          | —   |
| P3-D | ResNet-Hyena     | 16       | 256    | —      | —         | —          | —        | —          | —   |
| P3-E | ResNet-Attention | 16       | 256    | —      | —         | —          | —        | —          | —   |
| P3-F | ResNet-Hyena     | 8        | 1,024  | —      | —         | —          | —        | —          | —   |
| P3-G | ResNet-Attention | 8        | 1,024  | —      | —         | —          | —        | —          | —   |
| P3-H | ResNet-Hyena     | 4        | 4,096  | —      | —         | —          | —        | —          | —   |
| P3-I | ResNet-Attention | 4        | 4,096  | —      | —         | —          | —        | —          | —   |
| P3-J | ResNet-Hyena     | 2        | 16,384 | —      | —         | —          | —        | —          | —   |
| P3-K | ResNet-Attention | 2        | 16,384 | —      | —         | —          | —        | —          | —   |
| P3-L | ResNet-Hyena     | 1        | 65,536 | —      | —         | —          | —        | —          | —   |
| P3-M | ResNet-Attention | 1        | 65,536 | —      | —         | —          | —        | —          | —   |

#### supernova_explosion_64 (64³)

| ID   | Model            | Patch    | Tokens  | Epochs | val/VRMSE | test/VRMSE | Peak Mem | Throughput | W&B |
| ---- | ---------------- | -------- | ------- | ------ | --------- | ---------- | -------- | ---------- | --- |
| P3-N | UNet-ConvNeXt    | full res | 262,144 | —      | —         | —          | —        | —          | —   |
| P3-O | UNet-Hyena       | full res | 262,144 | —      | —         | —          | —        | —          | —   |
| P3-P | UNet-Attention   | full res | 262,144 | —      | —         | —          | —        | —          | —   |
| P3-Q | ResNet-Hyena     | 16       | 64      | —      | —         | —          | —        | —          | —   |
| P3-R | ResNet-Attention | 16       | 64      | —      | —         | —          | —        | —          | —   |
| P3-S | ResNet-Hyena     | 8        | 512     | —      | —         | —          | —        | —          | —   |
| P3-T | ResNet-Attention | 8        | 512     | —      | —         | —          | —        | —          | —   |
| P3-U | ResNet-Hyena     | 4        | 4,096   | —      | —         | —          | —        | —          | —   |
| P3-V | ResNet-Attention | 4        | 4,096   | —      | —         | —          | —        | —          | —   |
| P3-W | ResNet-Hyena     | 2        | 32,768  | —      | —         | —          | —        | —          | —   |
| P3-X | ResNet-Attention | 2        | 32,768  | —      | —         | —          | —        | —          | —   |
| P3-Y | ResNet-Hyena     | 1        | 262,144 | —      | —         | —          | —        | —          | —   |
| P3-Z | ResNet-Attention | 1        | 262,144 | —      | —         | —          | —        | —          | —   |

### Phase 4 — Full Table (All Datasets)

**Goal**: Establish broad coverage for the final SoTA table.

**Models**: Our best model from Phase 2–3 + CNextU-net (reproduced). FNO/TFNO/U-net numbers taken from the paper.
**Seeds**: 3 seeds (final results for the paper)
**LR**: Dataset-specific, from BASELINES.md Table 6 as starting point for CNextU-net; best LR from Phase 1 sweep applied to Hyena/Attention models.

Run configs for all 14 datasets in the full table (T1–T14 above).

Best model = TBD after Phases 1–3. Paper baselines reproduced from BASELINES.md.
Model codes: **C** = UNet-ConvNeXt (us), **H** = Best Hyena (TBD), **A** = Best Attention (TBD).

#### Summary Table

| Dataset                       | FNO         | TFNO       | U-net      | CNextU-net (paper) | CNextU-net (us) | Best Hyena | Best Attention |
| ----------------------------- | ----------- | ---------- | ---------- | ------------------ | --------------- | ---------- | -------------- |
| acoustic_scattering_maze      | 0.5062      | 0.5057     | 0.0351     | **0.0153**         | —               | —          | —              |
| active_matter                 | 0.3691      | 0.3598     | 0.2489     | **0.1034**         | —               | —          | —              |
| euler_multi_quadrants         | 0.4081      | 0.4163     | 0.1834     | **0.1531**         | —               | —          | —              |
| gray_scott_reaction_diffusion | **0.1365**  | 0.3633     | 0.2252     | 0.1761             | —               | —          | —              |
| helmholtz_staircase           | **0.00046** | 0.00346    | 0.01931    | 0.02758            | —               | —          | —              |
| MHD_64                        | 0.3605      | 0.3561     | 0.1798     | **0.1633**         | —               | —          | —              |
| rayleigh_benard               | 0.8395      | **0.6566** | 1.4860     | 0.6699             | —               | —          | —              |
| rayleigh_taylor_instability   | >10         | >10        | >10        | >10                | —               | —          | —              |
| shear_flow                    | 1.189       | 1.472      | 3.447      | **0.8080**         | —               | —          | —              |
| supernova_explosion_64        | 0.3783      | 0.3785     | **0.3063** | 0.3181             | —               | —          | —              |
| turbulence_gravity_cooling    | 0.2429      | 0.2673     | 0.6753     | **0.2096**         | —               | —          | —              |
| turbulent_radiative_layer_2D  | 0.5001      | 0.5016     | 0.2418     | **0.1956**         | —               | —          | —              |
| turbulent_radiative_layer_3D  | 0.5278      | 0.5187     | 0.3728     | **0.3667**         | —               | —          | —              |
| viscoelastic_instability      | 0.7212      | 0.7102     | 0.4185     | **0.2499**         | —               | —          | —              |

> Bold = best published. Values in *italic* (once filled) = new SoTA. CNextU-net (us) = our reproduction with extended training budget; expected to beat the paper numbers.

#### Run Table

##### T1 — `acoustic_scattering_maze`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T1-C-s1 | UNet-ConvNeXt        | 1e-3         | full res       | —      | —         | —          | —   |
| P4-T1-C-s2 | UNet-ConvNeXt        | 1e-3         | full res       | —      | —         | —          | —   |
| P4-T1-C-s3 | UNet-ConvNeXt        | 1e-3         | full res       | —      | —         | —          | —   |
| P4-T1-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T1-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T1-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T1-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T1-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T1-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T2 — `active_matter`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T2-C-s1 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T2-C-s2 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T2-C-s3 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T2-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T2-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T2-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T2-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T2-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T2-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T3 — `euler_multi_quadrants`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T3-C-s1 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T3-C-s2 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T3-C-s3 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T3-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T3-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T3-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T3-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T3-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T3-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T4 — `gray_scott_reaction_diffusion`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T4-C-s1 | UNet-ConvNeXt        | 1e-4         | full res       | —      | —         | —          | —   |
| P4-T4-C-s2 | UNet-ConvNeXt        | 1e-4         | full res       | —      | —         | —          | —   |
| P4-T4-C-s3 | UNet-ConvNeXt        | 1e-4         | full res       | —      | —         | —          | —   |
| P4-T4-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T4-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T4-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T4-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T4-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T4-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T5 — `helmholtz_staircase`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T5-C-s1 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T5-C-s2 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T5-C-s3 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T5-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T5-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T5-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T5-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T5-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T5-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T6 — `MHD_64`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T6-C-s1 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T6-C-s2 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T6-C-s3 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T6-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T6-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T6-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T6-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T6-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T6-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T7 — `rayleigh_benard`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T7-C-s1 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T7-C-s2 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T7-C-s3 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T7-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T7-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T7-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T7-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T7-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T7-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T8 — `rayleigh_taylor_instability`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T8-C-s1 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T8-C-s2 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T8-C-s3 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T8-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T8-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T8-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T8-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T8-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T8-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T9 — `shear_flow`

| ID         | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ---------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T9-C-s1 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T9-C-s2 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T9-C-s3 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T9-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T9-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T9-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T9-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T9-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T9-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T10 — `supernova_explosion_64`

| ID          | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ----------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T10-C-s1 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T10-C-s2 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T10-C-s3 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T10-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T10-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T10-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T10-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T10-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T10-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T11 — `turbulence_gravity_cooling`

| ID          | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ----------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T11-C-s1 | UNet-ConvNeXt        | 1e-3         | full res       | —      | —         | —          | —   |
| P4-T11-C-s2 | UNet-ConvNeXt        | 1e-3         | full res       | —      | —         | —          | —   |
| P4-T11-C-s3 | UNet-ConvNeXt        | 1e-3         | full res       | —      | —         | —          | —   |
| P4-T11-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T11-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T11-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T11-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T11-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T11-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T12 — `turbulent_radiative_layer_2D`

| ID          | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ----------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T12-C-s1 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T12-C-s2 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T12-C-s3 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T12-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T12-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T12-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T12-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T12-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T12-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T13 — `turbulent_radiative_layer_3D`

| ID          | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ----------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T13-C-s1 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T13-C-s2 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T13-C-s3 | UNet-ConvNeXt        | 5e-3         | full res       | —      | —         | —          | —   |
| P4-T13-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T13-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T13-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T13-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T13-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T13-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

##### T14 — `viscoelastic_instability`

| ID          | Model                | LR           | Patch Size     | Epochs | val/VRMSE | test/VRMSE | W&B |
| ----------- | -------------------- | ------------ | -------------- | ------ | --------- | ---------- | --- |
| P4-T14-C-s1 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T14-C-s2 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T14-C-s3 | UNet-ConvNeXt        | 5e-4         | full res       | —      | —         | —          | —   |
| P4-T14-H-s1 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T14-H-s2 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T14-H-s3 | Best Hyena (TBD)     | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T14-A-s1 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T14-A-s2 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |
| P4-T14-A-s3 | Best Attention (TBD) | Phase 1 best | Phase 2/3 best | —      | —         | —          | —   |

______________________________________________________________________

## Architecture Summary (v2 Models)

All patch-based models (ResNet, ViT5) share: depth=12, cosine schedule, 10% warmup, AdamW, weight_decay=1e-5, grad_clip=1.0, bf16-mixed.

| Model            | Backbone              | Block          | Hidden Dim       | Patch Size | Resolution Access |
| ---------------- | --------------------- | -------------- | ---------------- | ---------- | ----------------- |
| UNet-ConvNeXt    | WellUNet              | ConvNeXtBlock  | init_features=42 | None       | Full pixel        |
| UNet-Attention   | WellUNet              | AttentionBlock | init_features=42 | None       | Full pixel        |
| UNet-Hyena       | WellUNet              | HyenaBlock     | init_features=42 | None       | Full pixel        |
| ResNet-Hyena     | ResidualNetwork       | Hyena          | 512              | 4 or 8     | Patch-level       |
| ResNet-Attention | ResidualNetwork       | Attention      | 512              | 4 or 8     | Patch-level       |
| ViT5-Hyena       | ViT5GeneralPurposeNet | Hyena          | 384              | 4 or 8     | Patch-level       |
| ViT5-Attention   | ViT5GeneralPurposeNet | Attention      | 384              | 4 or 8     | Patch-level       |

> **TODO**: Add `UNet-Attention` config (does not exist yet). This is required for Phase 1.

______________________________________________________________________

### Nice to have

- [ ] **Model Scaling Experiment**: We didn't experiment yet with different model sizes (e.g., depth, hidden dim) for the best-performing Hyena/Attention models. This would be a nice addition to the paper if we have time.
- [ ] **FLOPs measurement**: for all model variants (use `torchinfo` or `fvcore`).
- [ ] **Rollout VRMSE**: report 6:12 and 13:30 windows for Phase 4 (matches paper Table 3).
- [ ] **Memory and throughput** for Phase 3 scaling plots.
- [ ] **ViT5-Hyena FiLM conditioning**: test on active_matter once Phase 1 baseline is established.

______________________________________________________________________

## Observations

*(none yet — fill in as results arrive)*

______________________________________________________________________

## How to Submit

```bash
# 1-GPU run (geodude partition)
sbatch --job-name=<name> examples/well/submit_well_ivi_1gpu_geodude.sh examples/well/v2/<dataset>/cfg_<model>.py

# 2-GPU run (geodude)
sbatch --job-name=<name> examples/well/submit_well_ivi_2gpu_geodude.sh examples/well/v2/<dataset>/cfg_<model>.py
```

## How to Download Datasets

```bash
bash scripts/download_well.sh <dataset_name>
# or on SLURM:
sbatch slurm/download_well.sh <dataset_name>
```
