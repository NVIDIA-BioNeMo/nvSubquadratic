# Pixel-Hyena vs ViT — TinyImageNet Ablation & Benchmark Tracker

Systematic ablation study of the Pixel-Hyena operator on TinyImageNet (200 classes, 64×64), followed by full-scale comparison against standard ViT on ImageNet.

## Task Description

- **Dataset (ablations)**: TinyImageNet (200 classes, 64×64 RGB)
- **Dataset (final)**: ImageNet-1K (1000 classes, 224×224 RGB)
- **Task**: Multi-class classification
- **Objective**: (1) Ablate Hyena operator components on TinyImageNet, (2) compare Pixel-Hyena vs ViT at S/B/L scale on ImageNet

## Compute Resources

| Partition  | GPUs                                           | Max GPUs/user | Max Time | Account        | User                 | Notes                    |
| :--------- | :--------------------------------------------- | :------------ | :------- | :------------- | :------------------- | :----------------------- |
| `geodude`  | 4 × RTX A5000 (24 GB)                          | 4             | 7 days   | `geodudeusers` | `dwessel`            |                          |
| `all6000`  | 8 × RTX 6000 (24 GB)                           | 2             | 7 days   | `all6000users` | `dwessel`, `dknigge` |                          |
| `cees`     | 8 × RTX A5000 (24 GB) per node (7 nodes total) | 8             | 7 days   | `ceesusers`    | `dknigge`            | **dknigge account only** |
| `cees6000` | 8 × RTX 6000 (24 GB) per node (2 nodes total)  | 8             | 4 days   | `ceesusers`    | `dknigge`            | **dknigge account only** |

> \[!IMPORTANT\]
> **Fixed batch-size rule**: All experiments use the **same effective batch size** (e.g. 128). When running on fewer GPUs, use gradient accumulation to match. Example: 1 GPU × bs 32 × accum 4 = 128, 4 GPUs × bs 32 × accum 1 = 128.

## Model Architecture (ViT-B Scale — TinyImageNet Ablations)

All ablation runs share ViT-B architecture:

- **Hidden dimension**: 768
- **Blocks**: 12
- **MLP**: GELU (expansion 4.0) for both Hyena and Attention
- **Precision**: bf16-mixed
- **Iterations**: 300,000 (shorter than 600k; sufficient for 64×64 ablations — reassess after first runs)
- **Scheduler**: Cosine with 5% warmup
- **Optimizer**: AdamW, grad_clip=1.0

### Parameter Comparison (ViT-B scale, patch-4)

| Component                     | ViT-B (Attention) | Pixel-Hyena | Notes                                                           |
| :---------------------------- | :---------------- | :---------- | :-------------------------------------------------------------- |
| Input projection (Patchify)   | 37K               | 37K         | Conv2d(3, 768, k=4, s=4)                                        |
| QKV projection (per block)    | 1.77M             | 1.77M       | Linear(768 → 2304)                                              |
| Mixer (per block)             | 0 (softmax)       | ~80K        | SIREN kernel (~58K) + short conv (~21K) + Gaussian mask (~1.5K) |
| Output projection (per block) | 590K              | 590K        | Linear(768 → 768)                                               |
| MLP (per block)               | 4.72M             | 4.72M       | GELU, 768 → 3072 → 768                                          |
| LayerNorms (per block)        | 3K                | 4.5K        | Hyena has extra pixelhyena_norm                                 |
| Classification head           | 154K              | 154K        | Linear(768 → 200)                                               |
| **Total (12 blocks)**         | **~85.2M**        | **~86.2M**  | Hyena adds ~1M from SIREN kernels                               |

> \[!NOTE\]
> Parameter counts are approximate — verify with `model.parameters()` after first run. The ~1% overhead from SIREN kernels makes this a fair comparison.

| Config Variant  | Mixer     | Patchify | Patch Size | Seq Length | Mask     | Grid   | Padding | RoPE |
| :-------------- | :-------- | :------- | :--------- | :--------- | :------- | :----- | :------ | :--- |
| Hyena (pixel)   | Hyena     | No       | —          | 4,096      | Gaussian | double | zero    | No   |
| Hyena (patch-4) | Hyena     | Yes      | 4          | 256        | Gaussian | double | zero    | No   |
| ViT (pixel)     | Attention | No       | —          | 4,096      | —        | —      | —       | Yes  |
| ViT (patch-4)   | Attention | Yes      | 4          | 256        | —        | —      | —       | Yes  |

### Data Augmentation (all configs)

- Mixup (α=0.8) + Cutmix (α=1.0), switch_prob=0.5
- RandAugment `rand-m9-n3-mstd0.5`
- Random crop (64×64, pad=4) + Horizontal flip

### Default Hyena Kernel Config

| Parameter        | Value                     |
| :--------------- | :------------------------ |
| Kernel           | `SIRENKernelND`           |
| `omega_0`        | 30.0                      |
| `hidden_omega_0` | 1.0                       |
| `mlp_hidden_dim` | 64                        |
| `num_layers`     | 3                         |
| `embedding_dim`  | 64                        |
| `L_cache`        | 64 (pixel) / 16 (patch=4) |

______________________________________________________________________

## 🔬 Experimental Phases

### Phase 0: Pipeline Validation ⭐ HIGHEST PRIORITY

**Goal**: Confirm the TinyImageNet training pipeline works end-to-end with a standard ViT-B + patchify baseline. This is our sanity check and reference point.

| #   | Experiment                               | Config                             | Partition    | GPUs | BS/GPU | Accum | Eff. BS | Status          | Val Acc | Job ID   | WandB                                                                         | Notes                                                                |
| :-- | :--------------------------------------- | :--------------------------------- | :----------- | :--- | :----- | :---- | :------ | :-------------- | :------ | :------- | :---------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| 0.1 | **ViT-B + patch-4 baseline**             | `attention_patchify.py`            | geodude      | 4    | 32     | 1     | 128     | ✅ Done         | 54.3%   | `137108` | —                                                                             | Reached 300k steps; stable convergence.                              |
| 0.2 | Hyena + patch-4 baseline                 | `hyena_patchify.py`                | hipster/perf | 4    | 32     | 1     | 128     | ✅ Completed    | 70.67%  | `174875` | [9iqbx19w](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9iqbx19w) | Sanity check Hyena pipeline (hipster)                                |
| 0.3 | **ViT-B/16 attention on ImageNet-1K** ⭐ | `attention_patchify_imagenet1k.py` | cees         | 8    | 128    | 1     | 1024    | ⏳ Awaiting WDS | —       | `140516` | —                                                                             | Converting to WebDataset; will resubmit once done. Target ≥70% top-1 |

**Success criteria**:

- **0.1/0.2** (TinyImageNet): ViT-B patch-4 converges to ≥ 55% val acc within ~100k iterations (DeiT-B on TinyImageNet literature range: 55–65%).
- **0.3** (ImageNet-1K): ViT-B/16 reaches ≥ 70% top-1 val acc within 300k iterations (~240 epochs at BS=1024).

______________________________________________________________________

### Phase 1: Kernel Ablation (Hyena + patch-4)

**Goal**: Validate SIREN superiority over RFF. Run on **Hyena + patch-4** (fast, 256 tokens).

| #   | Experiment     | Variable    | Config Change            | Partition | GPUs | Status       | Val Acc | Job ID   | WandB                                                                         | Notes                 |
| :-- | :------------- | :---------- | :----------------------- | :-------- | :--- | :----------- | :------ | :------- | :---------------------------------------------------------------------------- | :-------------------- |
| 1.1 | SIREN baseline | —           | base `hyena_patchify.py` | geodude   | 4    | 🔄 Finishing | 70.3%   | `140280` | [06hpkzo4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/06hpkzo4) | Step ~289k/300k.      |
| 1.2 | RFF kernel     | kernel_type | `RandomFourierKernelND`  | geodude   | 4    | ⏳ Pending   | —       | `140281` | —                                                                             | Expect ↓ acc vs SIREN |

**Hypothesis**: RFF lacks the expressiveness of SIREN's sine-based representation for vision, resulting in lower accuracy.

______________________________________________________________________

### Phase 2: SIREN ω₀ Sweep (Hyena + patch-4)

**Goal**: Find the optimal ω₀ that controls frequency expressiveness of the SIREN kernel!

| #   | Experiment            | ω₀  | Partition    | GPUs | Status       | Val Acc | Job ID   | WandB                                                                         | Notes                        |
| :-- | :-------------------- | :-- | :----------- | :--- | :----------- | :------ | :------- | :---------------------------------------------------------------------------- | :--------------------------- |
| 2.1 | ω₀ = 10               | 10  | hipster/cap  | 2    | 🔄 Running   | 69.9%   | `174895` | [yxxcr5wh](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yxxcr5wh) | Ep 242, ~189k/300k, ETA ~37h |
| 2.2 | ω₀ = 20               | 20  | hipster/cap  | 2    | 🔄 Running   | 70.5%   | `174896` | [c4x52706](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/c4x52706) | Ep 228, ~178k/300k, ETA ~41h |
| 2.3 | **ω₀ = 30 (default)** | 30  | hipster/perf | 4    | ✅ Completed | 70.67%  | `174875` | [9iqbx19w](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9iqbx19w) | = Phase 0.2 (same run)       |
| 2.4 | ω₀ = 60               | 60  | hipster/cap  | 2    | 🔄 Running   | 69.2%   | `174897` | [jc9bv226](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jc9bv226) | Ep 216, ~169k/300k, ETA ~44h |
| 2.5 | ω₀ = 100              | 100 | hipster/cap  | 2    | 🔄 Running   | 70.1%   | `174898` | [n86qahfw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n86qahfw) | Ep 183, ~143k/300k, ETA ~52h |

> \[!TIP\]
> **Efficiency**: Run 2.1–2.5 in parallel on 1 GPU each (4 on geodude + 1 on all6000), or use 1 GPU on geodude sequentially, gradient-accumulating to match effective batch size. With 4 GPUs on geodude we can run 4 of these in parallel.

**Hypothesis**: Optimal ω₀ is in \[20, 60\] range; too-high values cause oscillations, too-low lose detail.

______________________________________________________________________

### Phase 3: Kernel Hidden-Dim Sweep (Hyena + patch-4)

**Goal**: Find optimal SIREN MLP hidden dimension (expressiveness vs parameters tradeoff).

| #   | Experiment              | Hidden Dim | Partition | GPUs | Status     | Val Acc | Job ID | WandB | Notes            |
| :-- | :---------------------- | :--------- | :-------- | :--- | :--------- | :------ | :----- | :---- | :--------------- |
| 3.1 | hdim = 32               | 32         | geodude   | 1    | 📅 Planned | —       | —      | —     | Lean kernel      |
| 3.2 | **hdim = 64 (default)** | 64         | geodude   | 1    | 📅 Planned | —       | —      | —     | Reference        |
| 3.3 | hdim = 128              | 128        | geodude   | 1    | 📅 Planned | —       | —      | —     |                  |
| 3.4 | hdim = 256              | 256        | geodude   | 1    | 📅 Planned | —       | —      | —     | Expensive kernel |

> \[!NOTE\]
> Should be run with optimal ω₀ from Phase 2.

**Hypothesis**: Diminishing returns beyond 64–128; the kernel is a small MLP so going wider should help mildly but it adds parameters.

______________________________________________________________________

### Phase 4: Mask Ablation (Hyena + pixel — no patchify)

**Goal**: Evaluate the impact of the modulation mask on classification. Tested on full 4096-token pixel input where ringing artifacts are most pronounced.

| #   | Experiment                  | Mask                      | Partition | GPUs | Status     | Val Acc | Job ID | WandB | Notes                     |
| :-- | :-------------------------- | :------------------------ | :-------- | :--- | :--------- | :------ | :----- | :---- | :------------------------ |
| 4.1 | **Gaussian mask (default)** | `GaussianModulationND`    | all6000   | 4    | 📅 Planned | —       | —      | —     | Current default for pixel |
| 4.2 | No mask                     | `Identity`                | all6000   | 4    | 📅 Planned | —       | —      | —     | Does mask matter?         |
| 4.3 | Exponential mask            | `ExponentialModulationND` | all6000   | 4    | 📅 Planned | —       | —      | —     | Alternative decay         |

> \[!NOTE\]
> Full-res pixel Hyena (4096 tokens) is expensive. Use all6000 (8×RTX 6000) to speed up, or geodude with longer walltime. Running fewer parallel experiments here.

**Hypothesis**: Gaussian mask critical at pixel resolution (suppresses SIREN ringing at filter boundaries), less important with patchify.

______________________________________________________________________

### Phase 5: Positional Encoding / Grid / Padding Ablation (Hyena + patch-4)

**Goal**: Determine whether RoPE + circular convolution can match the double-grid + zero-padding baseline.

| #   | Experiment                            | RoPE | Grid   | Padding  | Partition | GPUs | Status     | Val Acc | Job ID | WandB | Notes                                          |
| :-- | :------------------------------------ | :--- | :----- | :------- | :-------- | :--- | :--------- | :------ | :----- | :---- | :--------------------------------------------- |
| 5.1 | **Default (no-rope + double + zero)** | No   | double | zero     | geodude   | 1    | 📅 Planned | —       | —      | —     | Baseline                                       |
| 5.2 | rope + double + zero                  | Yes  | double | zero     | geodude   | 1    | 📅 Planned | —       | —      | —     | Does RoPE help even with double grid?          |
| 5.3 | rope + circular + single              | Yes  | single | circular | geodude   | 1    | 📅 Planned | —       | —      | —     | Can RoPE compensate for aliased circular conv? |
| 5.4 | no-rope + single + zero               | No   | single | zero     | geodude   | 1    | 📅 Planned | —       | —      | —     | Cheap single grid but no pos info              |

**Hypothesis**: Double-grid + zero-pad is the most principled (no aliasing). RoPE may help with circular+single by providing explicit positional information, but is unlikely to fully close the gap.

______________________________________________________________________

### Phase 6: Learning-Rate & Weight-Decay Sweep (Final Hyena Config)

**Goal**: Fine-tune optimisation HPs *after all architectural ablations are settled*.

> \[!CAUTION\]
> Only begin Phase 6 after Phases 1–5 are complete and the "Gold" Hyena config is locked.

| #   | Experiment       | LR   | WD   | Partition | GPUs | Status     | Val Acc | Job ID | WandB | Notes                      |
| :-- | :--------------- | :--- | :--- | :-------- | :--- | :--------- | :------ | :----- | :---- | :------------------------- |
| 6.1 | lr=1e-3, wd=0.0  | 1e-3 | 0.0  | geodude   | 1    | 📅 Planned | —       | —      | —     | Conservative               |
| 6.2 | lr=3e-3, wd=0.0  | 3e-3 | 0.0  | geodude   | 1    | 📅 Planned | —       | —      | —     |                            |
| 6.3 | lr=8e-3, wd=0.0  | 8e-3 | 0.0  | geodude   | 1    | 📅 Planned | —       | —      | —     | Previous hyena_patchify LR |
| 6.4 | lr=1e-2, wd=0.0  | 1e-2 | 0.0  | geodude   | 1    | 📅 Planned | —       | —      | —     | Aggressive                 |
| 6.5 | lr=best, wd=0.01 | best | 0.01 | geodude   | 1    | 📅 Planned | —       | —      | —     | Some regularization        |
| 6.6 | lr=best, wd=0.05 | best | 0.05 | geodude   | 1    | 📅 Planned | —       | —      | —     | ViT-standard WD            |

______________________________________________________________________

### Phase 7: Patchification Showdown (Best Configs)

**Goal**: After ablations, compare the best Hyena config against best ViT config, both with and without patchification, on TinyImageNet.

| #   | Experiment             | Mixer     | Patchify  | Partition | GPUs | Status     | Val Acc | Job ID | WandB | Notes                       |
| :-- | :--------------------- | :-------- | :-------- | :-------- | :--- | :--------- | :------ | :----- | :---- | :-------------------------- |
| 7.1 | Hyena (pixel) — Gold   | Hyena     | No        | all6000   | 8    | 📅 Planned | —       | —      | —     | Best config from Phases 1–6 |
| 7.2 | Hyena (patch-4) — Gold | Hyena     | Yes (p=4) | geodude   | 4    | 📅 Planned | —       | —      | —     | Best config from Phases 1–6 |
| 7.3 | ViT (pixel)            | Attention | No        | all6000   | 8    | 📅 Planned | —       | —      | —     | 4096 tokens, O(n²)          |
| 7.4 | ViT (patch-4)          | Attention | Yes (p=4) | geodude   | 4    | 📅 Planned | —       | —      | —     | 256 tokens                  |

**Key question**: How much does patchification hurt each architecture? Hyena should degrade less since it can handle long sequences natively.

______________________________________________________________________

### Phase 8: ImageNet at Scale (S/B/L) — FUTURE

**Goal**: Full ViT vs Pixel-Hyena comparison at standard ViT scales on ImageNet-1K.

| Size  | Hidden | Blocks | Heads | Params (~Attn) | Status     |
| :---- | :----- | :----- | :---- | :------------- | :--------- |
| ViT-S | 384    | 12     | 6     | ~22M           | 📅 Planned |
| ViT-B | 768    | 12     | 12    | ~86M           | 📅 Planned |
| ViT-L | 1024   | 24     | 16    | ~307M          | 📅 Planned |

> \[!IMPORTANT\]
> Phase 8 requires significant compute and should only be started after the TinyImageNet ablation story (Phases 1–7) is solid.

______________________________________________________________________

## 📋 Scheduling Strategy

Given limited compute (4× A5000 on geodude, 8× RTX 6000 on all6000), we optimise for parallelism:

```
WEEK 1 (immediate):
├─ Phase 0.1: ViT-B baseline TinyImageNet (4 GPU geodude) ← FIRST PRIORITY
├─ Phase 0.2: Hyena baseline TinyImageNet (4 GPU geodude, after 0.1 finishes)
├─ Phase 0.3: ViT-B/16 ImageNet-1K (8 GPU cees) ← CAN RUN IN PARALLEL, independent dataset
└─ Phase 1.2: RFF ablation (can run on all6000 in parallel)

WEEK 1–2 (after Phase 0 validated):
├─ Phase 2: ω₀ sweep (4 jobs × 1 GPU on geodude, parallel)
├─ Phase 3: hidden-dim sweep (4 jobs × 1 GPU, after Phase 2 or on all6000)
└─ Phase 4: mask ablation (3 jobs on all6000, 4 GPU each — run 2 in parallel)

WEEK 2–3:
├─ Phase 5: pos-encoding ablation (4 jobs × 1 GPU on geodude)
└─ Phase 6: LR/WD sweep (depends on Phases 1—5 results)

WEEK 3–4:
└─ Phase 7: final patchification showdown (4 runs)

THEN:
└─ Phase 8: ImageNet scale-up (separate planning)
```

> \[!TIP\]
> **Parallel 1-GPU runs on geodude**: Since geodude has 4 GPUs, you can run up to 4 single-GPU jobs simultaneously (e.g. ω₀ sweep). Use `--gres=gpu:1 --exclusive=user` to avoid contention. Use `accumulate_grad_steps=4` to match effective batch size 128 (= 1 GPU × 32 × 4).

______________________________________________________________________

## Running Experiments

```bash
# Activate environment
conda activate nvsubq
source .env  # WandB API key + HF_TOKEN
export PYTHONPATH=.

# Single-GPU with gradient accumulation (effective BS=128)
srun --gres=gpu:1 -c 16 --mem=32G --partition=geodude --account=geodudeusers \
    python experiments/run.py \
    --config examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/<config>.py \
    train.accumulate_grad_steps=4

# 4-GPU on geodude (effective BS=128 = 4×32)
sbatch examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/run_<config>.sh

# SLURM script template for 1-GPU ablation:
# --partition=geodude --account=geodudeusers --gres=gpu:1
# --cpus-per-task=8 --mem=32G --time=48:00:00
# python experiments/run.py --config ... train.accumulate_grad_steps=4
```

______________________________________________________________________

## 🏆 Results

### Leaderboard (Best Results Per Architecture — TinyImageNet)

| Rank | Architecture | Config | Patchify | Val Acc | Val Loss | WandB |
| :--- | :----------- | :----- | :------- | :------ | :------- | :---- |
| —    | —            | —      | —        | —       | —        | —     |

### Patchification Impact

| Architecture | No Patch (4096 tok) | Patch-4 (256 tok) | Δ Acc | Notes |
| :----------- | :------------------ | :---------------- | :---- | :---- |
| Hyena        | —                   | —                 | —     | —     |
| Attention    | —                   | —                 | —     | —     |

### Ablation Summary

| Ablation          | Best Setting | Δ vs Default | Notes |
| :---------------- | :----------- | :----------- | :---- |
| Kernel type       | —            | —            | —     |
| ω₀                | —            | —            | —     |
| Kernel hidden-dim | —            | —            | —     |
| Mask              | —            | —            | —     |
| Pos-encoding      | —            | —            | —     |
| LR                | —            | —            | —     |
| Weight decay      | —            | —            | —     |

______________________________________________________________________

## Job Submission Log

> \[!IMPORTANT\]
> **Always update this log when submitting a job.** Record the job ID, config, and phase so we can trace results back to specific runs.

| Date             | Job ID   | Phase | Config                             | Cluster | Partition | GPUs | Status       | Val Acc | Notes                                                                         |
| :--------------- | :------- | :---- | :--------------------------------- | :------ | :-------- | :--- | :----------- | :------ | :---------------------------------------------------------------------------- |
| 2026-02-19       | `140280` | 1.1   | `hyena_patchify.py`                | IVI     | geodude   | 4    | 🔄 Running   | —       | SIREN baseline (ablation ref)                                                 |
| 2026-02-19       | `140281` | 1.2   | `hyena_patchify_rff.py`            | IVI     | geodude   | 4    | ⏳ Pending   | —       | RFF kernel ablation                                                           |
| 2026-02-17       | `137108` | 0.1   | `attention_patchify.py`            | IVI     | geodude   | 4    | ✅ Done      | 54.3%   | ViT-B baseline pipeline validation                                            |
| 2026-02-17       | `174875` | 0.2   | `hyena_patchify.py`                | hipster | perf      | 4    | ✅ Completed | 70.67%  | [9iqbx19w](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9iqbx19w) |
| 2026-02-17       | `174895` | 2.1   | `hyena_patchify.py` + ω₀=10        | hipster | capacity  | 2    | 🔄 Running   | 68.2%   | [yxxcr5wh](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yxxcr5wh) |
| 2026-02-17       | `174896` | 2.2   | `hyena_patchify.py` + ω₀=20        | hipster | capacity  | 2    | 🔄 Running   | 63.9%   | [c4x52706](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/c4x52706) |
| 2026-02-17       | `174897` | 2.4   | `hyena_patchify.py` + ω₀=60        | hipster | capacity  | 2    | 🔄 Running   | 61.8%   | [jc9bv226](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jc9bv226) |
| 2026-02-17       | `174898` | 2.5   | `hyena_patchify.py` + ω₀=100       | hipster | capacity  | 2    | 🔄 Running   | 22.9%   | [n86qahfw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n86qahfw) |
| 2026-02-19 00:58 | `139226` | 0.3   | `attention_patchify_imagenet1k.py` | IVI     | cees      | 8    | ❌ Stopped   | 9.3%    | Old config (LR=3e-3, no EMA/DropPath). Ran ~2 epochs.                         |
| 2026-02-19       | `140271` | 0.3   | `attention_patchify_imagenet1k.py` | IVI     | cees      | 8    | ❌ Cancelled | 6.7%    | v2: Cancelled due to NFS I/O bottleneck (0.10 it/s). Replaced by `140500`.    |
| 2026-02-19       | `140272` | 0.3   | `attention_patchify_imagenet1k.py` | IVI     | cees6000  | 8    | ❌ Cancelled | —       | Cancelled — cees6000 nodes fully occupied + GrpTRES cpu=128 shared limit      |
| 2026-02-20       | `140500` | 0.3   | `attention_patchify_imagenet1k.py` | IVI     | cees      | 8    | ❌ Cancelled | —       | v3: SSD staging too slow (3.4 MB/s rsync). Replaced by WebDataset approach.   |
| 2026-02-20       | `140516` | infra | WebDataset conversion              | IVI     | cees      | 0    | 🔄 Running   | —       | Converting HF Arrow → WebDataset TAR shards. ETA ~1–2h. CPU-only job.         |

______________________________________________________________________

## 📊 Observations & Insights

- **2026-02-19 19:20**: **Phase 0.2 (Hyena baseline) completed!**
  - Final validation accuracy: **70.67%**.
  - This is a massive improvement over the ViT-B (54.3%) and RFF baselines.
- **2026-02-19 23:10**: Status update on Phase 2 sweep:
  - Phase 2.1 (ω₀=10) is leading with **68.2%** accuracy at step ~200k.
  - All jobs are approaching completion on hipster/capacity.
- **2026-02-20 14:40**: Progress update on Phase 2 sweep:
  - All 4 ω₀ values converging to similar range (**69–70.5%** val acc), very close to Phase 0.2 (70.67%).
  - ω₀=20 currently leads at **70.5%** — may match or exceed the default ω₀=30.
  - ω₀=100 has recovered dramatically from earlier (22.9% → **70.1%**).
  - ETA: ω₀=10 finishes ~Feb 21 evening, ω₀=100 finishes ~Feb 22 evening.
- **2026-02-20 15:00**: Identified NFS I/O as root cause of 0.10 it/s throughput (40x slower than expected). HuggingFace Arrow cache (157 GB, 267 shards) on ZFS NFS causes massive random read latency with 112 data loader workers (14/GPU × 8 GPUs).
- **2026-02-20 17:10**: SSD staging approach abandoned — `rsync` at 3.4 MB/s would take 12+ hours. Pivoted to **WebDataset** (sequential TAR shard reads, no local copy needed).
- **2026-02-20 17:40**: WebDataset migration:
  - Created `experiments/datamodules/imagenet_wds.py` — drop-in `ImageNetWebDataModule` with same API.
  - Created `scripts/convert_imagenet_to_webdataset.py` — one-time converter (HF Arrow → TAR shards).
  - Added `USE_WEBDATASET = True` toggle in `attention_patchify_imagenet1k.py`.
  - Submitted conversion job `140516` (CPU-only, `ivi-cn020`). Output: `data/imagenet-wds/`.
  - Cancelled jobs `140500` (SSD staging) and `140271` (slow NFS).
  - **Next**: resubmit Phase 0.3 training once conversion finishes.
- **2026-02-20**: Cancelled job `140272` on cees6000. Root cause analysis:
  - **GrpTRES per-user limit on `ceesusers`**: `cpu=128, gres/gpu=8, mem=750G`
  - The run script requests `--cpus-per-task=96`. With job `140271` already consuming 96 CPUs on cees, the combined total (192) exceeds the 128 CPU group limit.
  - Both cees6000 nodes (`ivi-cn030`, `ivi-cn031`) were also fully occupied by other users.
  - **Fix for future cees6000 runs**: reduce `--cpus-per-task` to ≤32 (128 − 96 = 32 remaining), or wait until the cees job finishes. Alternatively, lower cees `--cpus-per-task` to free up headroom.
- **2026-02-19**: Phase 0.1 (ViT-B Attention) finished with **54.3% Val Acc**.
- **2026-02-21**: Phase 1.1 (Hyena SIREN baseline) is nearly complete. Current val accuracy is **70.3%** at step 288k. Accuracy is significantly higher than the Attention baseline.
- **2026-02-19**: Phase 0.1 (ViT-B Attention) finished with **54.3% Val Acc**. Stable convergence; success criteria (≥55%) nearly met.
- **2026-02-17**: Tracker created. Pipeline validation (Phase 0) is highest priority.
- **2026-02-17 22:35**: Submitted Phase 0.1 (ViT-B attention patchify) → Job `137108` on geodude (4× A5000). Estimated ~17–25h.
- **2026-02-17 23:00**: Submitted Phase 0.2 (Hyena patchify) → Job `174875` on hipster/performance (4× RTX 6000 Ada).
- **2026-02-17 23:00**: Initially submitted Phase 2 ω₀ sweep on performance partition — **moved to capacity (L4)** at 23:07 to avoid competing with Phase 0.2 for performance GPUs.
- **2026-02-17 23:07**: Resubmitted Phase 2 ω₀ sweep on hipster/capacity (L4, 1 GPU each, accum=4):
  - 2.1 ω₀=10 → Job `174887` (RUNNING on hipster-cn008)
  - 2.2 ω₀=20 → Job `174888` (RUNNING on hipster-cn009)
  - 2.4 ω₀=60 → Job `174889` (RUNNING on hipster-cn012)
  - 2.5 ω₀=100 → Job `174890` (RUNNING on hipster-cn013)
  - 2.3 ω₀=30 = Phase 0.2 (Job `174875`), no separate run needed.
- **2026-02-19 00:06**: Added Phase 0.3 — ViT-B/16 attention patchify on full ImageNet-1K as pipeline sanity check. Config: `attention_patchify_imagenet1k.py`. Script: `run_attention_patchify_imagenet1k_cees.sh`. Target: cees (8× A5000, ceesusers). Key changes vs TinyImageNet baseline: `patch_size=16` (196 tokens), `image_size=224`, `BATCH_SIZE=128/GPU` (eff. BS=1024), `LR=3e-3`, `use_three_augment=True`. Dataset cached under `data/imagenet` (symlink → ZFS). Can run in parallel with 0.1/0.2.
- **2026-02-19 00:12**: Submitted Phase 0.3 → Job `139175` on IVI/cees (8× RTX A5000, partition `cees`, account `ceesusers`). First run will download ILSVRC/imagenet-1k into `data/imagenet` (~140 GB).
- **2026-02-19 01:00**: Phase 0.3 resubmitted several times due to environment issues:
  - `139191`: Failed due to `torch` 2.10.0-dev / `torchvision` mismatch. Downgraded to stable 2.5.1.
  - `139193`: Failed due to missing `datasets` and `timm`. Fixed conda environment.
  - `139208`: Failed due to `PLACEHOLDER=None` bug in `lazy_config.py` causing `ImageNetDataModule` to receive a `DictConfig` instead of an object. Fixed `lazy_config.py`.
  - `139211`, `139218`: Failed due to `HF_TOKEN` not being exported from `.env`. Fixed Slurm script export logic.
  - `139226`: Ran ~2 epochs before disk quota exceeded (home storage full). Fixed by symlinking `runs/` to ZFS.
- **2026-02-19 01:02**: Synced augmentation strategy in `attention_patchify_imagenet1k.py` to match TinyImageNet (`RandAugment`).

______________________________________________________________________

**Last Updated**: 2026-02-21 23:35
**Status**: 🔄 Phase 1.1 finishing on IVI/geodude (Job `140280`), Phase 0.1 Done, Phase 1.2 Pending.
