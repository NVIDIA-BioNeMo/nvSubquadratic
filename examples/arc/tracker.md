# ARC-AGI Benchmark Progress Tracker

> **Agent instructions**
>
> - **Always use `fft_backend="subq_ops"`** in `CKConvND` for all Hyena-based ARC configs. This gives a free ~1.4–1.5x throughput improvement over `torch.fft` (see profiling results below). Do **not** set `compile_compatible_fftconv=True` when using subq_ops — it is only needed for the torch.fft path. Reference configs: `cfg_hyena_rearc_subq_ops.py`, `cfg_hyena_rearc_film_subq_ops.py`.

This document tracks the progress of implementing the ARC-AGI benchmark replication based on the VARC paper (*"ARC is a Vision Problem!"*) and our subsequent comparison using a subquadratic Hyena-ResNet.

______________________________________________________________________

## 🧠 The "Why" and "How"

### Why this methodology?

The VARC paper casts ARC as an image-to-image translation problem using Vision Transformers (or U-Nets). It introduces a **Task Token**—a learnable embedding prepended to the sequence that acts as the "rule" or "concept" for that specific ARC task.

Because of this design, the model **cannot be evaluated zero-shot on new tasks**. An unseen task has no existing Task Token. Without test-time training (TTT), a random task token yields effectively 0% accuracy.

### How do we reproduce this?

We split the pipeline into two strict phases:

1. **Offline Pretraining**: The model learns general visual priors and task-specific rules for the 400 *training* tasks (assisted by RE-ARC augmentations). This maps to a standard PyTorch Lightning training loop.
1. **Test-Time Training (TTT) Evaluator**: A custom standalone script. For each unseen evaluation task, we:
   - Discard the old task tokens.
   - Initialize a random task token.
   - Spawn a fresh optimizer.
   - Fine-tune the network for 100 epochs on the task's *demonstration (support)* pairs.
   - Make the final prediction on the *query* images.
   - Reload the base offline checkpoint for the next task.

______________________________________________________________________

## 🛠️ Execution Plan & Checklists

### Phase 1: Baseline Setup & Offline Pretraining

- [x] Implement `ARCViT` baseline architecture (`arc_vit.py`).
- [x] Implement standard `ARCWrapper` for pixel-level cross-entropy and exact match metrics.
- [x] Write base configuration (`examples/arc/_base.py`) and ViT config (`examples/arc/cfg_vit.py`).
- [x] Write tests validating model shapes and wrapper metrics.
- [x] Determine max batch sizes and hardware requirements (Max BS=384 on 1x A5000; requires `BATCH_SIZE=128` on 2x GPUs to match VARC's global 256).
- [x] Create SLURM submission scripts (`submit_arc_2gpu_geodude.sh`).
- [x] **Executable:** Run the offline pretraining for ARCViT.
  ```bash
  sbatch examples/arc/submit_arc_2gpu_geodude.sh examples/arc/cfg_vit.py
  ```

### Phase 2: Test-Time Training (TTT)

- [ ] Create `scripts/evaluation/eval_arc_ttt.py`.
- [ ] Implement task-token re-initialization and mini 100-epoch training loop per task.
- [ ] Implement saving of prediction results and calculation of final Exact Match on the evaluation set.
- [ ] **Executable:** Run TTT evaluation on the ARCViT offline checkpoint.
  ```bash
  python scripts/evaluation/eval_arc_ttt.py --checkpoint logs/arc_vit_.../last.ckpt --config examples/arc/cfg_vit.py
  ```

### Phase 3: Ensembling & Augmentation

- [ ] Expand the TTT script to handle $N$ random attempts (e.g., 10 runs per task) and color permutations (9 per task).
- [ ] Add majority-voting logic across the ensemble to reproduce the paper's peak performance (~60.4% ARC-1).

### Phase 4: Hyena-ResNet Implementation & Comparison

- [ ] Implement `ARCResNet` wrapper incorporating discrete-color `nn.Embedding`s into the standard codebase ResNet backbone.
- [ ] Author `examples/arc/cfg_hyena.py` with matched parameters (~18M) using the `CKConvND` + `Hyena` blocks.
- [ ] **Executable:** Run offline pretraining for Hyena-ResNet.
- [ ] **Executable:** Run TTT evaluation for Hyena-ResNet.

______________________________________________________________________

## ⚠️ Known Divergences from VARC (revisit if results don't reproduce)

### 1. `max_size`: 32 (ours) vs 30 (VARC)

**Why we differ:** 32 gives cleaner patch arithmetic (32/2=16 patches/side vs 15), avoids a latent crash, and the larger canvas costs negligible compute.

**Impact:** Slightly larger usable canvas (30×30 vs 28×28), different sequence length (256 vs 225 patch tokens), slightly different resolution-augmentation scale distribution.

**Latent VARC bug discovered:** 24 training examples have max grid dimension 29–30. With VARC's `max_size=30`, the resolution augmentation calls `randint(1, 0)` for these (raising `ValueError`), and the canvas placement overflows. VARC probably never noticed because DataLoader worker crashes are silently retried. **Our fix (already applied):** filter examples where `max_dim > max_size - 2` at index-build time in `ARCDataset.__init__`. This filter is safe for both `max_size=30` and `max_size=32`.

**If switching back to 30:** change `max_size=32` → `30` in `examples/arc/_base.py` and `examples/arc/cfg_vit.py`. The filter already handles the crash, no other code changes needed.

### 2. `num_colors`: 12 (ours) vs 10 (VARC)

**Why we differ:** The input canvas contains `IGNORE_INDEX=10` (padding) and `PAD_INDEX=11` (border markers). With `num_colors=10`, `nn.Embedding(10, embed_dim)` would receive out-of-bounds indices 10 and 11. VARC relies on CUDA's silent UB for out-of-bounds embedding lookups (the garbage embeddings get masked out by the attention mask). Our `num_colors=12` is explicit and safe on both CPU and GPU. **Fix already applied:** `pixel_values.long().clamp(0, self.num_colors - 1)` before the embedding lookup in `arc_vit.py` — this makes both `num_colors=10` and `num_colors=12` safe.

**Impact of `num_colors=12`:** the classification head has 12 output channels instead of 10, so `argmax` can predict the spurious classes 10 or 11. In practice this rarely happens (those logits are untrained), but it is a correctness gap vs VARC.

**If switching back to 10:** set `num_colors=10` in `examples/arc/cfg_vit.py`. The clamp is already in place so no other code changes needed.

### 3. `training_iterations` formula doesn't account for DDP `num_gpus`

**Current formula:** `ceil(NUM_EPOCHS × NUM_TRAINING_SAMPLES / batch_size)` = 844 steps with batch_size=128.  With 2 GPUs, each epoch takes 46 steps (not 84), so 844 steps = **~18 effective epochs**, not 10.

**Fix:** divide by `num_gpus`: `ceil(10 × 10_800 / (128 × 2)) = 422` for exact 10-epoch correspondence.  Or accept 18 epochs as a reasonable approximation (no overfitting observed at this scale).

### 5. Color permutation strategy: fixed copies (ours) vs on-the-fly (VARC)

**Current:** `ARCDataset` pre-generates 9 fixed colour permutations per example at index-build time, storing 10 copies (1 identity + 9 fixed) in the flat index. One full pass through the dataset sees the same 9 perms repeatedly.

**VARC:** samples a fresh random permutation in `__getitem__` on every access — a different perm each epoch, giving strictly more variety over the course of training.

**Impact:** over many epochs our model sees less colour diversity per example than VARC does. This could hurt generalisation if colour invariance is important to performance.

**If switching:** replace the pre-generated perm loop in `ARCDataset.__init__` with a single entry per example (no perm), and apply a random `perm_arr` sampled from `random.sample(range(10), 10)` inside `__getitem__`. This would also shrink the index by ~10×, reducing memory and data-loading overhead.

### 4. Val split uses held-out training tasks, not the evaluation split

**Current:** 10% of training tasks (40 tasks, ~124 examples, **1 val batch**) are withheld.  Metric resolution = 1/124 ≈ 0.8% — too coarse to show any gradient of improvement.  VARC trains on all 400 training tasks and evaluates on the separate 400-task evaluation split.

**Fix:** set `val_fraction=0` in the datamodule config and rely on `test_0/exact_match` (evaluation split) as the primary metric.  This also gives the model 40 more training tasks.

______________________________________________________________________

## 📊 Experiment Tracking

### 1. Offline Pretraining Logs

#### Our codebase (ARCViT + Hyena)

| Component                      | ARCViT (dim=512) |                  Hyena (dim=384) |
| ------------------------------ | ---------------: | -------------------------------: |
| `color_embed` (12 × dim)       |             6.1K |                             4.6K |
| `task_embed` (400 × dim)       |           204.8K |                           153.6K |
| `positional_embed` (256 × dim) |           131.1K | — (SIREN uses continuous coords) |
| Patch / Patchify               |            1.05M |                             590K |
| Encoder / Residual blocks      |           15.78M |                           18.27M |
| Head / Unpatchify              |            24.6K |                            18.4K |
| **Total**                      |       **17.20M** |                       **19.04M** |

Note: Models are within ~11% total capacity. The embedding-block gap (342K vs 158K) is driven by (a) different `EMBED_DIM` (512 vs 384) and (b) Hyena's lack of a learned positional embedding — it isn't a bug, it's an architectural asymmetry.

| Model                             | Params | Global BS         | GPUs       | Status                      | Best `val/exact_match` | Notes / WandB                                                                                                                                                                                                                                                                                                                                    |
| :-------------------------------- | :----- | :---------------- | :--------- | :-------------------------- | :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ❌ Overfit (SLURM 153461)   | 2.4% (epoch 1)         | Config: `cfg_vit.py` · [WandB 97tf96wk](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/97tf96wk) · Bug: 50k iters = ~600 epochs                                                                                                                                                                                                        |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ✅ Done (SLURM 153525)      | 0.81% val / 0.72% test | Config: `cfg_vit.py` · [WandB r1zfe0wv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r1zfe0wv) · 844 steps (~18 effective epochs); train loss ~0.8                                                                                                                                                                                   |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ❌ Cancelled (SLURM 153598) | *N/A*                  | Fixed val config; cancelled to fix max_size bug (was 64 instead of 32)                                                                                                                                                                                                                                                                           |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ❌ Cancelled (SLURM 153747) | 31.2% (epoch 49)       | Config: `cfg_vit_rearc.py` · [WandB q4uuikbj](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/q4uuikbj) · Cancelled at wall-time; max_size=64 (wrong, was a bug)                                                                                                                                                                        |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ❌ OOM (SLURM 153872)       | **73.4%** (epoch ~190) | Config: `cfg_vit_rearc.py` · [WandB rzdmi235](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rzdmi235) · max_size=32 · RE-ARC · OOM'd at epoch ~215 · best ckpt `epoch=190-step=307892.ckpt`                                                                                                                                           |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ❌ Cancelled (SLURM 154328) | 63.2% (epoch ~108)     | Config: `cfg_vit_rearc.py` · [WandB rzdmi235](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rzdmi235) · Restart of 153872; cancelled at epoch ~108 — regression vs prior session                                                                                                                                                      |
| Hyena                             | 19.04M | 256 (128×2)       | 2x geodude | ❌ Cancelled (SLURM 154329) | 54.7% (epoch ~138)     | Config: `cfg_hyena_rearc.py` · [WandB orhmff04](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/orhmff04) · fresh restart of 153871; cancelled after ~138 epochs                                                                                                                                                                        |
| Hyena                             | 19.04M | 256 (128×2)       | 2x geodude | ❌ Failed (SLURM 154555)    | **66.4%** (epoch ~500) | Config: `cfg_hyena_rearc_subq_ops.py` · [WandB orhmff04](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/orhmff04) · Continuation of 154329 · failed 2026-04-12 08:26 (exit -6/-15, likely OOM) · best ckpt `epoch=356-step=575484.ckpt`                                                                                                |
| Hyena FiLM                        | 19.04M | 256 (128×2)       | 2x geodude | ❌ Timeout (SLURM 154232)   | **59.1%** (epoch ~193) | Config: `cfg_hyena_rearc_film_subq_ops.py` · [WandB qbfg8qty](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qbfg8qty) · max_size=32 · FiLM conditioning · hit wall-time 2026-04-11 · best ckpt `epoch=191-step=309504.ckpt`                                                                                                           |
| Hyena FiLM                        | 19.04M | 256 (256×1)       | 1x a6000   | ❌ Cancelled (SLURM 154787) | *n/a*                  | Underperforming vs broadcast & AdaLN; cancelled to free GPU                                                                                                                                                                                                                                                                                      |
| Hyena AdaLN                       | 19.04M | 256 (128×2)       | 2x geodude | ❌ Timeout (SLURM 154569)   | **73.0%** (epoch ~431) | Config: `conditioning_ablation/cfg_hyena_rearc_adaln_subq_ops.py` · [WandB k70b39dq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/k70b39dq) · DiT-style AdaLN-Zero conditioning · hit wall-time 2026-04-12 · best ckpt `epoch=431-step=696384.ckpt` · near-parity with ARCViT (73.4%)                                                |
| Hyena AdaLN                       | 19.04M | 256 (128×2)       | 2x geodude | ❌ Failed (SLURM 154948)    | *n/a*                  | Config: `conditioning_ablation/cfg_hyena_rearc_adaln_subq_ops.py` · Continuation of 154569 · failed immediately: optimizer param group mismatch (checkpoint from before tanh/WD changes in `residual_block.py`)                                                                                                                                  |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | ❌ Failed (SLURM 154953)    | *n/a*                  | Config: `cfg_vit_rearc.py` · Checkpoint key mismatch: `color_embed`/`task_token_embed` renamed to `embedding.color_embed`/`embedding.task_embed` after ARCColorTaskEmbedding refactor · keys remapped in-place, backups saved as `*.bak`                                                                                                         |
| ARCViT                            | 17.20M | 256 (128×2)       | 2x geodude | 🔄 Running (SLURM 154955)   | **83.6%** (epoch 439)  | Config: `cfg_vit_rearc.py` · [WandB rzdmi235](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rzdmi235) · Continuation of 153872 · resumed from `epoch=190-step=307892.ckpt` · best ckpt `epoch=439-step=709280.ckpt` · currently epoch ~458, val=83.1% · still improving                                                               |
| Hyena AdaLN (stable)              | 24.35M | 256 (128×2)       | 2x geodude | 🔄 Running (SLURM 154949)   | **70.4%** (epoch 338)  | Config: `conditioning_ablation/cfg_hyena_rearc_adaln_stable_subq_ops.py` · [WandB dxnvk65v](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dxnvk65v) · Stabilised: tanh gate + condition_proj WD=1e-4 · best ckpt `epoch=338-step=546468.ckpt` · currently epoch ~338, val=69.8%                                                       |
| Hyena (grid=double)               | 13.70M | 256 (256×1)       | 1x a6000   | 🔄 Running (SLURM 154950)   | **65.5%** (epoch 278)  | Config: `cfg_hyena_rearc_subq_ops_grid_double.py` · [WandB ts8jamnr](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ts8jamnr) · grid_type='double': 31×31 SIREN kernel vs 15×15 baseline · best ckpt `epoch=278-step=449748.ckpt` · currently epoch ~278, val=65.1%                                                                    |
| Hyena (patch=1)                   | 13.25M | 256 (64×1×accum4) | 1x a6000   | 🔄 Running (SLURM 21874746) | **60.7%** (epoch 52)   | Config: `cfg_hyena_rearc_subq_ops_patch1.py` · [WandB lac5r48z](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lac5r48z) · pixel-space seq_len=1024 · Resumed from `epoch=52-step=85489.ckpt` (wall-time cancel of 154951) · resuming from epoch 54                                                                                    |
| ARCViT AdaLN                      | 32.96M | 256 (128×2)       | 2x geodude | ❌ Failed (SLURM 21881996)  | *n/a*                  | Config: `conditioning_ablation/cfg_vit_rearc_adaln.py` · Interrupted around Epoch 4575 via KeyboardInterrupt from dataloader multiprocessing · DiT-style AdaLN-Zero on ViT · task token NOT prepended (removed from sequence) · **expected to stagger**: ViT's native strength is task token participating in every attention layer; AdaLN replaces that with a weaker side-channel → removing sequence participation likely hurts ViT |
| Hyena (grid=single + circular)    | ~19M   | 256 (128×2)       | 2x geodude | ⏳ Not started              | *n/a*                  | Config: `cfg_hyena_rearc_subq_ops_circular.py` (to create) · identical to baseline subq_ops but `fft_padding="circular"` · ablation: does periodic padding help with edge-task generalisation in the 16×16 patch grid?                                                                                                                           |
| Hyena (per-layer broadcast)       | 19.04M | 256 (128×2)       | 2x H100    | 🔄 Running (SLURM 21931841) | *n/a*                  | Config: `conditioning_ablation/cfg_hyena_rearc_per_layer_broadcast_subq_ops.py` · re-inject task token additively before each residual block via `AdditiveCondResidualBlock(cond_gate="none")` · zero extra params · `task_injection="per_layer_broadcast"` on ARCResNet |
| Hyena (per-layer broadcast gated) | 19.04M | 256 (128×2)       | 2x H100    | 🔄 Running (SLURM 21931842) | *n/a*                  | Config: `conditioning_ablation/cfg_hyena_rearc_per_layer_broadcast_gated_subq_ops.py` · same as above but `cond_gate="scalar_zero_init"` → 12 extra scalar params (one tanh gate per block, zero-init) · starts identical to unconditional ResNet at step 0                                                                                       |
| Hyena (concat / task-in-sequence) | ~19M   | 256 (128×2)       | 2x geodude | ⏳ Not started              | *n/a*                  | Config: `cfg_hyena_rearc_seqtoken.py` (to create) · prepend task token as extra spatial "row" in the 2D feature map (ViT5-style concat conditioning) · Option D from conditioning ablation tracker · requires ARCResNet.forward() changes                                                                                                        |

#### VARC reference codebase

| Model  | Params | Global BS | GPUs       | Status                      | Best `eval_acc`  | Notes / WandB                                                                                                                |
| :----- | :----- | :-------- | :--------- | :-------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| ARCViT | 17.4M  | 64 (32×2) | 2x geodude | ❌ Cancelled (SLURM 153674) | 50.5% (epoch 31) | [WandB 3z3qtr79](https://wandb.ai/dafidofff/nvsubquadratic/runs/3z3qtr79) · Hit wall-time; strong trajectory, ~41h remaining |

### 2. Training Speed Profiling (A5000, 2026-04-10)

Profiled on NVIDIA RTX A5000 (`ivi-cn024`), batch size 128, 50 measurement steps.
Scripts: `benchmarks/arc/profile_training_bottleneck.py`, `benchmarks/arc/submit_profile_arc_geodude.sh`.
Results JSON: `benchmarks/arc/profile_2026-04-10.jsonl`.

#### Per-step timing breakdown (compiled mode)

| Component           |         ARCViT | Hyena (torch.fft) | Hyena (subq_ops) | Hyena FiLM (torch.fft) | Hyena FiLM (subq_ops) |
| ------------------- | -------------: | ----------------: | ---------------: | ---------------------: | --------------------: |
| Data fetch + to_gpu |          0.4ms |             0.4ms |            0.9ms |                  0.4ms |                 0.4ms |
| Forward + loss      |         37.4ms |            75.2ms |           52.9ms |                 98.8ms |                80.8ms |
| Backward            |        117.8ms |           156.0ms |          102.7ms |                194.9ms |               129.7ms |
| Grad clip           |          1.3ms |             2.1ms |            1.9ms |                  2.2ms |                 2.3ms |
| Optimizer step      |          2.6ms |             2.8ms |            2.8ms |                  3.1ms |                 3.3ms |
| **Full step**       |    **159.7ms** |       **236.7ms** |      **161.6ms** |            **299.7ms** |           **216.8ms** |
| **Throughput**      | **802 samp/s** |    **541 samp/s** |   **792 samp/s** |         **427 samp/s** |        **590 samp/s** |

#### Eager vs compiled comparison

|                     |    ARCViT | Hyena (torch.fft) | Hyena (subq_ops) | Hyena FiLM (torch.fft) | Hyena FiLM (subq_ops) |
| ------------------- | --------: | ----------------: | ---------------: | ---------------------: | --------------------: |
| Eager               |     250ms |             392ms |            333ms |                  447ms |                 378ms |
| Compiled            |     160ms |             237ms |            162ms |                  300ms |                 217ms |
| **Compile speedup** | **1.57x** |         **1.65x** |        **2.06x** |              **1.49x** |             **1.74x** |
| **vs ViT compiled** |         — |            −0.68x |          ≈parity |                 −0.53x |                −0.74x |

#### Key findings

1. **Massively compute-bound across all models.** Data loading is ~0.4ms vs 160–447ms of compute. No benefit from tuning workers, prefetch factor, or data pipeline.
1. **`torch.compile` is effective.** ViT: 1.57x, Hyena: 1.65x, Hyena FiLM: 1.49x. The backward pass benefits most.
1. **`subquadratic_ops_torch` (`fft_backend="subq_ops"`) is a major win.** It closes Hyena's gap to ViT completely (162ms vs 160ms) and significantly narrows it for FiLM (217ms vs 160ms). The combined eager→compiled + torch.fft→subq_ops gain is **2.42x** for plain Hyena.
1. **FiLM adds ~27% overhead over plain Hyena** (300ms vs 237ms compiled torch.fft). The extra cost is split evenly across forward (+32%) and backward (+25%) from the per-block FiLM generator MLPs across 12 residual blocks. With subq_ops this gap narrows: 217ms vs 162ms (+34%).
1. **subq_ops speedup is slightly smaller for FiLM (1.38x) than plain Hyena (1.47x)**, since the FiLM MLP compute runs in standard torch ops that don't benefit from the custom FFT kernel.

#### Recommended action

Always use `fft_backend="subq_ops"` in `CKConvND` for all Hyena-based ARC configs. Reference configs: `cfg_hyena_rearc_subq_ops.py`, `cfg_hyena_rearc_film_subq_ops.py`. The `compile_compatible_fftconv` flag is not needed with this backend.

______________________________________________________________________

### 3. TTT Final Evaluation (ARC-1)

| Model                     | Offline Ckpt               | TTT Epochs | Ensembles        | ARC-1 Accuracy | Notes                                                                                          |
| :------------------------ | :------------------------- | :--------- | :--------------- | :------------- | :--------------------------------------------------------------------------------------------- |
| VARC Paper                | N/A                        | 100        | None             | 52-56%         | Original paper result (Single Run)                                                             |
| VARC Paper                | N/A                        | 100        | 10 + color perm. | 60.4%          | Original paper result (Ensemble)                                                               |
| ARCViT (Ours)             | epoch=190-step=307892.ckpt | 100        | None             | *Pending*      | SLURM 153872 OOM'd · offline best 73.4% · not currently running                                |
| Hyena (Ours)              | epoch=356-step=575484.ckpt | 100        | None             | *Pending*      | SLURM 154555 running · offline best 66.4% (ep ~432) · still improving                          |
| Hyena FiLM (Ours)         | epoch=191-step=309504.ckpt | 100        | None             | *Pending*      | SLURM 154786 running (continuation of 154232) · offline best 59.1% (ep ~193) · still improving |
| Hyena AdaLN (Ours)        | TBD                        | 100        | None             | *Pending*      | 154948 failed (optimizer mismatch); fresh stable run at 154949                                 |
| Hyena AdaLN stable (Ours) | TBD (154949 running)       | 100        | None             | *Pending*      | Fresh start with tanh gate + WD + cond dropout · SLURM 154949 running                          |

______________________________________________________________________

### 4. Key Experimental Insights (2026-04-15)

#### A. grid_type single vs double — use single for ablations, double for final run

- **Observation:** grid=double reaches slightly higher val exact match at end of training vs grid=single, but lowers throughput from ~4 it/s to ~3.5 it/s (~12% slower) due to the larger 31×31 SIREN kernel and bigger FFT.
- **Conclusion:** keep `grid_type='single'` for all ablations (faster iteration). Once best conditioning is established, run one final high-quality run with `grid_type='double'`.
- **Still to try:** `grid_type='single'` + `fft_padding='circular'` — circular padding may help edge-pixel tasks without the throughput cost of double. Create `cfg_hyena_rearc_subq_ops_circular.py`.

#### B. AdaLN improves Hyena; AdaLN likely hurts ViT

- **Observation:** Hyena broadcast (broadcast-only) peaked at 66.4% (~500 ep). Hyena AdaLN reached 73.0% at ep ~431, near-parity with ARCViT's 73.4%. This is a clear gain from per-layer residual-stream modulation.
- **Why AdaLN helps Hyena:** Hyena has no native mechanism for the task token to influence intermediate feature activations — the broadcast injects it once at the input embedding and it dissipates. AdaLN re-injects shift/scale/gate at every block, giving the task token ongoing influence throughout the residual stream.
- **Why AdaLN is expected to hurt ViT:** ARCViT's task token lives inside the attention sequence and participates in every self-attention layer. `conditioning_mode='adaln'` removes the token from the sequence entirely and replaces it with a side-channel — a strictly weaker information path for ViT.
- **Still to try:**
  - Multi-layer broadcast (Option A from conditioning ablation tracker): re-add task token additively before each block, zero extra params. Good lower-bound check.
  - ViT5-style concat conditioning for Hyena (Option D): prepend task token as extra spatial row in the 2D feature map. Closest analog to ViT's in-sequence participation.

#### C. Pixel space (patch=1) shows strong early-training signal — need more epochs

- **Observation:** Hyena (patch=1) reached 60.7% at epoch 52. For context, the baseline Hyena broadcast was well below that at the same epoch count (still in the fast-rise phase ~30–40%). This suggests pixel-space granularity provides a meaningful inductive bias for ARC's discrete-color grids.
- **Caveat:** Pixel space is 4× slower per epoch (seq_len=1024 vs 256) and uses grad accum (4 steps). The epoch-count comparison is not apples-to-apples in wall-clock time. Need to let this run to ~200+ epochs to draw firm conclusions.
- **Next logical step:** once pixel space is validated at longer training, try combining `patch_size=1` + AdaLN conditioning (the two improvements are independent and likely additive).
