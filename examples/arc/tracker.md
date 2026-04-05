# ARC-AGI Benchmark Progress Tracker

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

### 4. Val split uses held-out training tasks, not the evaluation split

**Current:** 10% of training tasks (40 tasks, ~124 examples, **1 val batch**) are withheld.  Metric resolution = 1/124 ≈ 0.8% — too coarse to show any gradient of improvement.  VARC trains on all 400 training tasks and evaluates on the separate 400-task evaluation split.

**Fix:** set `val_fraction=0` in the datamodule config and rely on `test_0/exact_match` (evaluation split) as the primary metric.  This also gives the model 40 more training tasks.

______________________________________________________________________

## 📊 Experiment Tracking

### 1. Offline Pretraining Logs

| Model  | Params | Global Batch Size | GPUs       | Training Time             | End `val/exact_match`      | Notes / WandB                                                                                                                                                                  |
| :----- | :----- | :---------------- | :--------- | :------------------------ | :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ARCViT | ~3.7M  | 256 (128×2)       | 2x geodude | ❌ Overfit (SLURM 153461) | 2.4% (best epoch 1)        | Config: `cfg_vit.py` · [WandB 97tf96wk](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/97tf96wk) · Bug: 50k iters = ~600 epochs                                      |
| ARCViT | ~3.7M  | 256 (128×2)       | 2x geodude | ✅ Done (SLURM 153525)    | 0.81% val / **0.72% test** | Config: `cfg_vit.py` · [WandB r1zfe0wv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r1zfe0wv) · 844 steps (~18 effective epochs); no overfitting; train loss ~0.8 |
| ARCViT | ~3.7M  | 256 (128×2)       | 2x geodude | 🔄 Running (SLURM 153598) | *Pending*                  | Fixed val: all 400 training tasks used; val = eval-split demos; TTT callback every 5 epochs                                                                                    |
| Hyena  | ~18M   | 256 (128×2)       | 2x geodude | *Pending*                 | *Pending*                  | Config: `cfg_hyena.py`                                                                                                                                                         |

### 2. TTT Final Evaluation (ARC-1)

| Model         | Offline Ckpt | TTT Epochs | Ensembles        | ARC-1 Accuracy | Notes                              |
| :------------ | :----------- | :--------- | :--------------- | :------------- | :--------------------------------- |
| VARC Paper    | N/A          | 100        | None             | 52-56%         | Original paper result (Single Run) |
| VARC Paper    | N/A          | 100        | 10 + color perm. | 60.4%          | Original paper result (Ensemble)   |
| ARCViT (Ours) | *Pending*    | 100        | None             | *Pending*      | Run 3 (fixed val) in progress      |
| Hyena (Ours)  | *Pending*    | 100        | None             | *Pending*      | Subquadratic benchmark             |
