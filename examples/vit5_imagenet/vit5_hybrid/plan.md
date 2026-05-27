# Plan: vit5_hybrid Experiment Launch Infrastructure

Session ID: `vit5_hybrid`

## Context

We need to launch 8 training experiments (4 hybrid configs x 2 patch sizes) from the TRACKER at `examples/vit5_imagenet/vit5_hybrid/TRACKER.md`. The key requirement is using a **new container image** (`nvsubquadratic-slurm-x86_64-04-17-2026.sqsh`) instead of the old one baked into the existing slurm scripts. The existing configs and training infrastructure are already correct — we only need a new submit script and a bug fix.

______________________________________________________________________

## Files Created/Modified

### 1. Created `scripts/slurm/submit_hybrid.sh` (new file)

Based on `scripts/slurm/submit_in1k_cls.sh` with these changes:

| What                    | Old value                                             | New value                                                                  |
| ----------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| Container image default | `.../farhadr/enroot/nvsubquadratic-slurm-x86_64.sqsh` | `.../amoradzadeh/hyena/enroot/nvsubquadratic-slurm-x86_64-04-17-2026.sqsh` |
| ImageNet paths          | `${CONTAINER_DATA}/imagenet`                          | `${CONTAINER_DATA}/imagenet_folder`                                        |
| Job name                | `nvsubq.v5patch`                                      | `nvsubq.v5hybrid`                                                          |
| Usage comments          | References v5_patch configs                           | References vit5_hybrid configs                                             |
| `PYTHON_CMD` block      | No HF/W&B cache dirs                                  | Add `HF_HOME`, `WANDB_DIR`, etc. env vars                                  |
| Config overrides        | No compile_mode                                       | Add `compile_mode=max-autotune-no-cudagraphs`                              |

**Security**: Uses `export WANDB_API_KEY="${WANDB_API_KEY:-}"` (safe fallback). No hardcoded secrets.

### 2. Fixed `scripts/slurm/queue.sh` (line 45) — config override passthrough bug

**Problem**: Extra args (like `net.patch_size=8`) were placed BEFORE the script name in the `sbatch` call, so sbatch tried to parse them as sbatch options and failed.

```bash
# Before (broken for config overrides):
sbatch ${dep_flag} "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}" "${SCRIPT_NAME}" "${CONFIG}"

# After (overrides passed as script positional args):
sbatch ${dep_flag} "${SCRIPT_NAME}" "${CONFIG}" "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}"
```

______________________________________________________________________

## No Changes Needed

- **Config files** (`full_attention.py`, `hybrid_ha.py`, `hybrid_hhha.py`, `full_hyena.py`) — already correct
- **TRACKER.md** — already has the right tables; results filled in after runs complete
- **`_base_config.py`** — `build_hybrid_net()` already handles `patch_size` via OmegaConf interpolation

______________________________________________________________________

## Launch Commands

From `/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena/vit5_nvsubq/`:

```bash
# Patch 16 (default batch=256, no grad accum, ~12 chained 4h jobs)
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/full_attention.py
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/full_hyena.py

# Patch 8 (batch=64, accum=4 to maintain effective batch 2048, ~20 chained jobs)
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/full_attention.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/full_hyena.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
```

**Batch size rationale for patch 8**: 784 tokens/image (vs 196 for patch 16). Default batch=256 will OOM on H100 80GB. The v5_patch experiments use batch=64 + accum=4 for patch 8, maintaining effective batch = 8 GPUs x 64 x 4 = 2048.

**Job chain count**: Patch-16 FLOPs ~9-10 GFLOPs -> 12 jobs is sufficient. Patch-8 FLOPs ~38-45 GFLOPs with 4x grad accum -> 20 jobs. Extra jobs are harmless (autoresume exits immediately if training is complete).

______________________________________________________________________

## Risk: Container may not have hybrid code

The code is baked into the container at `/workspaces/nvSubquadratic-private`. The new sqsh was built on 04-17-2026 — we need to verify it includes the `vit5_hybrid` configs (which are on the `dwromero/hybrid-vit5` branch). If it doesn't, training will fail at import time.

**Workaround**: Mount the local repo into the container by adding to the mounts in `submit_hybrid.sh`:

```
${WORKDIR}:/workspaces/nvSubquadratic-private
```

______________________________________________________________________

## Test Job

Single patch-8 test job submitted as validation before full launch:

```bash
sbatch scripts/slurm/submit_hybrid.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py \
    net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
```

## Verification Checklist

1. Submit a single patch-8 job, verify no OOM and correct batch config
1. Submit a single patch-16 job, check slurm output for successful training start
1. Check W&B for the run appearing under `vit5_hybrid` job group
1. If both pass, launch the full set of 8 chained experiments

## First Window Results (Patch 16 & 8)

Completed first 4h window for all 8 experiments. Steady-state throughput:

| Experiment     | Patch | Epochs in 4h | it/s  | imgs/s |
| -------------- | ----- | ------------ | ----- | ------ |
| full_attention | 8     | 85           | 16.45 | 1,053  |
| full_attention | 16    | 285          | 13.86 | 3,548  |
| hybrid_ha      | 8     | 55           | 11.08 | 709    |
| hybrid_ha      | 16    | 173          | 8.63  | 2,209  |
| hybrid_hhha    | 8     | 48           | 10.13 | 648    |
| hybrid_hhha    | 16    | 150          | 7.64  | 1,956  |
| full_hyena     | 8     | 45           | 9.35  | 599    |
| full_hyena     | 16    | 134          | 6.99  | 1,789  |

Note: it/s not directly comparable across patch sizes (batch/GPU differs: 256 for p16, 64 for p8).

______________________________________________________________________

## Phase 2: Patch 4, 2, 1 (4-node / 32 GPU)

Created `scripts/slurm/submit_hybrid_4node.sh` — identical to `submit_hybrid.sh` but with `--nodes=4`.

Batch config (4 nodes = 32 GPUs, effective batch = 2048):

| Patch | batch/GPU | accum_steps | tokens/img |
| ----- | --------- | ----------- | ---------- |
| 4     | 16        | 4           | 3,141      |
| 2     | 4         | 16          | 12,549     |
| 1     | 1         | 64          | 50,181     |

### Launch Commands (single test jobs)

```bash
# Patch 4
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py net.patch_size=4 dataset.batch_size=16 train.accumulate_grad_steps=4
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py net.patch_size=4 dataset.batch_size=16 train.accumulate_grad_steps=4
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py net.patch_size=4 dataset.batch_size=16 train.accumulate_grad_steps=4
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/full_hyena.py net.patch_size=4 dataset.batch_size=16 train.accumulate_grad_steps=4

# Patch 2
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py net.patch_size=2 dataset.batch_size=4 train.accumulate_grad_steps=16
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py net.patch_size=2 dataset.batch_size=4 train.accumulate_grad_steps=16
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py net.patch_size=2 dataset.batch_size=4 train.accumulate_grad_steps=16
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/full_hyena.py net.patch_size=2 dataset.batch_size=4 train.accumulate_grad_steps=16

# Patch 1
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py net.patch_size=1 dataset.batch_size=1 train.accumulate_grad_steps=64
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py net.patch_size=1 dataset.batch_size=1 train.accumulate_grad_steps=64
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py net.patch_size=1 dataset.batch_size=1 train.accumulate_grad_steps=64
sbatch scripts/slurm/submit_hybrid_4node.sh examples/vit5_imagenet/vit5_hybrid/full_hyena.py net.patch_size=1 dataset.batch_size=1 train.accumulate_grad_steps=64
```

### Account Assignment

- Patch 16/8 chain jobs (second wave): `healthcareeng_bionemo`
- Patch 4/2/1 test jobs: `healthcareeng_research`

______________________________________________________________________

## Final Results — Patch 8 & 16 (800 epochs)

All 7 priority experiments completed. Full_hyena p16 stopped at 799 but is treated as final (val_acc_ema stable).

| Experiment     | Patch 16       | Patch 8   |
| -------------- | -------------- | --------- |
| full_attention | 0.817          | **0.835** |
| hybrid_ha      | **0.819**      | 0.829     |
| hybrid_hhha    | 0.815          | 0.826     |
| full_hyena     | 0.813 (ep 799) | 0.825     |

**Key takeaways:**

- Hybrid architectures match/beat pure attention at p16 (hybrid_ha = 0.819 vs full_attn = 0.817).
- Patch 8 gives ~2% absolute accuracy boost across all variants.
- Within p16, all 4 variants within 0.6%. Within p8, all 4 within 1.0%.
- Throughput crossover: Hyena > Attention between patch 4 and patch 2 (~3K–12K tokens).

______________________________________________________________________

## Phase 3: KERNEL_OMEGA_0 Ablation (Patch 8, 4 nodes)

Ablation study launched 2026-04-21 to test whether a higher SIREN-kernel base frequency improves Hyena-containing configs at patch 8.

**Base value**: `KERNEL_OMEGA_0 = 10.0` (set in `_base_config.py:69`)
**Ablation value**: `KERNEL_OMEGA_0 = 20.0` (patch-8)

### Approach: CLI override (zero code changes)

Override path (verified via smoke test):

```
net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=20.0
```

Existing configs (`full_hyena.py`, `hybrid_hhha.py`, `hybrid_ha.py`) are unmodified. Different CLI overrides produce different run-hashes, so results land in separate `run_<hash>/` dirs — no collision with original omega=10 runs.

### Configuration

- **Scope**: Only Hyena-containing configs (`full_hyena`, `hybrid_hhha`, `hybrid_ha`). `full_attention` has no H block and is skipped.
- **Hardware**: 4 nodes × 8 GPUs = 32 GPUs.
- **Batch config**: `dataset.batch_size=64 train.accumulate_grad_steps=1` → effective batch = 32 × 64 × 1 = **2048** (matches original).
- **Iters/epoch**: ~626 (vs 2502 for 1-node runs). Each iter does 4× more work, net wall-clock per epoch similar or faster.
- **Expected epochs/window**: ~150–200 epochs per 4h window → **~5 chain jobs to reach 800**.

### Launch Commands

```bash
OMEGA_OVR='net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0=20.0'

# full_hyena p8 omega=20.0
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid_4node.sh 7 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena.py \
    net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=1 $OMEGA_OVR

# hybrid_hhha p8 omega=20.0
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid_4node.sh 7 \
    examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py \
    net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=1 $OMEGA_OVR

# hybrid_ha p8 omega=20.0
bash scripts/slurm/queue.sh scripts/slurm/submit_hybrid_4node.sh 7 \
    examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py \
    net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=1 $OMEGA_OVR
```

### Baseline for comparison

| Config (p8, omega=10) | val/acc_ema | Patch-8 omega=20.0 |
| --------------------- | ----------- | ------------------ |
| full_hyena            | 0.825       | TBD                |
| hybrid_hhha           | 0.826       | TBD                |
| hybrid_ha             | 0.829       | TBD                |

______________________________________________________________________

## Notes on Pending / Future Ablations

- **Patch 4 original (omega=10.0) runs**: in progress from Phase 2.
- **Patch 2 / Patch 1**: only throughput-benchmark runs so far (patch 1 preempted on backfill partition due to very long epochs).

______________________________________________________________________

## Verification Checklist

1. ~~Submit a single patch-8 job, verify no OOM and correct batch config~~ DONE
1. ~~Submit a single patch-16 job, check slurm output for successful training start~~ DONE
1. ~~Check W&B for the run appearing under `vit5_hybrid` job group~~ DONE
1. ~~If both pass, launch the full set of 8 chained experiments~~ DONE
1. ~~Monitor patch 4/2/1 test jobs for OOM~~ DONE (patch 1 preempted, not retried)
1. ~~Chain full 800-epoch p8/p16 runs~~ DONE (7/8 complete; full_hyena p16 at ep 799)
1. ~~Verify omega_0 CLI override resolves correctly~~ DONE (smoke test 2026-04-21)
1. Run omega=20.0 ablation for hyena-containing p8 configs and compare to omega=10 baselines
