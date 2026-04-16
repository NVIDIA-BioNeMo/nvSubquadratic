# Plan: vit5_hybrid Experiment Launch Infrastructure

Session ID: `vit5_hybrid`

## Context

We need to launch 8 training experiments (4 hybrid configs x 2 patch sizes) from the TRACKER at `examples/vit5_imagenet/vit5_hybrid/TRACKER.md`. The key requirement is using a **new container image** (`nvsubquadratic-slurm-x86_64-04-17-2026.sqsh`) instead of the old one baked into the existing slurm scripts. The existing configs and training infrastructure are already correct — we only need a new submit script and a bug fix.

---

## Files Created/Modified

### 1. Created `slurm/submit_hybrid.sh` (new file)

Based on `slurm/submit_in1k_cls.sh` with these changes:

| What | Old value | New value |
|------|-----------|-----------|
| Container image default | `.../farhadr/enroot/nvsubquadratic-slurm-x86_64.sqsh` | `.../amoradzadeh/hyena/enroot/nvsubquadratic-slurm-x86_64-04-17-2026.sqsh` |
| ImageNet paths | `${CONTAINER_DATA}/imagenet` | `${CONTAINER_DATA}/imagenet_folder` |
| Job name | `nvsubq.v5patch` | `nvsubq.v5hybrid` |
| Usage comments | References v5_patch configs | References vit5_hybrid configs |
| `PYTHON_CMD` block | No HF/W&B cache dirs | Add `HF_HOME`, `WANDB_DIR`, etc. env vars |
| Config overrides | No compile_mode | Add `compile_mode=max-autotune-no-cudagraphs` |

**Security**: Uses `export WANDB_API_KEY="${WANDB_API_KEY:-}"` (safe fallback). No hardcoded secrets.

### 2. Fixed `slurm/queue.sh` (line 45) — config override passthrough bug

**Problem**: Extra args (like `net.patch_size=8`) were placed BEFORE the script name in the `sbatch` call, so sbatch tried to parse them as sbatch options and failed.

```bash
# Before (broken for config overrides):
sbatch ${dep_flag} "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}" "${SCRIPT_NAME}" "${CONFIG}"

# After (overrides passed as script positional args):
sbatch ${dep_flag} "${SCRIPT_NAME}" "${CONFIG}" "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}"
```

---

## No Changes Needed

- **Config files** (`full_attention.py`, `hybrid_ha.py`, `hybrid_hhha.py`, `full_hyena.py`) — already correct
- **TRACKER.md** — already has the right tables; results filled in after runs complete
- **`_base_config.py`** — `build_hybrid_net()` already handles `patch_size` via OmegaConf interpolation

---

## Launch Commands

From `/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena/vit5_nvsubq/`:

```bash
# Patch 16 (default batch=256, no grad accum, ~12 chained 4h jobs)
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/full_attention.py
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/full_hyena.py

# Patch 8 (batch=64, accum=4 to maintain effective batch 2048, ~20 chained jobs)
bash slurm/queue.sh slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/full_attention.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
bash slurm/queue.sh slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
bash slurm/queue.sh slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
bash slurm/queue.sh slurm/submit_hybrid.sh 20 examples/vit5_imagenet/vit5_hybrid/full_hyena.py net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
```

**Batch size rationale for patch 8**: 784 tokens/image (vs 196 for patch 16). Default batch=256 will OOM on H100 80GB. The v5_patch experiments use batch=64 + accum=4 for patch 8, maintaining effective batch = 8 GPUs x 64 x 4 = 2048.

**Job chain count**: Patch-16 FLOPs ~9-10 GFLOPs -> 12 jobs is sufficient. Patch-8 FLOPs ~38-45 GFLOPs with 4x grad accum -> 20 jobs. Extra jobs are harmless (autoresume exits immediately if training is complete).

---

## Risk: Container may not have hybrid code

The code is baked into the container at `/workspaces/nvSubquadratic-private`. The new sqsh was built on 04-17-2026 — we need to verify it includes the `vit5_hybrid` configs (which are on the `dwromero/hybrid-vit5` branch). If it doesn't, training will fail at import time.

**Workaround**: Mount the local repo into the container by adding to the mounts in `submit_hybrid.sh`:
```
${WORKDIR}:/workspaces/nvSubquadratic-private
```

---

## Test Job

Single patch-8 test job submitted as validation before full launch:
```bash
sbatch slurm/submit_hybrid.sh examples/vit5_imagenet/vit5_hybrid/full_attention.py \
    net.patch_size=8 dataset.batch_size=64 train.accumulate_grad_steps=4
```

## Verification Checklist

1. Submit a single patch-8 job, verify no OOM and correct batch config
2. Submit a single patch-16 job, check slurm output for successful training start
3. Check W&B for the run appearing under `vit5_hybrid` job group
4. If both pass, launch the full set of 8 chained experiments
