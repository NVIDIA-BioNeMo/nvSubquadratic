# Plan: Fresh TRACKER Sweep — Compile Disabled (2026-04-28)

## Context

Earlier `vit5_hybrid` runs at patch 8/16 (Phase 1 chains, omega ablation, etc.)
all used `compile=True` with `compile_mode=max-autotune-no-cudagraphs` or
`default`. In recent re-run attempts, post-autoresume DDP jobs have shown a
recurring NCCL desync — workers hang at low step count, watchdog fires SIGABRT
after 30–60 min. The hang reproduces across `max-autotune-no-cudagraphs`,
`default`, and `reduce-overhead` compile modes; extending the per-op timeout to
1 h delays but does not prevent it (see `plan.old.md` for the full failure
log).

This plan starts a **fresh** TRACKER sweep with `compile=False` to remove
torch.compile from the variable space. The TRACKER tables for Patch 16, 8, 4
are currently empty — this sweep will populate them.

## Recipe (common to all runs)

- v5 recipe: 800 epochs, LAMB lr=4e-3, wd=0.05, cosine, 3-Augment, Mixup/CutMix,
  EMA 0.99996.
- Effective batch size = **2048**.
- `compile=False` (overrides `config.compile=True` in the leaf `.py` files;
  `compile_mode` is then ignored at runtime — see `experiments/run.py:164-169`).
- Hardware: **1 node × 8 H100 GPUs**, `slurm/submit_hybrid.sh`.
- 4 h walltime per job, chained restarts via `slurm/queue.sh`.
- Account: `healthcareeng_research` (script default — do not auto-switch).

| Patch | tokens/img | per-GPU batch | accum | num GPUs | effective |
| ----- | ---------- | ------------- | ----- | -------- | --------- |
| 16    | 196        | 256           | 1     | 8        | 2048      |
| 8     | 784        | 256           | 1     | 8        | 2048      |
| 4     | 3 136      | 128           | 2     | 8        | 2048      |

## Configs (4 base configs in `examples/vit5_imagenet/vit5_hybrid/`)

| Config              | Pattern    | Hyena:Attn |
| ------------------- | ---------- | ---------- |
| `full_attention.py` | `A×12`     | 0:12       |
| `hybrid_ha.py`      | `(HA)×6`   | 6:6        |
| `hybrid_hhha.py`    | `(HHHA)×3` | 9:3        |
| `full_hyena.py`     | `H×12`     | 12:0       |

KAN (`full_hyena_kan_p8_grid2.py`) is **not** in this sweep — TRACKER scope is
the 4-config × 3-patch grid only.

## Launch Commands

From repo root `/lustre/fsw/healthcareeng_bionemo/amoradzadeh/hyena/vit5_multinode/`.

`slurm/submit_hybrid.sh:45` already injects
`compile_mode=max-autotune-no-cudagraphs` as a CLI override; the value is
ignored at runtime when `compile=False`. We pass `compile=False` as an extra
positional arg via `queue.sh`, which appends it to `CONFIG_OVERRIDES` (later
overrides win).

### Patch 16 (4 chains × 12 windows)

```bash
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/full_attention.py compile=False net.patch_size=16 dataset.batch_size=256 train.accumulate_grad_steps=1
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py      compile=False net.patch_size=16 dataset.batch_size=256 train.accumulate_grad_steps=1
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py    compile=False net.patch_size=16 dataset.batch_size=256 train.accumulate_grad_steps=1
bash slurm/queue.sh slurm/submit_hybrid.sh 12 examples/vit5_imagenet/vit5_hybrid/full_hyena.py     compile=False net.patch_size=16 dataset.batch_size=256 train.accumulate_grad_steps=1
```

### Patch 8 (4 chains × 16 windows)

```bash
bash slurm/queue.sh slurm/submit_hybrid.sh 16 examples/vit5_imagenet/vit5_hybrid/full_attention.py compile=False net.patch_size=8 dataset.batch_size=256 train.accumulate_grad_steps=1
bash slurm/queue.sh slurm/submit_hybrid.sh 16 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py      compile=False net.patch_size=8 dataset.batch_size=256 train.accumulate_grad_steps=1
bash slurm/queue.sh slurm/submit_hybrid.sh 16 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py    compile=False net.patch_size=8 dataset.batch_size=256 train.accumulate_grad_steps=1
bash slurm/queue.sh slurm/submit_hybrid.sh 16 examples/vit5_imagenet/vit5_hybrid/full_hyena.py     compile=False net.patch_size=8 dataset.batch_size=256 train.accumulate_grad_steps=1
```

### Patch 4 (4 chains × 24 windows)

```bash
bash slurm/queue.sh slurm/submit_hybrid.sh 24 examples/vit5_imagenet/vit5_hybrid/full_attention.py compile=False net.patch_size=4 dataset.batch_size=128 train.accumulate_grad_steps=2
bash slurm/queue.sh slurm/submit_hybrid.sh 24 examples/vit5_imagenet/vit5_hybrid/hybrid_ha.py      compile=False net.patch_size=4 dataset.batch_size=128 train.accumulate_grad_steps=2
bash slurm/queue.sh slurm/submit_hybrid.sh 24 examples/vit5_imagenet/vit5_hybrid/hybrid_hhha.py    compile=False net.patch_size=4 dataset.batch_size=128 train.accumulate_grad_steps=2
bash slurm/queue.sh slurm/submit_hybrid.sh 24 examples/vit5_imagenet/vit5_hybrid/full_hyena.py     compile=False net.patch_size=4 dataset.batch_size=128 train.accumulate_grad_steps=2
```

## Risks

1. **Throughput hit.** Without compile, expect ~1.3–1.7 it/s vs ~2.85 it/s with
   `max-autotune-no-cudagraphs` (1-node baseline). 800 epochs × 156 it/epoch ÷
   1.5 it/s ≈ 23 h pure training; with restart overhead, plan ~6–8 chained 4 h
   windows for patch 8/16 and ~10–12 for patch 4 (accum=2 is ~2× slower per
   epoch).

2. **Patch 8 + batch=256 OOM risk.** `plan.old.md` noted "Default batch=256 will
   OOM on H100 80GB" at patch 8 *with* compile. Without compile, memory savings
   from kernel fusion are lost — OOM is more likely. Sanity-test one patch-8
   job per config before fanning out the chain. Fallback:
   `dataset.batch_size=64 train.accumulate_grad_steps=4` (still effective 2048).

3. **Patch 4 + batch=128 OOM risk.** 3 136 tokens/image × batch 128 = 401 k
   tokens per GPU. Earlier patch-4 experiments used batch=16 + accum=4 on
   4-node — much smaller per-GPU memory footprint. If OOM at p4, fall back to
   `dataset.batch_size=64 train.accumulate_grad_steps=4` (effective 2048).

4. **Fresh run dirs.** `RUN_NAME_HASH = md5(CONFIG_FILE + CONFIG_OVERRIDES +
   EXPERIMENT_NAME)[:8]` includes the new `compile=False` token, so each run
   creates its own `runs/<config>/run_<hash>/` dir — no collision with prior
   compile=True chains; nothing is overwritten.

5. **In-flight chains.** The currently running compile=True chains
   (`5161468` hhha old max-autotune, `5165199` KAN, `5165243` hhha new default)
   are unrelated to this fresh sweep and will continue independently. Cancel
   manually if you don't want them consuming compute alongside this sweep.

## Verification Checklist

- [ ] Smoke: submit one **patch-16** job for `hybrid_hhha.py` (lowest-risk
      shape); confirm it reaches step 100 without OOM or NCCL hang.
- [ ] Smoke: submit one **patch-8** job per config (highest OOM risk:
      `compile=False`+`batch=256`+`784 tokens`); confirm step 100 without OOM.
- [ ] Smoke: submit one **patch-4** job; confirm `accum=2` keeps memory under
      80 GB at step 100.
- [ ] After the first 4 h window of each chain (TIMEOUT or COMPLETED via
      WalltimeCheckpointer), confirm autoresume from `last.ckpt` works in the
      next chained job.
- [ ] Once each of the 12 runs reaches 800 epochs, populate `val/acc_ema` and
      `test/acc` in `TRACKER.md`. Update its `it/s (1 GPU)` column with the
      effective imgs/s ÷ 8 GPUs from the steady-state log line.

## Files

- `examples/vit5_imagenet/vit5_hybrid/plan.md` — this file (new).
- `examples/vit5_imagenet/vit5_hybrid/plan.old.md` — previous plan, archived for
  reference (compile=True history, omega ablation, NCCL-desync postmortems).
- `examples/vit5_imagenet/vit5_hybrid/TRACKER.md` — result tables, populated as
  runs complete.
- No code changes required; `compile=False` is a runtime CLI override.
