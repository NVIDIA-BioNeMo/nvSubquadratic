# Chain extension plan — 2026-04-23

## Goal
1. Extend the two ongoing 4-node hybrid-KAN training chains by 12 more jobs each so they keep autoresuming past their current tails.
2. Launch a new 16-job chain for a 3-layer KAN variant (shape `2 → 32 → 32 → out_dim`).

## Configs in flight
Both chains submitted via `slurm/submit_hybrid_kan_4node.sh` with identical overrides: `dataset.batch_size=64 train.accumulate_grad_steps=1` (patch size 8 default, effective batch = 2048 across 32 GPUs).

| Experiment | Config | Run dir | Original chain tail |
| --- | --- | --- | --- |
| `full_hyena_kan` | `examples/vit5_imagenet/vit5_hybrid/full_hyena_kan.py` | `runs/full_hyena_kan/run_b971eee2` | 27243516 (pending, deps on 27243515 running) |
| `full_hyena_kan_mlp_eq_out` | `examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_mlp_eq_out.py` | `runs/full_hyena_kan_mlp_eq_out/run_8701f745` | 27245531 (pending, deps on 27245526 running) |

Verified mapping by reproducing `RUN_NAME_HASH` (md5 of `config + overrides + experiment_name`, first 8 chars) — both hashes matched the existing `run_b971eee2` / `run_8701f745` dirs, so the deterministic run-name check-in guarantees the new jobs autoresume into the same checkpoint dir.

## Submission mechanism
Used the existing helper `slurm/queue.sh <submit_script> <num_jobs> <config> [overrides...] [start_after_jid=<jid>]`. It submits a linear `--dependency=afterany:<prev>` chain off the specified tail job.

Wrapped the calls in `bash -c 'source ~/.bashrc && ...'` so `WANDB_API_KEY` (exported at line 41 of `~/.bashrc`) is present in the env sbatch captures with `--export=ALL`. Non-interactive Bash-tool shells do not inherit it otherwise, and the submit script relies on it being forwarded into the container — missing key was the root cause of the 27195650–27195655 chain failing at `wandb.init` on 2026-04-22.

## Commands run
```bash
bash -c '
source ~/.bashrc
cd /lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_bionemo/amoradzadeh/hyena/vit5_nvsubq

bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 12 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1 \
    start_after_jid=27243516

bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 12 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_mlp_eq_out.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1 \
    start_after_jid=27245531
'
```

## Result
24 jobs submitted, all pending with chained `afterany` dependencies.

| Chain | New job IDs | Depends on (head) |
| --- | --- | --- |
| `full_hyena_kan` (+12) | 27259809 … 27259820 | 27259809 → 27243516 |
| `full_hyena_kan_mlp_eq_out` (+12) | 27259821 … 27259833 (skips 27259831) | 27259821 → 27245531 |

Total queue for the user after the extension submission: 32 jobs (2 running + 30 pending).

## New 3-layer KAN chain
Created `examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_3layer.py` (clone of `full_hyena_kan.py` with `KAN_NUM_LAYERS = 3`). With `data_dim=2` and `mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM=32`, the KAN layer chain is `2 → 32 → 32 → out_dim` (out_dim = `${net.hidden_dim}`). Chose a new `.py` file over CLI overrides because the existing codebase keeps one config file per variant (e.g. `full_hyena_kan_mlp_eq_out.py` is a one-parameter sibling of `full_hyena_kan.py`).

Submitted 16 chained jobs (fresh chain, no `start_after_jid`):
```bash
bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 16 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_3layer.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1
```

| Chain | Run dir | Job IDs | Head deps |
| --- | --- | --- | --- |
| `full_hyena_kan_3layer` | `runs/full_hyena_kan_3layer/run_b2a8e258` | 27259923 … 27259938 (16) | 27259923 has no dep (fresh) |

Run-name hash `b2a8e258` pre-computed from the md5 recipe to confirm the expected checkpoint directory before launch.

Total queue after all three submissions: **47 jobs** (2 running + 45 pending).

## Monitoring
```bash
squeue -u $USER --format="%.12i %.30j %.10T %.10M %.20R %E"
# last-in-chain watch:
squeue -j 27259820   # full_hyena_kan tail
squeue -j 27259833   # full_hyena_kan_mlp_eq_out tail
squeue -j 27259938   # full_hyena_kan_3layer tail
```

Chain logs auto-append to `runs/<experiment>/<run>/job_chain.log` as each job starts/finishes.

## 2026-04-24 — 3-layer chain re-extension
The `full_hyena_kan_3layer` chain (run_b2a8e258) finished its 16-job run when tail 27271886 COMPLETED at 17:09 PDT today, leaving zero queued jobs for that experiment. The other two plan chains were checked at the same time:
- `full_hyena_kan_mlp_eq_out`: still active (27271881 running + 27271882 pending).
- `full_hyena_kan`: tail 27272681 was CANCELLED at 11:12 PDT; replaced shortly after by a new `full_hyena_kan_siren` chain (run_7985e5bb), 27273980 onward — out of scope for this plan.

Submitted 16 more jobs onto `full_hyena_kan_3layer` with no `start_after_jid` (chain was empty, so first job has no dep and runs immediately; autoresume re-uses the existing `run_b2a8e258` checkpoint dir):

```bash
bash -c 'source ~/.bashrc && \
  bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 16 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_3layer.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1'
```

| Chain | New job IDs | Head state |
| --- | --- | --- |
| `full_hyena_kan_3layer` (+16) | 27282827 … 27282843 (skips 27282831) | 27282827 RUNNING (no dep) |

New tail to watch: `squeue -j 27282843`.

## 2026-04-25 — End-of-day chain status check
All four hybrid-KAN chains had emptied their queues by 15:23 PDT today. Reached out to each chain's most recent error log to distinguish "training finished cleanly (max_steps=500000)" from "chain stopped early":

| Chain | Final state | Best `val/acc_ema` | At step |
| --- | --- | --- | --- |
| `full_hyena_kan` (run_b971eee2) | FINISHED — `Trainer.fit stopped: max_steps=500000` (job 27272150 on 2026-04-24 10:57 PDT). 27272681 was a CANCELLED post-finish restart, not a training failure. | **0.82788** | 480000 |
| `full_hyena_kan_siren` (run_7985e5bb) | FINISHED — max_steps=500000 (job 27274414 on 2026-04-25 15:23 PDT). | 0.82148 | 460000 |
| `full_hyena_kan_3layer` (run_b2a8e258) | FINISHED — max_steps=500000 (job 27282843 on 2026-04-25 10:05 PDT). | 0.81972 | 455000 |
| `full_hyena_kan_mlp_eq_out` (run_8701f745) | NOT FINISHED — last successful checkpoint at step 155000 (job 27271881, 2026-04-24 20:59 PDT); follow-up 27271882 was killed (SIGTERM, exit 15:0) mid-`wandb` artifact download at 21:13 PDT and the chain stopped. | 0.68172 (so far) | 155000 |

**Decision:** the three finished chains stay as-is (no resubmit — they hit the configured target). `full_hyena_kan_mlp_eq_out` is incomplete with ~345k steps remaining; resubmission was prepared (12 more jobs via `slurm/queue.sh`) but the launch was cancelled by the user before sending — leaving that chain idle pending further direction.

## 2026-04-25 — New experiment: full_hyena_kan @ patch_size=8, grid_range=[-1, 1]
Goal: evaluate the same `full_hyena_kan` model under two simultaneous changes vs. the existing finished `full_hyena_kan` chain (which actually ran at `patch_size=16` — verified by `patch_size: 16` lines in `runs/full_hyena_kan/run_b971eee2`'s slurm log; the earlier "patch size 8 default" remark in this plan was incorrect, the prior chain only inherited the patch-8 batch-recipe overrides).

### Knob-level diff vs prior `full_hyena_kan` (run_b971eee2)
- `patch_size`: 16 → 8. Sequence length grows 196 → 784 (4×). `_GRID_H` (KAN `L_cache`) auto-adjusts 14 → 28.
- `grid_range`: `[-5.0, 5.0]` → `[-1.0, 1.0]`. `KANKernelND` computes `grid_num = int((high − low) · L_cache)`, so `grid_num` becomes 2·28 = **56** (vs. 10·14 = 140 before) — fewer B-spline knots, smaller KAN params, coarser spline resolution. Inputs are unaffected: `grid_cache` is always `linspace(-1, 1, 2·L_cache − 1)`, so the entire input domain still falls inside the (new, tighter) knot range.

### Config file
Created `examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_p8_grid1.py` (clone of `full_hyena_kan.py`) — encodes both knobs in-file rather than via CLI overrides, mirroring how `siren` / `3layer` / `mlp_eq_out` siblings each pin their one variable:
```python
PATCH_SIZE = 8                             # passed to build_hybrid_net(...)
kernel_cfg = LazyConfig(KANKernelND)(
    ...
    grid_range=[-1.0, 1.0],
)
```

### Submission
Pre-computed `RUN_NAME_HASH` from md5(`config + overrides + experiment_name`)[:8] → `da3bd599`; verified `runs/full_hyena_kan_p8_grid1/` did not exist (fresh run, no resume).

```bash
bash -c 'source ~/.bashrc && \
  bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 20 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_p8_grid1.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1'
```

20 jobs (vs. the 16 used for prior chains) to budget for the slower per-step time at patch_size=8 (longer sequence partly offset by cheaper KAN).

| Chain | Run dir | Job IDs | Head state |
| --- | --- | --- | --- |
| `full_hyena_kan_p8_grid1` | `runs/full_hyena_kan_p8_grid1/run_da3bd599` | 27302486 … 27302505 (20) | 27302486 PENDING, no dep (fresh) |

New tail to watch: `squeue -j 27302505`.

### Sanity check after first job starts
Once 27302486 begins, confirm in its slurm log:
- `patch_size: 8` (4× the prior chain).
- `grid_num: 56` for KAN layers (was 140).
- training step counter actually advances (not stuck on data prep).

Compare final `val/acc_ema` against the leaderboard:
- patch=16, grid=[-5,5]: **0.828** (`full_hyena_kan`)
- patch=16, grid=[-1,1] + Sine: 0.821 (`full_hyena_kan_siren`)
- patch=16, grid=[-5,5], 3-layer KAN: 0.820 (`full_hyena_kan_3layer`)

**Caveat (already flagged in planning):** patch_size and grid_range move together in this run — if accuracy regresses we won't know which knob is responsible without an ablation. A follow-up isolating one knob may be needed depending on the result.

## 2026-04-25 — `full_hyena_kan_p8_grid1` chain crashed and was cancelled
Job 27302486 (head of the grid1 chain) ran ~1h28m, completed the first 5000 training steps and the first validation pass (`val/acc_ema=0.00232` — essentially random, expected at step 5k on ImageNet-1k), then died with exit code 137 (NCCL ABORT → SIGKILL) immediately after saving `epoch=7-step=5000.ckpt`:

```
[rank4]: ProcessGroupNCCL watchdog thread terminated with exception:
         CUDA error: an illegal memory access was encountered (cudaErrorIllegalAddress)
```

Only rank 4 faulted; the other 31 ranks were torn down by the NCCL watchdog. Job 27302487 auto-resumed at 21:52, was mid-recompile/warmup (14 min in, no training steps yet) when cancelled. Jobs 27302488–27302505 cancelled while pending.

**Suspected cause:** `cudaErrorIllegalAddress` is consistent with an out-of-bounds index into a tensor. The most likely culprit given the config diff is the tighter `grid_range=[-1, 1]`: `KANKernelND` indexes B-spline basis using inputs against the knot grid; if any input falls outside the knot range the spline-evaluation kernel can index past the buffer. Surviving 5000 steps and only failing at the first val/checkpoint boundary is also consistent with a rare edge-case batch rather than a deterministic bug. A transient hardware fault on `batch-block7-02875` (rank-0 node) cannot be fully ruled out from one log.

## 2026-04-25 — Resubmit with `grid_range=[-2, 2]`
Decided to widen the grid rather than re-attempt `[-1, 1]`. New config file `examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_p8_grid2.py` (clone of `_p8_grid1.py` with `grid_range=[-2.0, 2.0]`). With `L_cache=_GRID_H=28` at patch_size=8, `grid_num = int((high − low) · L_cache) = 4 · 28 = 112` (vs. 56 for grid1, 140 for the patch=16 baseline) — wider knot range than grid1 should keep all inputs inside the spline support if the rank-4 fault was indeed input-out-of-range.

Submitted 20 chained jobs (fresh chain, no `start_after_jid`):

```bash
bash -c 'source ~/.bashrc && \
  bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 20 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_p8_grid2.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1'
```

| Chain | Job IDs | Head state |
| --- | --- | --- |
| `full_hyena_kan_p8_grid2` | 27303711 … 27303730 (20) | 27303711 PENDING (Resources), no dep (fresh) |

New tail to watch: `squeue -j 27303730`. Run dir will appear at `runs/full_hyena_kan_p8_grid2/run_<hash>/` once the head job starts (hash not pre-computed this round).

### Sanity check after first job starts
- `patch_size: 8` and `grid_num: 112` for KAN layers in the slurm log.
- Training advances past step 5000 (the previous fault point) without an `illegal memory access`.
- If it dies at the same step on the same node, suspect hardware on `batch-block7-02875` and consider node exclusion. If it dies at step 5k on a different node, the input-out-of-range hypothesis is much stronger and we'd want to clamp inputs in `KANKernelND` rather than chase the grid wider.

The `_p8_grid1` config and orphaned `runs/full_hyena_kan_p8_grid1/run_da3bd599` checkpoint dir were left in place (not deleted) for reference / potential post-mortem reproduction.

## 2026-04-26 — `full_hyena_kan_p8_grid2` chain stalled on corrupt W&B autoresume artifact
The grid2 chain ran cleanly through jobs 27303711 and 27303712 (8h total, last clean checkpoint at step 55000, run dir `runs/full_hyena_kan_p8_grid2/run_02e72b20`, W&B run id `rQKWt3uz`). Then jobs 27303713 through 27303720 each died <2 min after start with exit 143:

```
RuntimeError: PytorchStreamReader failed reading file data/104:
              invalid header or archive is corrupted
```

Stack trace was inside `torch.load()` from `pytorch_lightning…checkpoint_connector.resume_start` — i.e. the autoresume checkpoint, not a runtime CUDA fault (so this incident is *unrelated* to the grid1 `cudaErrorIllegalAddress` from the day before).

### Root cause
`experiments/run.py:213-237` autoresume tries W&B *first* and only falls back to local `last.ckpt` if W&B fails. `experiments/utils/checkpointing.py:80-84` calls `wandb.Api().artifact().download(root=".artifacts/{run_id}/{alias}")` — which is **idempotent on manifest digest**: once the cache file exists, every subsequent restart reuses the on-disk copy without re-downloading. So when job 27303713 wrote a corrupted blob to `${WORKDIR}/.artifacts/rQKWt3uz/latest/last.ckpt` (647226205 B — note: smaller than the local `runs/.../checkpoints/last.ckpt` at 647226269 B), every later restart hit the same corrupted file and crashed identically. Local `runs/.../checkpoints/last.ckpt` was intact the whole time but never reached.

### Fix
```bash
rm -rf /lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_bionemo/amoradzadeh/hyena/vit5_nvsubq/.artifacts/rQKWt3uz/latest
scontrol release 27303721
```

Only the `rQKWt3uz/latest/` cache for this specific run was deleted; the four sibling caches (`2J1kGDlf, LxZl27M3, snwFovxX, vzm9oYMg` — other chains) were left alone.

### Outcome
- Jobs **27303721, 27303722** still hit the same `PytorchStreamReader` error after the cache wipe — the W&B download itself is non-deterministic and returned a corrupt blob on those two attempts. Lost ~3 more minutes per job.
- Job **27303723** (started 09:53 PDT) re-downloaded a clean blob, autoresume succeeded, training resumed at step ~55000 with `train/loss≈3.4–4.8` and `train/acc≈0.33–0.40` (i.e. weights loaded — random init would be loss ~6.9, acc ~0.001). Currently RUNNING and progressing past step 130 of the resumed job.

10 jobs (27303713–27303722) were burned by the corruption (~20 min wallclock × 4 nodes wasted) — but no training-step progress was lost since the last successful checkpoint at step 55000 was on disk and on W&B.

### Chain extension (+8)
Submitted 8 more jobs to keep the chain autoresuming past the original 20-job tail:
```bash
bash -c 'source ~/.bashrc && \
  bash slurm/queue.sh slurm/submit_hybrid_kan_4node.sh 8 \
    examples/vit5_imagenet/vit5_hybrid/full_hyena_kan_p8_grid2.py \
    dataset.batch_size=64 train.accumulate_grad_steps=1 \
    start_after_jid=27303730'
```

| Chain | New job IDs | Depends on |
| --- | --- | --- |
| `full_hyena_kan_p8_grid2` (+8) | 27312961 … 27312966, 27312983, 27312984 | 27312961 → 27303730 |

New tail to watch: `squeue -j 27312984`. Total queued for the chain after this extension: 16 jobs (1 running 27303723 + 15 pending).

### Followups if it happens again
- The W&B download flake on 721/722 right after a fresh cache delete suggests this is a real (rare) corruption mode on either the upload or CDN side. If a future chain stalls in the same way, the same fix applies — `rm -rf .artifacts/<run_id>/<alias>/` and let the next job re-download. Three attempts have been sufficient so far.
- A more robust fix would be to extend `download_checkpoint()` in `experiments/utils/checkpointing.py` to verify the downloaded file is a valid zip archive (e.g. `zipfile.ZipFile(...).testzip()`) before returning, retrying on failure and finally falling back to local `last.ckpt`. Not done — out of scope for today.
