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
