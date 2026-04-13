"""Profile the ARC training pipeline to identify the wall-clock bottleneck.

Measures data loading, forward, backward, and optimizer step independently
to show where time is actually spent in the training loop.

Usage (default ViT config):
    PYTHONPATH=. python benchmarks/arc/profile_training_bottleneck.py

Usage (eager mode, no compile):
    PYTHONPATH=. python benchmarks/arc/profile_training_bottleneck.py --eager

Usage (Hyena config):
    PYTHONPATH=. python benchmarks/arc/profile_training_bottleneck.py \\
        --config examples/arc/cfg_hyena_rearc.py

Usage (skip RE-ARC for faster dataset setup):
    PYTHONPATH=. python benchmarks/arc/profile_training_bottleneck.py --no-rearc
"""

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from nvsubquadratic.lazy_config import instantiate


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_config(config_path: str):
    """Dynamically load a config module and return its config object."""
    spec = importlib.util.spec_from_file_location("_arc_cfg", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def build_datamodule(cfg, no_rearc: bool, num_workers: int):
    """Instantiate ARCDataModule from config, optionally disabling RE-ARC."""
    from omegaconf import OmegaConf

    dataset_cfg = cfg.dataset
    # Override num_workers and optionally strip rearc_dir
    overrides = {"num_workers": num_workers}
    if no_rearc:
        overrides["rearc_dir"] = None
    dataset_cfg = OmegaConf.merge(dataset_cfg, overrides)
    dm = instantiate(dataset_cfg)
    dm.setup("fit")
    return dm


def make_dataloader(dm, num_workers: int, prefetch_factor: int):
    """Rebuild the training DataLoader with given num_workers + prefetch_factor."""
    from torch.utils.data import DataLoader

    from experiments.datamodules.arc import _collate_fn

    return DataLoader(
        dm.train_dataset,
        batch_size=dm.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dm.pin_memory,
        drop_last=True,
        collate_fn=_collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def make_fake_batch(batch_size: int, max_size: int, num_tasks: int, device: torch.device):
    """Build a synthetic batch matching the ARC training batch format."""
    return {
        "input": torch.randint(0, 10, (batch_size, max_size, max_size), device=device),
        "label": torch.randint(0, 12, (batch_size, max_size, max_size), device=device),
        "condition": {
            "task_id": torch.randint(0, num_tasks, (batch_size,), device=device),
            "attention_mask": torch.ones(batch_size, max_size, max_size, device=device, dtype=torch.long),
        },
    }


def bench_dataloader(dm, num_workers: int, prefetch_factor: int, num_steps: int, warmup: int = 5):
    """Benchmark pure data loading (no GPU compute) for one (workers, prefetch) combo."""
    loader = make_dataloader(dm, num_workers=num_workers, prefetch_factor=prefetch_factor)
    it = iter(loader)
    for _ in range(warmup):
        next(it)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_steps):
        next(it)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del loader, it
    ms = elapsed / num_steps * 1000
    tput = dm.batch_size * num_steps / elapsed
    return ms, tput


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    device = torch.device("cuda")
    mode_str = "eager" if args.eager else "compiled"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_name = torch.cuda.get_device_name()
    node = os.environ.get("SLURMD_NODENAME", "unknown")

    # ── Load config ──────────────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)

    # Pull key hyperparams from config
    batch_size = cfg.dataset.batch_size
    max_size = cfg.dataset.max_size
    # num_tasks: read from the network config
    num_tasks = cfg.net.num_tasks

    print(f"Device:     {gpu_name} (node={node})")
    print(f"Config:     {args.config}")
    print(f"Mode:       {mode_str.upper()}")
    print(f"Batch size: {batch_size}  |  Max size: {max_size}  |  Num tasks: {num_tasks}")
    print(f"Workers:    {args.num_workers}  |  Steps: {args.num_steps}")
    print(f"RE-ARC:     {'disabled' if args.no_rearc else 'enabled'}")
    print(f"Timestamp:  {timestamp}")
    print()

    # ── Setup DataModule ─────────────────────────────────────────────────────
    print("Setting up DataModule...")
    # Build with default prefetch; we'll rebuild per sweep below
    dm = build_datamodule(cfg, no_rearc=args.no_rearc, num_workers=args.num_workers)
    n_train = len(dm.train_dataset)
    print(f"Train dataset: {n_train:,} examples")
    print()

    # ── Phase 0: Prefetch factor sweep ───────────────────────────────────────
    print("=" * 60)
    print("PHASE 0: Prefetch factor sweep (data loading only)")
    print("=" * 60)
    prefetch_factors = [2, 4, 8, 16]
    pf_results = {}
    best_pf, best_ms = 2, float("inf")
    for pf in prefetch_factors:
        ms, tput = bench_dataloader(dm, num_workers=args.num_workers, prefetch_factor=pf, num_steps=args.num_steps)
        pf_results[pf] = {"ms": round(ms, 1), "tput": round(tput)}
        print(f"  prefetch_factor={pf:>2d}: {ms:6.1f}ms/batch, {tput:>6.0f} samples/sec")
        if ms < best_ms:
            best_ms, best_pf = ms, pf
    print(f"  >> Best: prefetch_factor={best_pf} ({best_ms:.1f}ms)")
    print()

    # ── Phase 1: Data loading benchmark ─────────────────────────────────────
    print("=" * 60)
    print(f"PHASE 1: Data loading speed (prefetch_factor={best_pf})")
    print("=" * 60)
    data_ms, data_tput = bench_dataloader(
        dm, num_workers=args.num_workers, prefetch_factor=best_pf, num_steps=args.num_steps, warmup=5
    )
    print(f"  Per batch:  {data_ms:.1f}ms")
    print(f"  Throughput: {data_tput:.0f} samples/sec")
    print()

    # ── Phase 2: Forward + backward, synthetic batch ─────────────────────────
    print("=" * 60)
    print(f"PHASE 2: Forward + backward — synthetic batch (mode={mode_str})")
    print("=" * 60)
    net = instantiate(cfg.net).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_params:,}")

    if not args.eager:
        if getattr(cfg, "compile_compatible_fftconv", False):
            import nvsubquadratic.ops.fftconv as _fftconv

            _fftconv.COMPILE_COMPATIBLE = True
            print("  Enabled compile-compatible FFT convolution")
        compile_mode = getattr(cfg, "compile_mode", "default") or "default"
        print(f"  Compiling with mode={compile_mode!r}...")
        net = torch.compile(net, mode=compile_mode)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=10)
    fake_batch = make_fake_batch(batch_size, max_size, num_tasks, device)
    # Keep a copy of the label for the loss fn (label is not popped in synthetic path)
    fake_label = fake_batch["label"].clone()

    warmup_iters = 5 if args.eager else 20
    print(f"  Warming up ({warmup_iters} iters)...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(warmup_iters):
            out = net(fake_batch)["logits"]
            loss = loss_fn(out, fake_label)
            loss.backward()
            net.zero_grad()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.num_steps):
            out = net(fake_batch)["logits"]
            loss = loss_fn(out, fake_label)
            loss.backward()
            net.zero_grad()
    end_ev.record()
    torch.cuda.synchronize()

    compute_ms_total = start_ev.elapsed_time(end_ev)
    compute_ms = compute_ms_total / args.num_steps
    compute_tput = batch_size * args.num_steps / (compute_ms_total / 1000)
    print(f"  Per step:   {compute_ms:.1f}ms")
    print(f"  Throughput: {compute_tput:.0f} samples/sec")
    print()

    # ── Phase 3: Optimizer step ──────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 3: Optimizer step (AdamW)")
    print("=" * 60)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.0)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(fake_batch)["logits"]
        loss = loss_fn(out, fake_label)
        loss.backward()
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(args.num_steps):
        optimizer.step()
    end_ev.record()
    torch.cuda.synchronize()

    optim_ms = start_ev.elapsed_time(end_ev) / args.num_steps
    print(f"  Per step: {optim_ms:.1f}ms")
    print()

    # ── Phase 4: Full training step with component breakdown ─────────────────
    print("=" * 60)
    print("PHASE 4: Full training step — component breakdown (real data)")
    print("=" * 60)
    loader = make_dataloader(dm, num_workers=args.num_workers, prefetch_factor=best_pf)
    it = iter(loader)

    SKIP = 10
    components = [
        "dataloader_fetch",
        "to_gpu",
        "batch_transfer",
        "forward_loss",
        "backward",
        "grad_clip",
        "optimizer_step",
        "zero_grad",
        "total_iter",
    ]
    timings = {k: [] for k in components}

    print(f"  Warming up ({SKIP} iters)...")
    optimizer.zero_grad()
    for _ in range(SKIP):
        raw = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
        batch = dm.on_before_batch_transfer(batch, 0)
        label = batch.pop("label")
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = net(batch)["logits"]
            loss = loss_fn(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    print("  Measuring...")

    for i in range(args.num_steps):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        # 1. DataLoader fetch
        t0 = time.perf_counter()
        raw = next(it)
        t1 = time.perf_counter()

        # 2. Move to GPU
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # 3. on_before_batch_transfer (dict rearrangement)
        t4 = time.perf_counter()
        batch = dm.on_before_batch_transfer(batch, 0)
        label = batch.pop("label")
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        # 4. Forward + loss
        t6 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = net(batch)["logits"]
            loss = loss_fn(logits, label)
        torch.cuda.synchronize()
        t7 = time.perf_counter()

        # 5. Backward
        t8 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t9 = time.perf_counter()

        # 6. Grad clip
        t10 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        torch.cuda.synchronize()
        t11 = time.perf_counter()

        # 7. Optimizer step
        t12 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t13 = time.perf_counter()

        # 8. Zero grad
        t14 = time.perf_counter()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t15 = time.perf_counter()

        t_end = time.perf_counter()

        timings["dataloader_fetch"].append((t1 - t0) * 1000)
        timings["to_gpu"].append((t3 - t2) * 1000)
        timings["batch_transfer"].append((t5 - t4) * 1000)
        timings["forward_loss"].append((t7 - t6) * 1000)
        timings["backward"].append((t9 - t8) * 1000)
        timings["grad_clip"].append((t11 - t10) * 1000)
        timings["optimizer_step"].append((t13 - t12) * 1000)
        timings["zero_grad"].append((t15 - t14) * 1000)
        timings["total_iter"].append((t_end - t_start) * 1000)

    print()
    component_stats = {}
    sum_components = 0.0
    for name, vals in timings.items():
        arr = np.array(vals)
        mean, std, p95 = arr.mean(), arr.std(), np.percentile(arr, 95)
        label = f"{name}:"
        component_stats[name] = {"mean": round(mean, 1), "std": round(std, 1), "p95": round(p95, 1)}
        if name != "total_iter":
            sum_components += mean
        print(f"  {label:<22s} {mean:>7.1f}ms  (std {std:>5.1f}, p95 {p95:>7.1f})")

    full_ms = np.mean(timings["total_iter"])
    full_tput = batch_size * 1000.0 / full_ms
    print(f"\n  Sum of components:     {sum_components:>7.1f}ms")
    print(f"  Sync overhead:         {full_ms - sum_components:>7.1f}ms")
    print(f"  Throughput:            {full_tput:>7.0f} samples/sec")
    print()

    # ── Summary ──────────────────────────────────────────────────────────────
    # Use phase-4 data loading as the "data" cost, phase-2 fwd+bwd as compute
    p4_data_ms = component_stats["dataloader_fetch"]["mean"] + component_stats["to_gpu"]["mean"]
    p4_fwdbwd_ms = component_stats["forward_loss"]["mean"] + component_stats["backward"]["mean"]
    p4_optim_ms = component_stats["optimizer_step"]["mean"]
    overhead_ms = full_ms - p4_data_ms - p4_fwdbwd_ms - p4_optim_ms

    print("=" * 60)
    print("SUMMARY (per step, from Phase 4 component breakdown)")
    print("=" * 60)
    print(f"  Data (fetch+to_gpu):  {p4_data_ms:7.1f}ms  ({p4_data_ms / full_ms * 100:5.1f}%)")
    print(f"  Fwd+loss+bwd:         {p4_fwdbwd_ms:7.1f}ms  ({p4_fwdbwd_ms / full_ms * 100:5.1f}%)")
    print(f"  Optimizer step:       {p4_optim_ms:7.1f}ms  ({p4_optim_ms / full_ms * 100:5.1f}%)")
    print(f"  Other:                {overhead_ms:7.1f}ms  ({overhead_ms / full_ms * 100:5.1f}%)")
    print("  " + "-" * 40)
    print(f"  Full step:            {full_ms:7.1f}ms  ({full_tput:.0f} samples/sec)")
    print()

    # Pure compute vs data comparison
    print(f"  Pure compute (Phase 2, {mode_str}): {compute_ms:.1f}ms/step")
    print(f"  Data loading (Phase 1):            {data_ms:.1f}ms/step")
    if data_ms > compute_ms:
        ratio = data_ms / compute_ms
        print(f"\n  ** DATA-BOUND: data loading is {ratio:.1f}x slower than compute **")
    else:
        ratio = compute_ms / data_ms
        print(f"\n  ** COMPUTE-BOUND: compute is {ratio:.1f}x slower than data loading **")

    # ── Write JSON ────────────────────────────────────────────────────────────
    results = {
        "timestamp": timestamp,
        "node": node,
        "gpu": gpu_name,
        "config": args.config,
        "mode": mode_str,
        "batch_size": batch_size,
        "max_size": max_size,
        "num_tasks": num_tasks,
        "num_params": n_params,
        "num_workers": args.num_workers,
        "num_steps": args.num_steps,
        "no_rearc": args.no_rearc,
        "n_train_examples": n_train,
        "prefetch_sweep": pf_results,
        "best_prefetch_factor": best_pf,
        "data_ms": round(data_ms, 1),
        "data_tput": round(data_tput),
        "compute_ms": round(compute_ms, 1),
        "compute_tput": round(compute_tput),
        "optim_ms": round(optim_ms, 1),
        "full_ms": round(full_ms, 1),
        "full_tput": round(full_tput),
        "component_stats": component_stats,
        "data_pct": round(p4_data_ms / full_ms * 100, 1),
        "compute_pct": round(p4_fwdbwd_ms / full_ms * 100, 1),
        "bound": "data" if data_ms > compute_ms else "compute",
    }

    out_dir = Path("benchmarks/arc")
    out_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / f"profile_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(results) + "\n")
    print(f"  Results appended to {jsonl_path}")

    del loader, it


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile ARC training pipeline bottleneck")
    parser.add_argument(
        "--config",
        default="examples/arc/cfg_vit_rearc.py",
        help="Path to ARC config file (default: cfg_vit_rearc.py)",
    )
    parser.add_argument("--eager", action="store_true", help="Eager mode (no torch.compile)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count")
    parser.add_argument("--num-steps", type=int, default=50, help="Steps per benchmark phase")
    parser.add_argument("--no-rearc", action="store_true", help="Skip RE-ARC data (faster setup)")
    main(parser.parse_args())
