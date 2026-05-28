# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A/B comparison: baseline vs optimized WELL dataloader + training step.

Runs both configurations back-to-back on the same GPU and prints a side-by-side
comparison. Tests three optimizations:
  1. persistent_workers=True  (avoids worker restart each epoch)
  2. prefetch_factor=4 vs 2   (better I/O overlap)
  3. Direct output_fields extraction vs formatter.process_input()

Usage:
    PYTHONPATH=. python benchmarks/well/bench_ab_comparison.py \
        --config examples/well/supernova_explosion_64/cfg_vit5_attention.py \
        [--num-steps 100] [--warmup-steps 20] [--num-dl-batches 200] [--compile]
"""

import argparse
import gc
import time

import pytorch_lightning as pl
import torch
from einops import rearrange
from the_well.data import WellDataModule as BaseWellDataModule
from the_well.data.data_formatter import DefaultChannelsLastFormatter

from experiments.utils.cli import load_config_from_file
from nvsubquadratic.lazy_config import instantiate


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="A/B benchmark: baseline vs optimized WELL pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=100, help="Training steps per variant")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps to skip")
    parser.add_argument("--num-dl-batches", type=int, default=200, help="Batches for dataloader-only bench")
    parser.add_argument("--num-dl-epochs", type=int, default=3, help="Epochs for dataloader bench")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    return parser.parse_args()


# ─── Dataloader creation ────────────────────────────────────────────────────


def make_baseline_loader(well_dm, batch_size, num_workers):
    """Baseline: delegates to the_well's DataLoader (no persistent_workers, prefetch=default)."""
    return well_dm.train_dataloader()


def make_optimized_loader(well_dm, batch_size, num_workers, prefetch_factor=4):
    """Optimized: own DataLoader with persistent_workers + higher prefetch."""
    return torch.utils.data.DataLoader(
        well_dm.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


# ─── Training step variants ─────────────────────────────────────────────────


def training_step_baseline(batch, network, loss_fn, formatter, n_steps_output, amp_dtype, use_amp):
    """Baseline training step: uses formatter.process_input() to extract target."""
    input_fields = batch["input_fields"]
    ndim = input_fields.ndim
    if ndim == 5:
        model_input = rearrange(input_fields, "b t h w c -> b h w (t c)")
    elif ndim == 6:
        model_input = rearrange(input_fields, "b t d h w c -> b d h w (t c)")
    else:
        raise ValueError(f"Unexpected ndim={ndim}")
    if "constant_fields" in batch:
        model_input = torch.cat([model_input, batch["constant_fields"]], dim=-1)

    # Baseline: use formatter (does redundant rearrange + nan_to_num)
    _, y_ref = formatter.process_input(batch)
    target = y_ref[:, 0] if n_steps_output == 1 else y_ref

    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        pred = network({"input": model_input, "condition": None})["logits"]
        loss = loss_fn(pred, target)
    return loss


def training_step_optimized(batch, network, loss_fn, formatter, n_steps_output, amp_dtype, use_amp):
    """Optimized training step: extracts target directly from batch."""
    input_fields = batch["input_fields"]
    ndim = input_fields.ndim
    if ndim == 5:
        model_input = rearrange(input_fields, "b t h w c -> b h w (t c)")
    elif ndim == 6:
        model_input = rearrange(input_fields, "b t d h w c -> b d h w (t c)")
    else:
        raise ValueError(f"Unexpected ndim={ndim}")
    if "constant_fields" in batch:
        model_input = torch.cat([model_input, batch["constant_fields"]], dim=-1)

    # Optimized: direct extraction (no redundant rearrange/nan_to_num)
    y_ref = batch["output_fields"]
    target = y_ref[:, 0] if n_steps_output == 1 else y_ref

    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        pred = network({"input": model_input, "condition": None})["logits"]
        loss = loss_fn(pred, target)
    return loss


# ─── Benchmark runners ──────────────────────────────────────────────────────


def bench_dataloader(loader, num_batches, num_epochs, warmup_batches=5):
    """Benchmark dataloader throughput across epochs."""
    results = []
    for epoch in range(num_epochs):
        total_samples = 0
        t_start = None
        for i, batch in enumerate(loader):
            if i >= warmup_batches + num_batches:
                break
            if i == warmup_batches:
                t_start = time.perf_counter()
            if i >= warmup_batches:
                total_samples += batch["input_fields"].shape[0]
        elapsed = time.perf_counter() - t_start
        results.append(
            {
                "epoch": epoch,
                "elapsed_s": elapsed,
                "batches_per_sec": num_batches / elapsed,
                "samples_per_sec": total_samples / elapsed,
            }
        )
    return results


def bench_training(
    loader,
    network,
    loss_fn,
    formatter,
    n_steps_output,
    step_fn,
    optimizer,
    amp_dtype,
    use_amp,
    num_steps,
    warmup_steps,
    device,
):
    """Benchmark training step throughput."""
    step = 0
    total_steps = warmup_steps + num_steps
    step_times = []
    data_times = []
    data_start = time.perf_counter()

    for batch in loader:
        if step >= total_steps:
            break

        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data_end = time.perf_counter()

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = step_fn(batch, network, loss_fn, formatter, n_steps_output, amp_dtype, use_amp)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if step >= warmup_steps:
            step_times.append(t1 - t0)
            data_times.append(data_end - data_start)
        step += 1
        data_start = time.perf_counter()

    step_t = torch.tensor(step_times)
    data_t = torch.tensor(data_times)
    total_t = step_t + data_t
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    return {
        "data_ms": data_t.mean().item() * 1000,
        "step_ms": step_t.mean().item() * 1000,
        "total_ms": total_t.mean().item() * 1000,
        "it_per_sec": 1.0 / total_t.mean().item(),
        "peak_gpu_gb": peak_mem,
    }


def print_comparison(label, baseline, optimized):
    """Print side-by-side comparison."""
    print(f"\n{'':>24}  {'Baseline':>12}  {'Optimized':>12}  {'Speedup':>10}")
    print("-" * 65)
    for key in baseline:
        b = baseline[key]
        o = optimized[key]
        if "ms" in key or "_s" in key:
            speedup = f"{b / o:.2f}x" if o > 0 else "N/A"
            print(f"  {key:>22}  {b:12.1f}  {o:12.1f}  {speedup:>10}")
        elif "per_sec" in key or "it_" in key:
            speedup = f"{o / b:.2f}x" if b > 0 else "N/A"
            print(f"  {key:>22}  {b:12.2f}  {o:12.2f}  {speedup:>10}")
        else:
            print(f"  {key:>22}  {b:12.2f}  {o:12.2f}")


def main():
    """Benchmark WELL dataloader throughput."""
    args = parse_args()
    config = load_config_from_file(args.config)
    pl.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    batch_size = config.dataset.batch_size
    num_workers = config.dataset.num_workers
    n_steps_output = config.dataset.n_steps_output

    print("=" * 65)
    print("  WELL Pipeline A/B Benchmark")
    print("=" * 65)
    print(f"  Config:      {args.config}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Compile:     {args.compile}")
    print()

    # ─── Setup WELL datamodule (shared) ──────────────────────────────────
    from the_well.data.normalization import ZScoreNormalization

    well_dm = BaseWellDataModule(
        well_base_path=config.dataset.well_base_path,
        well_dataset_name=config.dataset.well_dataset_name,
        batch_size=batch_size,
        use_normalization=config.dataset.use_normalization,
        normalization_type=ZScoreNormalization if config.dataset.use_normalization else None,
        n_steps_input=config.dataset.n_steps_input,
        n_steps_output=n_steps_output,
        max_rollout_steps=config.dataset.max_rollout_steps,
        min_dt_stride=config.dataset.min_dt_stride,
        max_dt_stride=config.dataset.max_dt_stride,
        data_workers=num_workers,
    )
    metadata = well_dm.train_dataset.metadata
    formatter = DefaultChannelsLastFormatter(metadata)

    input_channels = config.dataset.n_steps_input * metadata.n_fields + metadata.n_constant_fields
    output_channels = metadata.n_fields

    # ─── Part 1: Dataloader throughput ───────────────────────────────────
    print("=" * 65)
    print("  Part 1: Dataloader throughput")
    print("=" * 65)

    print("\n  [Baseline] the_well DataLoader (no persistent_workers, prefetch=default)")
    baseline_loader = make_baseline_loader(well_dm, batch_size, num_workers)
    bl_dl = bench_dataloader(baseline_loader, args.num_dl_batches, args.num_dl_epochs)
    del baseline_loader
    gc.collect()

    print("  [Optimized] persistent_workers=True, prefetch_factor=4")
    opt_loader = make_optimized_loader(well_dm, batch_size, num_workers)
    opt_dl = bench_dataloader(opt_loader, args.num_dl_batches, args.num_dl_epochs)
    del opt_loader
    gc.collect()

    print(f"\n{'':>24}  {'Baseline':>12}  {'Optimized':>12}  {'Speedup':>10}")
    print("-" * 65)
    for epoch in range(args.num_dl_epochs):
        b = bl_dl[epoch]
        o = opt_dl[epoch]
        speedup = f"{o['batches_per_sec'] / b['batches_per_sec']:.2f}x"
        print(
            f"  {'Epoch ' + str(epoch):>22}"
            f"  {b['batches_per_sec']:10.1f}/s"
            f"  {o['batches_per_sec']:10.1f}/s"
            f"  {speedup:>10}"
        )

    # ─── Part 2: Training step throughput ────────────────────────────────
    print()
    print("=" * 65)
    print("  Part 2: Training step throughput (fwd + bwd + optim)")
    print("=" * 65)

    use_amp = "bf16" in config.train.precision or "16" in config.train.precision
    amp_dtype = torch.bfloat16 if "bf16" in config.train.precision else torch.float16

    def make_model_and_optim():
        net = instantiate(config.net, in_channels=input_channels, out_channels=output_channels)
        if getattr(config, "compile_compatible_fftconv", False):
            import nvsubquadratic.ops.fftconv as _fftconv

            _fftconv.COMPILE_COMPATIBLE = True
        if args.compile:
            mode = getattr(config, "compile_mode", None)
            net = torch.compile(net, **{"mode": mode} if mode else {})
        net = net.to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        return net, opt

    # Baseline
    print("\n  [Baseline] the_well DataLoader + formatter.process_input()")
    net_b, opt_b = make_model_and_optim()
    bl_loader = make_baseline_loader(well_dm, batch_size, num_workers)
    torch.cuda.reset_peak_memory_stats(device)
    bl_train = bench_training(
        bl_loader,
        net_b,
        torch.nn.MSELoss(),
        formatter,
        n_steps_output,
        training_step_baseline,
        opt_b,
        amp_dtype,
        use_amp,
        args.num_steps,
        args.warmup_steps,
        device,
    )
    del net_b, opt_b, bl_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Optimized
    print("  [Optimized] persistent_workers + direct output_fields")
    net_o, opt_o = make_model_and_optim()
    opt_loader = make_optimized_loader(well_dm, batch_size, num_workers)
    torch.cuda.reset_peak_memory_stats(device)
    opt_train = bench_training(
        opt_loader,
        net_o,
        torch.nn.MSELoss(),
        formatter,
        n_steps_output,
        training_step_optimized,
        opt_o,
        amp_dtype,
        use_amp,
        args.num_steps,
        args.warmup_steps,
        device,
    )
    del net_o, opt_o, opt_loader

    print_comparison("Training Step", bl_train, opt_train)

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    speedup = opt_train["it_per_sec"] / bl_train["it_per_sec"]
    print(f"  End-to-end it/s:  {bl_train['it_per_sec']:.2f} -> {opt_train['it_per_sec']:.2f}  ({speedup:.2f}x)")
    data_speedup = bl_train["data_ms"] / opt_train["data_ms"] if opt_train["data_ms"] > 0 else float("inf")
    print(f"  Data loading:     {bl_train['data_ms']:.1f}ms -> {opt_train['data_ms']:.1f}ms  ({data_speedup:.2f}x)")
    step_speedup = bl_train["step_ms"] / opt_train["step_ms"] if opt_train["step_ms"] > 0 else float("inf")
    print(f"  Train step:       {bl_train['step_ms']:.1f}ms -> {opt_train['step_ms']:.1f}ms  ({step_speedup:.2f}x)")
    print()


if __name__ == "__main__":
    main()
