"""Benchmark WELL dataloader throughput.

Measures how fast batches can be loaded from disk, isolating the dataloader
from model forward/backward. Useful for diagnosing I/O bottlenecks.

Usage:
    PYTHONPATH=. python benchmarks/well/bench_dataloader.py \
        --config examples/well/supernova_explosion_64/cfg_vit5_attention.py \
        [--num-batches 200] [--num-epochs 3] [--warmup-batches 10]
"""

import argparse
import time

import torch

from experiments.utils.cli import load_config_from_file
from nvsubquadratic.lazy_config import instantiate


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark WELL dataloader throughput")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-batches", type=int, default=200, help="Batches to iterate per epoch")
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs to measure (tests persistent_workers)"
    )
    parser.add_argument("--warmup-batches", type=int, default=10, help="Warmup batches to skip before timing")
    return parser.parse_args()


def bench_epoch(loader, num_batches, warmup_batches):
    """Time one epoch of dataloader iteration, returning batches/sec."""
    total_samples = 0
    t_start = None

    for i, batch in enumerate(loader):
        if i >= warmup_batches + num_batches:
            break
        if i == warmup_batches:
            t_start = time.perf_counter()
        if i >= warmup_batches:
            bs = batch["input_fields"].shape[0]
            total_samples += bs

    elapsed = time.perf_counter() - t_start
    measured_batches = num_batches
    return {
        "elapsed_s": elapsed,
        "batches": measured_batches,
        "samples": total_samples,
        "batches_per_sec": measured_batches / elapsed,
        "samples_per_sec": total_samples / elapsed,
    }


def main():
    """Benchmark WELL dataloader throughput."""
    args = parse_args()
    config = load_config_from_file(args.config)

    print(f"Config: {args.config}")
    print(f"Batch size: {config.dataset.batch_size}")
    print(f"Num workers: {config.dataset.num_workers}")
    prefetch = getattr(config.dataset, "prefetch_factor", "default")
    print(f"Prefetch factor: {prefetch}")
    print(
        f"Measuring {args.num_batches} batches/epoch, {args.num_epochs} epochs, {args.warmup_batches} warmup batches"
    )
    print()

    # Setup datamodule
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    loader = datamodule.train_dataloader()

    # Check if persistent_workers is enabled
    if hasattr(loader, "loader"):
        # _DownsampledDataLoader wrapper
        inner = loader.loader
    else:
        inner = loader
    pw = getattr(inner, "persistent_workers", False) if isinstance(inner, torch.utils.data.DataLoader) else "N/A"
    pf = getattr(inner, "prefetch_factor", None) if isinstance(inner, torch.utils.data.DataLoader) else "N/A"
    print(f"DataLoader persistent_workers={pw}, prefetch_factor={pf}")
    print("-" * 60)

    for epoch in range(args.num_epochs):
        result = bench_epoch(loader, args.num_batches, args.warmup_batches)
        print(
            f"Epoch {epoch}: {result['batches_per_sec']:.1f} batches/s | "
            f"{result['samples_per_sec']:.1f} samples/s | "
            f"{result['elapsed_s']:.2f}s"
        )

    print("-" * 60)
    print("Epoch 0 includes worker startup overhead.")
    print("Epoch 1+ shows steady-state throughput (persistent_workers benefit).")


if __name__ == "__main__":
    main()
