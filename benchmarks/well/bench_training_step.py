"""Benchmark WELL end-to-end training throughput (forward + backward + optimizer).

Measures wall-clock it/s for actual training steps, including model forward,
loss computation, backward pass, and optimizer step. Runs without W&B logging
or checkpointing to isolate training throughput.

Usage:
    PYTHONPATH=. python benchmarks/well/bench_training_step.py \
        --config examples/well/supernova_explosion_64/cfg_vit5_attention.py \
        [--num-steps 200] [--warmup-steps 20] [--compile]
"""

import argparse
import time

import pytorch_lightning as pl
import torch
from einops import rearrange

from experiments.utils.cli import load_config_from_file
from nvsubquadratic.lazy_config import instantiate


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark WELL training step throughput")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-steps", type=int, default=200, help="Training steps to measure")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps (skip timing)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on model")
    return parser.parse_args()


def process_batch_input(batch):
    """Process WELL batch format into model input (same as WELLRegressionWrapper._process_batch_input)."""
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
    return model_input


def main():
    """Benchmark WELL training step throughput (forward + backward + optimizer)."""
    args = parse_args()
    config = load_config_from_file(args.config)

    pl.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    n_steps_output = config.dataset.n_steps_output

    print(f"Config: {args.config}")
    print(f"Batch size: {config.dataset.batch_size}")
    print(f"Compile: {args.compile}")
    print(f"Precision: {config.train.precision}")
    print(f"Steps: {args.warmup_steps} warmup + {args.num_steps} measured")
    print()

    # Setup datamodule
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader()

    # Setup model
    network = instantiate(config.net, in_channels=datamodule.input_channels, out_channels=datamodule.output_channels)

    if getattr(config, "compile_compatible_fftconv", False):
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True

    if args.compile:
        mode = getattr(config, "compile_mode", None)
        compile_kwargs = {"mode": mode} if mode else {}
        print(f"Compiling model (mode={mode})...")
        network = torch.compile(network, **compile_kwargs)

    network = network.to(device)
    loss_fn = torch.nn.MSELoss()

    # Setup optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-5)

    # Use AMP scaler for bf16-mixed
    use_amp = "bf16" in config.train.precision or "16" in config.train.precision
    amp_dtype = torch.bfloat16 if "bf16" in config.train.precision else torch.float16

    # Run training loop
    step = 0
    total_steps = args.warmup_steps + args.num_steps
    step_times = []
    data_times = []

    print("Running benchmark...")
    data_start = time.perf_counter()

    for batch in loader:
        if step >= total_steps:
            break

        # Move batch to GPU
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data_end = time.perf_counter()

        # Forward + backward
        t0 = time.perf_counter()

        model_input = process_batch_input(batch)
        target = batch["output_fields"]
        if n_steps_output == 1:
            target = target[:, 0]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred = network({"input": model_input, "condition": None})["logits"]
            loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.perf_counter()

        if step >= args.warmup_steps:
            step_times.append(t1 - t0)
            data_times.append(data_end - data_start)

        step += 1
        data_start = time.perf_counter()

        if step == args.warmup_steps:
            print(f"  Warmup complete ({args.warmup_steps} steps)")

    # Report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    step_times_t = torch.tensor(step_times)
    data_times_t = torch.tensor(data_times)
    total_times_t = step_times_t + data_times_t

    batch_size = config.dataset.batch_size

    print(f"  Batch size:           {batch_size}")
    print(f"  Steps measured:       {len(step_times)}")
    print()
    print("  --- Per-step timing (ms) ---")
    print(
        f"  Data loading:         {data_times_t.mean().item() * 1000:8.1f} +/- {data_times_t.std().item() * 1000:.1f}"
    )
    print(
        f"  Train step (fwd+bwd): {step_times_t.mean().item() * 1000:8.1f} +/- {step_times_t.std().item() * 1000:.1f}"
    )
    print(
        f"  Total:                {total_times_t.mean().item() * 1000:8.1f} +/- {total_times_t.std().item() * 1000:.1f}"
    )
    print()
    print("  --- Throughput ---")
    print(f"  it/s:                 {1.0 / total_times_t.mean().item():.2f}")
    print(f"  samples/s:            {batch_size / total_times_t.mean().item():.1f}")
    print()

    # Peak memory
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"  Peak GPU memory:      {peak_mem:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
