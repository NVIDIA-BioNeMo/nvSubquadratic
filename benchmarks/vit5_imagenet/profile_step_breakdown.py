r"""Fine-grained per-phase profiling of a single training step.

Unlike profile_training_bottleneck.py (which measures each component
*independently*), this script instruments a real training loop to
measure where wall-clock time is actually spent *within each step*.

Each step is broken down into:
  1. DALI fetch  — next(iterator)
  2. Mixup/CutMix — timm Mixup on GPU
  3. Layout permute — NCHW → NHWC (if channels_first=False)
  4. Forward pass
  5. Backward pass (includes overlapped DDP allreduce)
  6. Optimizer step + zero_grad

Two passes are run:
  A) Instrumented — torch.cuda.synchronize() between each phase for
     accurate per-phase wall-clock timing (adds ~0.3ms sync overhead/step).
  B) Natural — no mid-step syncs, only sync at step boundary for true
     end-to-end step time.

Usage:
    PYTHONPATH=. torchrun --nproc_per_node=8 benchmarks/vit5_imagenet/profile_step_breakdown.py \\
        --dali-fused --ddp --num-workers 12 --model-size small
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from experiments.datamodules.imagenet import AugmentConfig, MixupConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


os.environ.setdefault("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
os.environ.setdefault("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

try:
    from apex.optimizers import FusedLAMB as Lamb

    OPTIMIZER_NAME = "Apex FusedLAMB"
except ImportError:
    from torch_optimizer import Lamb

    OPTIMIZER_NAME = "torch_optimizer.Lamb"

BATCH_SIZE = 256
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = IMAGE_SIZE // PATCH_SIZE

MODEL_PRESETS = {
    "small": {"hidden_dim": 384, "num_heads": 6, "num_blocks": 12, "num_registers": 4},
    "base": {"hidden_dim": 768, "num_heads": 12, "num_blocks": 12, "num_registers": 4},
}

AUGMENT_CFG = AugmentConfig(use_three_augment=True, color_jitter=0.3)
MIXUP_CFG = MixupConfig(mixup=0.8, cutmix=1.0, mixup_prob=1.0, mixup_switch_prob=0.5, smoothing=0.0)


def build_model(preset):
    """Build and return ViT-5 model for the given preset config dict."""
    hd = preset["hidden_dim"]
    return instantiate(
        LazyConfig(ViT5ClassificationNet)(
            in_channels=3,
            num_classes=1000,
            hidden_dim=hd,
            num_blocks=preset["num_blocks"],
            patch_size=PATCH_SIZE,
            image_size=IMAGE_SIZE,
            num_registers=preset["num_registers"],
            dropout_rate=0.0,
            norm_cfg=LazyConfig(RMSNorm)(dim=hd, eps=1e-6),
            block_cfg=LazyConfig(ViT5ResidualBlock)(
                sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                    hidden_dim=hd,
                    num_heads=preset["num_heads"],
                    num_patches_h=NUM_PATCHES,
                    num_patches_w=NUM_PATCHES,
                    num_registers=preset["num_registers"],
                    qk_norm=LazyConfig(RMSNorm)(dim=hd // preset["num_heads"], eps=1e-6),
                ),
                sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=hd, eps=1e-6),
                mlp_cfg=LazyConfig(MLP)(
                    dim=hd,
                    activation="gelu",
                    expansion_factor=4.0,
                    dropout_cfg=LazyConfig(nn.Dropout)(p=0.0),
                ),
                mlp_norm_cfg=LazyConfig(RMSNorm)(dim=hd, eps=1e-6),
                hidden_dim=hd,
                layer_scale_init=1e-4,
                drop_path_rate=0.05,
            ),
        )
    )


def build_dali_loader(optimized, device_id, prefetch_factor=3, num_workers=12):
    """Build and return a DALI dataloader and its datamodule."""
    common = {
        "data_dir": os.environ["IMAGENET_PATH"],
        "imagefolder_dir": os.environ.get("IMAGENET_FOLDER_PATH"),
        "prefetch_factor": prefetch_factor,
        "batch_size": BATCH_SIZE,
        "num_workers": num_workers,
        "pin_memory": True,
        "seed": 42,
        "image_size": IMAGE_SIZE,
        "final_image_size": IMAGE_SIZE,
        "num_classes": 1000,
        "drop_labels": False,
        "task": "classification",
        "augment_cfg": AUGMENT_CFG,
        "device_id": device_id,
    }
    if optimized == "fused":
        from experiments.datamodules.dali_imagenet_fused import DALIImageNetFusedDataModule

        dm = DALIImageNetFusedDataModule(**common)
    elif optimized == "v2":
        from experiments.datamodules._deprecated.dali_imagenet_optimized import DALIImageNetOptimizedDataModule

        dm = DALIImageNetOptimizedDataModule(**common)
    else:
        from experiments.datamodules._deprecated.dali_imagenet import DALIImageNetDataModule

        dm = DALIImageNetDataModule(**common)
    dm.setup("fit")
    return dm.train_dataloader(), dm


def main(args):
    """Run the fine-grained step breakdown profiler."""
    preset = MODEL_PRESETS[args.model_size]

    use_ddp = args.ddp
    local_rank = 0
    world_size = 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

    # Per-rank Triton cache to avoid compile race conditions
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_rank{local_rank}"

    device = torch.device("cuda", local_rank)
    rank0 = local_rank == 0

    def log(msg=""):
        if rank0:
            print(msg, flush=True)

    dali_opt = "fused" if args.dali_fused else ("v2" if args.dali_optimized else "")
    mode_str = "eager" if args.eager else "compiled"
    ddp_str = f"DDP x{world_size}" if use_ddp else "single-GPU"
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    gpu_name = torch.cuda.get_device_name()
    model_tag = args.model_size.upper()

    log(f"{'=' * 70}")
    log("STEP BREAKDOWN PROFILER")
    log(f"{'=' * 70}")
    log(f"  Device: {gpu_name} (node={node})")
    log(f"  Model:  ViT-5-{model_tag} (dim={preset['hidden_dim']})")
    log(f"  Mode:   {mode_str}, {ddp_str}")
    log(f"  DALI:   {dali_opt or 'original'}")
    log(f"  Batch:  {BATCH_SIZE}/GPU, Workers: {args.num_workers}")
    log()

    # ── Build model ──────────────────────────────────────────────────
    raw_model = build_model(preset).to(device)
    if not args.eager:
        log("Compiling model (max-autotune)...")
        raw_model = torch.compile(raw_model, mode="max-autotune")

    if use_ddp:
        model = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Lamb(model.parameters(), lr=4e-3, weight_decay=0.05)

    # ── Build dataloader ─────────────────────────────────────────────
    loader, dm = build_dali_loader(
        optimized=dali_opt,
        device_id=local_rank,
        prefetch_factor=3,
        num_workers=args.num_workers,
    )
    dm.trainer = type("_Mock", (), {"training": True})()

    # ── Helper: prepare batch (full on_before_batch_transfer for non-fused) ─
    is_fused = dali_opt == "fused"

    def prepare_batch_full(raw_batch):
        """Call dm.on_before_batch_transfer for non-fused, inline for fused."""
        if not is_fused:
            return dm.on_before_batch_transfer(raw_batch, 0)
        images, labels = raw_batch
        labels = labels.to(device=images.device)
        if dm.mixup_fn is not None:
            images, labels = dm.mixup_fn(images, labels)
        if hasattr(dm, "channels_first") and not dm.channels_first:
            images = images.permute(0, 2, 3, 1).contiguous()
        if labels.ndim == 1:
            labels = labels.view(-1)
        return {"input": images, "label": labels, "condition": None}

    # ── Warmup (triggers torch.compile on each rank) ─────────────────
    warmup_iters = 5 if args.eager else 25
    log(f"Warming up ({warmup_iters} iters)...")
    it = iter(loader)
    for _ in range(warmup_iters):
        batch = prepare_batch_full(next(it))
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
            loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    if use_ddp:
        dist.barrier()
    log("Warmup complete.\n")

    num_steps = args.num_steps

    # ══════════════════════════════════════════════════════════════════
    # PASS A: Instrumented — sync between each phase
    # ══════════════════════════════════════════════════════════════════
    log(f"{'=' * 70}")
    log(f"PASS A: Instrumented ({num_steps} steps, sync between phases)")
    log(f"{'=' * 70}")

    if is_fused:
        phase_names = [
            "dali_fetch",
            "mixup",
            "permute",
            "forward",
            "backward",
            "optim_step",
            "zero_grad",
        ]
    else:
        phase_names = [
            "dali_fetch",
            "batch_transfer",
            "forward",
            "backward",
            "optim_step",
            "zero_grad",
        ]
    phase_times = {name: [] for name in phase_names}
    step_times_instrumented = []

    torch.cuda.synchronize()
    if use_ddp:
        dist.barrier()

    for _ in range(num_steps):
        step_t0 = time.perf_counter()

        # 1. DALI fetch
        torch.cuda.synchronize()
        t = time.perf_counter()
        raw_batch = next(it)
        torch.cuda.synchronize()
        phase_times["dali_fetch"].append((time.perf_counter() - t) * 1000)

        if is_fused:
            images, labels = raw_batch

            # 2. Mixup/CutMix
            t = time.perf_counter()
            labels = labels.to(device=images.device)
            if dm.mixup_fn is not None:
                images, labels = dm.mixup_fn(images, labels)
            torch.cuda.synchronize()
            phase_times["mixup"].append((time.perf_counter() - t) * 1000)

            # 3. Layout permute (NCHW → NHWC)
            t = time.perf_counter()
            if hasattr(dm, "channels_first") and not dm.channels_first:
                images = images.permute(0, 2, 3, 1).contiguous()
            if labels.ndim == 1:
                labels = labels.view(-1)
            batch = {"input": images, "label": labels, "condition": None}
            torch.cuda.synchronize()
            phase_times["permute"].append((time.perf_counter() - t) * 1000)
        else:
            # Non-fused: full on_before_batch_transfer (augmentations + normalize)
            t = time.perf_counter()
            batch = dm.on_before_batch_transfer(raw_batch, 0)
            torch.cuda.synchronize()
            phase_times["batch_transfer"].append((time.perf_counter() - t) * 1000)

        # 4. Forward
        t = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
        torch.cuda.synchronize()
        phase_times["forward"].append((time.perf_counter() - t) * 1000)

        # 5. Backward (includes DDP allreduce overlap)
        t = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        phase_times["backward"].append((time.perf_counter() - t) * 1000)

        # 6. Optimizer step
        t = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        phase_times["optim_step"].append((time.perf_counter() - t) * 1000)

        # 7. Zero grad
        t = time.perf_counter()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        phase_times["zero_grad"].append((time.perf_counter() - t) * 1000)

        step_times_instrumented.append((time.perf_counter() - step_t0) * 1000)

    # ── Print PASS A results ─────────────────────────────────────────
    log()
    log(f"  {'Phase':<20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}  {'% step':>7s}")
    log(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}  {'─' * 7}")

    mean_step = statistics.mean(step_times_instrumented)
    total_phase_mean = 0.0

    for name in phase_names:
        vals = phase_times[name]
        m = statistics.mean(vals)
        total_phase_mean += m
        med = statistics.median(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0
        lo, hi = min(vals), max(vals)
        pct = m / mean_step * 100
        log(f"  {name:<20s} {m:7.2f}ms {med:7.2f}ms {sd:7.2f}ms {lo:7.2f}ms {hi:7.2f}ms  {pct:6.1f}%")

    sync_overhead = mean_step - total_phase_mean
    log(
        f"  {'sync overhead':<20s} {sync_overhead:7.2f}ms {'':>8s} {'':>8s} {'':>8s} {'':>8s}  {sync_overhead / mean_step * 100:6.1f}%"
    )
    log(f"  {'─' * 20} {'─' * 8}")
    log(f"  {'TOTAL (instrumented)':<20s} {mean_step:7.2f}ms")
    log()

    if is_fused:
        serial_gpu_aug = statistics.mean(phase_times["mixup"]) + statistics.mean(phase_times["permute"])
        log(f"  Serial GPU augmentation (mixup + permute): {serial_gpu_aug:.2f}ms")
    else:
        serial_gpu_aug = statistics.mean(phase_times["batch_transfer"])
        log(f"  Serial GPU augmentation (on_before_batch_transfer): {serial_gpu_aug:.2f}ms")

    # ══════════════════════════════════════════════════════════════════
    # PASS B: Natural — no mid-step syncs (true step time)
    # ══════════════════════════════════════════════════════════════════
    log()
    log(f"{'=' * 70}")
    log(f"PASS B: Natural ({num_steps} steps, sync only at step boundary)")
    log(f"{'=' * 70}")

    step_times_natural = []

    torch.cuda.synchronize()
    if use_ddp:
        dist.barrier()

    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        batch = prepare_batch_full(next(it))
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        step_times_natural.append((time.perf_counter() - t0) * 1000)

    nat_mean = statistics.mean(step_times_natural)
    nat_med = statistics.median(step_times_natural)
    nat_std = statistics.stdev(step_times_natural) if len(step_times_natural) > 1 else 0
    nat_tput = BATCH_SIZE / (nat_mean / 1000)
    agg_tput = nat_tput * world_size

    log(f"  Mean:   {nat_mean:.2f}ms")
    log(f"  Median: {nat_med:.2f}ms")
    log(f"  Std:    {nat_std:.2f}ms")
    log(f"  Per-GPU throughput: {nat_tput:.0f} samp/s")
    if use_ddp:
        log(f"  Aggregate throughput: {agg_tput:.0f} samp/s ({world_size} GPUs)")
    log()

    # ══════════════════════════════════════════════════════════════════
    # PASS C: CUDA events (no CPU sync, pure GPU timeline)
    # ══════════════════════════════════════════════════════════════════
    log(f"{'=' * 70}")
    log(f"PASS C: CUDA events ({num_steps} steps, no CPU sync, GPU timeline)")
    log(f"{'=' * 70}")

    events_per_step = []
    if is_fused:
        ev_names = [
            "step_start",
            "after_fetch",
            "after_mixup",
            "after_permute",
            "after_fwd",
            "after_bwd",
            "after_optim",
        ]
    else:
        ev_names = ["step_start", "after_fetch", "after_transfer", "after_fwd", "after_bwd", "after_optim"]

    torch.cuda.synchronize()
    if use_ddp:
        dist.barrier()

    for _ in range(num_steps):
        evs = {n: torch.cuda.Event(enable_timing=True) for n in ev_names}

        evs["step_start"].record()

        raw_batch = next(it)
        evs["after_fetch"].record()

        if is_fused:
            images, labels = raw_batch
            labels = labels.to(device=images.device)
            if dm.mixup_fn is not None:
                images, labels = dm.mixup_fn(images, labels)
            evs["after_mixup"].record()

            if hasattr(dm, "channels_first") and not dm.channels_first:
                images = images.permute(0, 2, 3, 1).contiguous()
            if labels.ndim == 1:
                labels = labels.view(-1)
            batch = {"input": images, "label": labels, "condition": None}
            evs["after_permute"].record()
        else:
            batch = dm.on_before_batch_transfer(raw_batch, 0)
            evs["after_transfer"].record()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)["logits"]
            loss = loss_fn(out, batch["label"])
        evs["after_fwd"].record()

        loss.backward()
        evs["after_bwd"].record()

        optimizer.step()
        evs["after_optim"].record()

        events_per_step.append(evs)

    torch.cuda.synchronize()

    if is_fused:
        gpu_phase_names = ["fetch", "mixup", "permute", "forward", "backward", "optim"]
        gpu_phase_pairs = [
            ("step_start", "after_fetch"),
            ("after_fetch", "after_mixup"),
            ("after_mixup", "after_permute"),
            ("after_permute", "after_fwd"),
            ("after_fwd", "after_bwd"),
            ("after_bwd", "after_optim"),
        ]
    else:
        gpu_phase_names = ["fetch", "batch_transfer", "forward", "backward", "optim"]
        gpu_phase_pairs = [
            ("step_start", "after_fetch"),
            ("after_fetch", "after_transfer"),
            ("after_transfer", "after_fwd"),
            ("after_fwd", "after_bwd"),
            ("after_bwd", "after_optim"),
        ]
    gpu_phase_times = {n: [] for n in gpu_phase_names}
    gpu_step_totals = []

    for evs in events_per_step:
        total = evs["step_start"].elapsed_time(evs["after_optim"])
        gpu_step_totals.append(total)
        for name, (ev_start, ev_end) in zip(gpu_phase_names, gpu_phase_pairs):
            gpu_phase_times[name].append(evs[ev_start].elapsed_time(evs[ev_end]))

    log()
    log(f"  {'Phase':<20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s}  {'% step':>7s}")
    log(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8}  {'─' * 7}")

    gpu_mean_step = statistics.mean(gpu_step_totals)
    for name in gpu_phase_names:
        vals = gpu_phase_times[name]
        m = statistics.mean(vals)
        med = statistics.median(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0
        pct = m / gpu_mean_step * 100
        log(f"  {name:<20s} {m:7.2f}ms {med:7.2f}ms {sd:7.2f}ms  {pct:6.1f}%")

    log(f"  {'─' * 20} {'─' * 8}")
    log(f"  {'TOTAL (GPU events)':<20s} {gpu_mean_step:7.2f}ms")
    log()

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    log(f"{'=' * 70}")
    log(f"SUMMARY — ViT-5-{model_tag}, DALI-{dali_opt or 'original'}, {ddp_str}")
    log(f"{'=' * 70}")

    data_phases = statistics.mean(phase_times["dali_fetch"])
    if is_fused:
        serial_aug = serial_gpu_aug
    else:
        serial_aug = statistics.mean(phase_times["batch_transfer"])
    compute = statistics.mean(phase_times["forward"]) + statistics.mean(phase_times["backward"])
    optim = statistics.mean(phase_times["optim_step"]) + statistics.mean(phase_times["zero_grad"])
    theoretical_min = max(data_phases + serial_aug, compute) + optim

    log(f"  Natural step time:        {nat_mean:7.2f}ms")
    log(f"  Instrumented step time:   {mean_step:7.2f}ms  (sync overhead: {sync_overhead:.1f}ms)")
    log(f"  GPU event step time:      {gpu_mean_step:7.2f}ms")
    log()
    log(f"  DALI fetch:               {data_phases:7.2f}ms")
    if is_fused:
        log(
            f"  Serial GPU augment:       {serial_aug:7.2f}ms  (mixup={statistics.mean(phase_times['mixup']):.2f} + permute={statistics.mean(phase_times['permute']):.2f})"
        )
    else:
        log(f"  Batch transfer (augment): {serial_aug:7.2f}ms  (ThreeAugment + ColorJitter + normalize)")
    log(f"  Forward:                  {statistics.mean(phase_times['forward']):7.2f}ms")
    log(f"  Backward (+allreduce):    {statistics.mean(phase_times['backward']):7.2f}ms")
    log(f"  Optimizer + zero_grad:    {optim:7.2f}ms")
    log()
    log(f"  Theoretical min:          {theoretical_min:7.2f}ms  = max(dali+aug, fwd+bwd) + optim")
    log(f"  Actual (natural):         {nat_mean:7.2f}ms")
    log(f"  Gap:                      {nat_mean - theoretical_min:7.2f}ms")
    min_component = min(data_phases + serial_aug, compute)
    if min_component > 0:
        log(
            f"  Overlap efficiency:       {((data_phases + serial_aug) + compute - (nat_mean - optim)) / min_component * 100:.1f}%"
        )
    log()

    # ── Write JSON results ───────────────────────────────────────────
    if rank0:
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "node": node,
            "gpu": gpu_name,
            "model_size": args.model_size,
            "mode": mode_str,
            "ddp": use_ddp,
            "world_size": world_size,
            "dali_variant": dali_opt or "original",
            "batch_size": BATCH_SIZE,
            "num_workers": args.num_workers,
            "num_steps": num_steps,
            "natural_step_ms": round(nat_mean, 2),
            "natural_step_median_ms": round(nat_med, 2),
            "instrumented_step_ms": round(mean_step, 2),
            "gpu_event_step_ms": round(gpu_mean_step, 2),
            "phases_ms": {name: round(statistics.mean(phase_times[name]), 2) for name in phase_names},
            "gpu_phases_ms": {name: round(statistics.mean(gpu_phase_times[name]), 2) for name in gpu_phase_names},
            "serial_aug_ms": round(serial_aug, 2),
            "theoretical_min_ms": round(theoretical_min, 2),
            "gap_ms": round(nat_mean - theoretical_min, 2),
            "agg_throughput": round(agg_tput),
        }

        out_dir = Path("benchmarks/vit5_imagenet")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"step_breakdown_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(out_path, "a") as f:
            f.write(json.dumps(results) + "\n")
        log(f"Results appended to {out_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-grained step breakdown profiler")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--dali-fused", action="store_true")
    parser.add_argument("--dali-optimized", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--model-size", choices=["small", "base"], default="small")
    main(parser.parse_args())
