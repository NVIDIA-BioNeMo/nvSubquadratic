"""Profile the full training loop to identify overhead sources.

Compares:
  1. Pure compute (no DataLoader, static tensors)
  2. Full loop with DataLoader + preprocessing (no PL)
  3. Breakdown of each component with CUDA synchronization

This tells us exactly where the ~200ms gap between pure compute (83ms)
and PL-reported iteration time (284ms) comes from.
"""

import time

import numpy as np
import torch
from einops import rearrange
from examples.well.gray_scott_reaction_diffusion.cfg_hyena_gaussian_mask import get_config

from experiments.datamodules.pde.well import WellDataModule
from nvsubquadratic.lazy_config import instantiate


def time_it(fn, n=30, skip=10, sync=True):
    """Time a callable over *n* iterations, discarding the first *skip* for warmup."""
    times = []
    for i in range(n):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if sync:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= skip:
            times.append((t1 - t0) * 1000)
    return np.array(times), result


# ─── Setup ────────────────────────────────────────────────────────────────────
config = get_config()
device = torch.device("cuda")

# Model (uncompiled first)
net = instantiate(config.net, in_channels=8, out_channels=2).to(device)
n_params = sum(p.numel() for p in net.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

# DataModule
dm = WellDataModule(
    well_base_path="/shared/data/image_datasets/the_well/datasets",
    well_dataset_name="gray_scott_reaction_diffusion",
    batch_size=16,
    num_workers=12,
    use_normalization=True,
    n_steps_input=4,
    n_steps_output=1,
    max_rollout_steps=100,
    prefetch_factor=4,
    persistent_workers=True,
)
dm.prepare_data()
dm.setup()

loader = dm.train_dataloader()
data_iter = iter(loader)

# Warmup DataLoader
for _ in range(5):
    batch = next(data_iter)
print("DataLoader warmup done")

# Warmup model
dummy_inp = torch.randn(16, 128, 128, 8, device=device)
for _ in range(3):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net({"input": dummy_inp, "condition": None})["logits"]
        loss = out.float().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
print("Model warmup done\n")


# ─── Profile individual components ───────────────────────────────────────────
print("=" * 70)
print("COMPONENT-LEVEL PROFILING (with cuda.synchronize)")
print("=" * 70)

N = 50
SKIP = 10
timings = {
    k: []
    for k in [
        "dataloader_fetch",
        "to_gpu",
        "preprocess",
        "forward",
        "backward",
        "grad_clip",
        "optimizer_step",
        "zero_grad",
        "total_iter",
    ]
}

for i in range(N):
    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    # 1. DataLoader fetch
    t0 = time.perf_counter()
    batch = next(data_iter)
    t1 = time.perf_counter()

    # 2. Move to GPU
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    # 3. Preprocess (matches WELLRegressionWrapper._process_batch_input)
    t4 = time.perf_counter()
    model_input = rearrange(batch["input_fields"], "b t h w c -> b h w (t c)")
    if "constant_fields" in batch:
        model_input = torch.cat([model_input, batch["constant_fields"]], dim=-1)
    target = batch["output_fields"][:, 0]
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    # 4. Forward + loss
    t6 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = net({"input": model_input, "condition": None})["logits"]
        loss = torch.nn.functional.mse_loss(pred, target)
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

    t_total_end = time.perf_counter()

    if i >= SKIP:
        timings["dataloader_fetch"].append((t1 - t0) * 1000)
        timings["to_gpu"].append((t3 - t2) * 1000)
        timings["preprocess"].append((t5 - t4) * 1000)
        timings["forward"].append((t7 - t6) * 1000)
        timings["backward"].append((t9 - t8) * 1000)
        timings["grad_clip"].append((t11 - t10) * 1000)
        timings["optimizer_step"].append((t13 - t12) * 1000)
        timings["zero_grad"].append((t15 - t14) * 1000)
        timings["total_iter"].append((t_total_end - t_total_start) * 1000)

total_components = 0.0
for name, vals in timings.items():
    arr = np.array(vals)
    label = f"{name}:"
    if name != "total_iter":
        total_components += arr.mean()
    print(f"  {label:<22s} {arr.mean():>7.1f}ms  (std {arr.std():>5.1f}, p95 {np.percentile(arr, 95):>7.1f})")

print(f"\n  Sum of components:     {total_components:>7.1f}ms")
print(f"  Total iteration:       {np.mean(timings['total_iter']):>7.1f}ms")
print(f"  Sync overhead:         {np.mean(timings['total_iter']) - total_components:>7.1f}ms")
print(f"\n  Throughput:            {1000.0 / np.mean(timings['total_iter']):>7.1f} it/s")
print("\n  PL callback reported:    284ms → 3.5 it/s")
print("  Pure compute (earlier):   83ms → 12.1 it/s")

del data_iter, loader

# ─── Now repeat with torch.compile ──────────────────────────────────────────
print("\n\n")
print("=" * 70)
print("COMPILED MODEL — COMPONENT-LEVEL PROFILING (with cuda.synchronize)")
print("=" * 70)

del net, optimizer
torch.cuda.empty_cache()

net = instantiate(config.net, in_channels=8, out_channels=2).to(device)
print("Compiling model...")
net = torch.compile(net, mode="max-autotune-no-cudagraphs")
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

loader = dm.train_dataloader()
data_iter = iter(loader)

# Warmup DataLoader + compiled model
for i in range(8):
    batch = next(data_iter)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    model_input = rearrange(batch["input_fields"], "b t h w c -> b h w (t c)")
    target = batch["output_fields"][:, 0]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = net({"input": model_input, "condition": None})["logits"]
        loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
print("Compiled model warmup done\n")

timings2 = {
    k: []
    for k in [
        "dataloader_fetch",
        "to_gpu",
        "preprocess",
        "forward",
        "backward",
        "grad_clip",
        "optimizer_step",
        "zero_grad",
        "total_iter",
    ]
}

for i in range(N):
    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    batch = next(data_iter)
    t1 = time.perf_counter()

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    model_input = rearrange(batch["input_fields"], "b t h w c -> b h w (t c)")
    if "constant_fields" in batch:
        model_input = torch.cat([model_input, batch["constant_fields"]], dim=-1)
    target = batch["output_fields"][:, 0]
    torch.cuda.synchronize()
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = net({"input": model_input, "condition": None})["logits"]
        loss = torch.nn.functional.mse_loss(pred, target)
    torch.cuda.synchronize()
    t7 = time.perf_counter()

    t8 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t9 = time.perf_counter()

    t10 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    torch.cuda.synchronize()
    t11 = time.perf_counter()

    t12 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    t13 = time.perf_counter()

    t14 = time.perf_counter()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t15 = time.perf_counter()

    t_total_end = time.perf_counter()

    if i >= SKIP:
        timings2["dataloader_fetch"].append((t1 - t0) * 1000)
        timings2["to_gpu"].append((t3 - t2) * 1000)
        timings2["preprocess"].append((t5 - t4) * 1000)
        timings2["forward"].append((t7 - t6) * 1000)
        timings2["backward"].append((t9 - t8) * 1000)
        timings2["grad_clip"].append((t11 - t10) * 1000)
        timings2["optimizer_step"].append((t13 - t12) * 1000)
        timings2["zero_grad"].append((t15 - t14) * 1000)
        timings2["total_iter"].append((t_total_end - t_total_start) * 1000)

total_comp2 = 0.0
for name, vals in timings2.items():
    arr = np.array(vals)
    label = f"{name}:"
    if name != "total_iter":
        total_comp2 += arr.mean()
    print(f"  {label:<22s} {arr.mean():>7.1f}ms  (std {arr.std():>5.1f}, p95 {np.percentile(arr, 95):>7.1f})")

print(f"\n  Sum of components:     {total_comp2:>7.1f}ms")
print(f"  Total iteration:       {np.mean(timings2['total_iter']):>7.1f}ms")
print(f"  Throughput:            {1000.0 / np.mean(timings2['total_iter']):>7.1f} it/s")

del data_iter, loader
