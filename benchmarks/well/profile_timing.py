"""Profile forward/backward/optimizer breakdown for the Gray-Scott Hyena model.

Compares compiled vs uncompiled, with proper CUDA synchronization.
"""

import time

import numpy as np
import torch
from examples.well.gray_scott_reaction_diffusion.cfg_hyena_gaussian_mask import get_config

from nvsubquadratic.lazy_config import instantiate


def profile_model(net, optimizer, inp, label=""):
    """Time forward, backward, grad-clip, and optimizer steps for a model."""
    N = 30
    SKIP = 10

    fwd_times, bwd_times, clip_times, opt_times = [], [], [], []

    for i in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = net(inp)["logits"]
            loss = out.float().mean()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        t6 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t7 = time.perf_counter()

        optimizer.zero_grad()

        if i >= SKIP:
            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((t3 - t2) * 1000)
            clip_times.append((t5 - t4) * 1000)
            opt_times.append((t7 - t6) * 1000)

    fwd = np.array(fwd_times)
    bwd = np.array(bwd_times)
    clip = np.array(clip_times)
    opt = np.array(opt_times)

    total = fwd.mean() + bwd.mean() + clip.mean() + opt.mean()
    print(f"\n=== {label} ===")
    print(f"Forward:       {fwd.mean():>7.1f}ms  (std {fwd.std():.1f})")
    print(f"Backward:      {bwd.mean():>7.1f}ms  (std {bwd.std():.1f})")
    print(f"Grad clip:     {clip.mean():>7.1f}ms  (std {clip.std():.1f})")
    print(f"Optimizer:     {opt.mean():>7.1f}ms  (std {opt.std():.1f})")
    print(f"Clip+Opt:      {(clip.mean() + opt.mean()):>7.1f}ms")
    print(f"Total compute: {total:>7.1f}ms  ({1000 / total:.1f} it/s)")


config = get_config()
x = torch.randn(16, 128, 128, 8, device="cuda", dtype=torch.float32)
inp = {"input": x, "condition": None}

# --- Uncompiled ---
net = instantiate(config.net, in_channels=8, out_channels=2).cuda()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
n_params = sum(p.numel() for p in net.parameters())
print(f"Parameters: {n_params:,} trainable")

for _ in range(5):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(inp)["logits"]
        loss = out.float().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()

profile_model(net, optimizer, inp, label="UNCOMPILED")

# --- Compiled ---
del net, optimizer
torch.cuda.empty_cache()

net = instantiate(config.net, in_channels=8, out_channels=2).cuda()
print("Compiling model...")
net = torch.compile(net, mode="max-autotune-no-cudagraphs")
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

for _ in range(5):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(inp)["logits"]
        loss = out.float().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
print("Compile warmup done")

profile_model(net, optimizer, inp, label="COMPILED (max-autotune-no-cudagraphs)")
