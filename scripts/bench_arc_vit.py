import time

import torch
import torch.nn.functional as F

from nvsubquadratic.networks.baselines.arc_vit import ARCViT


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Mock configuration
max_size = 32
num_colors = 12
embed_dim = 256
depth = 6
num_heads = 8
mlp_dim = 512
patch_size = 2

model = ARCViT(
    num_tasks=800,
    max_size=max_size,
    num_colors=num_colors,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    patch_size=patch_size,
)


def params_count(model):
    return sum(p.numel() for p in model.parameters())


print(f"Model Parameters: {params_count(model) / 1e6:.2f} M")

if device == "cuda":
    model = model.to(device)


def test_batch_size(batch_size):
    input_tensor = torch.randint(0, 10, (batch_size, max_size, max_size), device=device)
    task_id = torch.randint(0, 800, (batch_size,), device=device)
    attention_mask = torch.ones((batch_size, max_size, max_size), device=device)
    labels = torch.randint(0, 10, (batch_size, max_size, max_size), device=device)

    input_and_cond = {"input": input_tensor, "condition": {"task_id": task_id, "attention_mask": attention_mask}}

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        for _ in range(3):
            output = model(input_and_cond)
            loss = F.cross_entropy(output["logits"], labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, 0
        raise e

    # Benchmark
    iters = 10
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    try:
        for _ in range(iters):
            output = model(input_and_cond)
            loss = F.cross_entropy(output["logits"], labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, 0
        raise e

    avg_time = (time.time() - start) / iters
    return True, avg_time


# Find max batch size using doubling
batch_size = 32
found_max = False
max_working_bs = 32
best_time = 0

print("Benchmarking max batch size...")
while not found_max:
    success, avg_time = test_batch_size(batch_size)
    if success:
        print(
            f"Batch size {batch_size} succeeded, {1 / avg_time:.2f} iters/sec ({batch_size / avg_time:.2f} items/sec)"
        )
        max_working_bs = batch_size
        best_time = avg_time
        batch_size *= 2
        # Safety limit to avoid waiting forever
        if batch_size > 8192:
            break
    else:
        print(f"Batch size {batch_size} OOM'd")
        found_max = True

# Refine search between max_working_bs and batch_size (which OOM'd)
if max_working_bs < 8192 and found_max:
    low = max_working_bs
    high = batch_size
    while low + 64 < high:
        mid = (low + high) // 2
        mid = (mid // 32) * 32  # Keep it a multiple of 32
        success, avg_time = test_batch_size(mid)
        if success:
            print(f"Batch size {mid} succeeded, {1 / avg_time:.2f} iters/sec ({mid / avg_time:.2f} items/sec)")
            max_working_bs = mid
            best_time = avg_time
            low = mid
        else:
            print(f"Batch size {mid} OOM'd")
            high = mid

print(f"Estimated Max Batch Size: {max_working_bs}")
print(f"Average time per iteration for max BS: {best_time:.4f} seconds")
