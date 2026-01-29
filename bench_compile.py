
import torch
import time
from nvsubquadratic.ops.delta_rule import delta_rule_recursive, delta_rule_parallel

def benchmark(fn, q, k, v, beta, name, num_iters=20):
    # Warmup
    for _ in range(5):
        fn(q, k, v, beta)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        fn(q, k, v, beta)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / num_iters
    print(f"{name}: {avg_time:.4f}s")
    return avg_time

if __name__ == "__main__":
    B, L, H, D = 16, 4096, 8, 160
    device = "cuda"
    dtype = torch.float32
    
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    beta = torch.randn(H, D, device=device, dtype=dtype)

    print(f"Benchmarking Delta Rule (L={L}, D={D}) on {device}")
    
    t1 = benchmark(delta_rule_recursive, q, k, v, beta, "Recursive (no compile)")
    
    # Simple wrapper for parallel to match signature
    def parallel_fn(q, k, v, beta):
        y, _ = delta_rule_parallel(q, k, v, beta)
        return y

    t2 = benchmark(parallel_fn, q, k, v, beta, "Parallel (no compile)")
    
    print("Compiling...")
    compiled_recursive = torch.compile(delta_rule_recursive)
    compiled_parallel = torch.compile(parallel_fn)
    
    t3 = benchmark(compiled_recursive, q, k, v, beta, "Recursive (compiled)")
    t4 = benchmark(compiled_parallel, q, k, v, beta, "Parallel (compiled)")
    
    print(f"Improvement (Recursive -> Compiled Parallel): {(t1/t4):.2f}x")
    print(f"Improvement (Parallel -> Compiled Parallel): {(t2/t4):.2f}x")
