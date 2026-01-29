
import torch
import time
from nvsubquadratic.ops.delta_rule import delta_rule_recursive, delta_rule_parallel

def run_bench(name, func, q, k, v, beta, num_warmup=2, num_iters=5):
    # Warmup
    for _ in range(num_warmup):
        func(q, k, v, beta)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        func(q, k, v, beta)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_iters
    print(f"{name}: {avg_time:.4f}s")
    return avg_time

if __name__ == "__main__":
    B, L, H, D = 8, 4096, 8, 160
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    beta = torch.randn(H, D, device=device, dtype=dtype)
    
    # Normalize k as per model
    k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
    
    print(f"Benchmarking L={L} on {device}...")
    
    t_rec = run_bench("Recursive (Optimized Level 1)", delta_rule_recursive, q, k, v, beta)
    t_par = run_bench("Parallel (Level 3)", delta_rule_parallel, q, k, v, beta)
    
    improvement = (t_rec - t_par) / t_rec * 100
    print(f"Parallel Improvement vs Recursive: {improvement:.2f}%")
    
    if torch.cuda.is_available():
        print("\nTesting with torch.compile...")
        rec_compiled = torch.compile(delta_rule_recursive)
        par_compiled = torch.compile(delta_rule_parallel)
        
        t_rec_comp = run_bench("Recursive Compiled", rec_compiled, q, k, v, beta, num_warmup=2)
        t_par_comp = run_bench("Parallel Compiled", par_compiled, q, k, v, beta, num_warmup=2)
        
        print(f"Final Speedup (Parallel Compiled vs Recursive Baseline): {t_rec / t_par_comp:.2f}x")
    else:
        print("\nSkipping torch.compile (CPU only environment detected)")
