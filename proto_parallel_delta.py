
import torch
import torch.nn.functional as F

def delta_rule_parallel(q, k, v, beta, chunk_size=64):
    """Parallel implementation of the Delta Rule using associative scan logic.
    
    Formula: S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    
    We use a chunked approach to maintain efficiency and avoid O(L D^2) memory.
    """
    B, L, H, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # 1. Reshape and setup
    q = q.transpose(1, 2).reshape(B * H, L, D)  # (N, L, D) where N = B*H
    k = k.transpose(1, 2).reshape(B * H, L, D)
    v = v.transpose(1, 2).reshape(B * H, L, D)
    
    # beta in model is (H, D), here we want (B*H, D, 1)
    # The original recursive code: gamma = beta.unsqueeze(-1).repeat(B, 1, 1).contiguous()
    gamma = beta.unsqueeze(-1).repeat(B, 1, 1).contiguous() # (N, D, 1)
    
    num_chunks = L // chunk_size
    assert L % chunk_size == 0, "Sequence length must be divisible by chunk size for this prototype"
    
    q_chunks = q.reshape(B * H, num_chunks, chunk_size, D)
    k_chunks = k.reshape(B * H, num_chunks, chunk_size, D)
    v_chunks = v.reshape(B * H, num_chunks, chunk_size, D)
    
    all_final_states = torch.zeros(B * H, num_chunks, D, D, device=device, dtype=dtype)
    all_transitions = torch.zeros(B * H, num_chunks, D, D, device=device, dtype=dtype)
    
    # Sequential pass to get parameters for EACH chunk
    for c in range(num_chunks):
        s = torch.zeros(B * H, D, D, device=device, dtype=dtype)
        a = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).repeat(B * H, 1, 1)
        
        kc = k_chunks[:, c]
        vc = v_chunks[:, c]
        
        for t in range(chunk_size):
            ki = kc[:, t].unsqueeze(-1) # (N, D, 1)
            vi = vc[:, t].unsqueeze(-2) # (N, 1, D)
            
            # Transition matrix update: A_new = (I - gamma*k*k^T) A_old
            ki_T = ki.transpose(-1, -2)
            # (gamma * k) is (N, D, 1)
            g_k = gamma * ki
            
            # A_new = A_old - g_k @ k^T @ A_old
            update_a = g_k * torch.matmul(ki_T, a)
            a = a - update_a
            
            # State update (if starting from 0): S_new = (I - gamma*k*k^T) S_old + gamma*k*v^T
            update_s = g_k * torch.matmul(ki_T, s)
            s = s - update_s + g_k * vi
            
        all_final_states[:, c] = s
        all_transitions[:, c] = a
        
    # 3. Parallel Scan over chunks
    start_states = [torch.zeros(B * H, D, D, device=device, dtype=dtype)]
    curr_s = torch.zeros(B * H, D, D, device=device, dtype=dtype)
    
    for c in range(num_chunks - 1):
        # S_next = A_chunk @ S_curr + B_chunk
        curr_s = torch.matmul(all_transitions[:, c], curr_s) + all_final_states[:, c]
        start_states.append(curr_s)
        
    # 4. Final pass: Compute outputs within each chunk using start_states
    outputs = torch.empty(B * H, L, D, device=device, dtype=dtype)
    
    for c in range(num_chunks):
        state = start_states[c]
        qc = q_chunks[:, c]
        kc = k_chunks[:, c]
        vc = v_chunks[:, c]
        
        for t in range(chunk_size):
            qi = qc[:, t].unsqueeze(-2) # (N, 1, D)
            ki = kc[:, t].unsqueeze(-1) # (N, D, 1)
            vi = vc[:, t].unsqueeze(-2) # (N, 1, D)
            
            # Update state (same as inner loop above)
            ki_T = ki.transpose(-1, -2)
            g_k = gamma * ki
            
            # S_new = S_old + g_k (v^T - k^T S_old)
            # v_pred = k^T @ S_old
            v_pred = torch.matmul(ki_T, state) # (N, 1, D)
            delta = vi - v_pred
            state = state + g_k * delta
            
            # Read state: yt = q^T @ S
            yt = torch.matmul(qi, state).squeeze(1)
            outputs[:, c * chunk_size + t] = yt
            
    y = outputs.reshape(B, H, L, D).transpose(1, 2).contiguous()
    return y

if __name__ == "__main__":
    # Test for correctness
    from nvsubquadratic.ops.delta_rule import delta_rule_recursive
    
    B, L, H, D = 2, 64, 2, 32
    device = torch.device("cpu")
    dtype = torch.float64 # Use float64 for strict correctness check
    
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    beta = torch.rand(H, D, device=device, dtype=dtype) * 0.1 # Small positive beta
    
    # Key normalization as in DeltaHyena.forward
    eps = 1e-6
    k = k / (torch.norm(k, dim=-1, keepdim=True) + eps)
    
    y_rec = delta_rule_recursive(q, k, v, beta)
    y_par = delta_rule_parallel(q, k, v, beta, chunk_size=16)
    
    diff = (y_rec - y_par).abs().max().item()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-12
    print("Correctness check passed!")
