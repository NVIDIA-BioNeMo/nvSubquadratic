# TODO: Add license header here

import torch
from einops import rearrange, repeat


def delta_rule_recursive(q, k, v, beta):
    """Recursive implementation of the Delta Rule update.
    
    Args:
        q: (B, L, H, D) - Query tokens
        k: (B, L, H, D) - Key tokens (already normalized)
        v: (B, L, H, D) - Value tokens
        beta: (H, D) or (H, 1) - Learning rate
        
    Returns:
        y: (B, L, H, D) - Output tokens
    """
    B, L, H, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # Flatten batch and heads for simpler indexing
    q = q.transpose(1, 2).reshape(B * H, L, D)
    k = k.transpose(1, 2).reshape(B * H, L, D)
    v = v.transpose(1, 2).reshape(B * H, L, D)
    
    # State: (B*H, D, D)
    state = torch.zeros(B * H, D, D, device=device, dtype=dtype)
    y = torch.empty(B * H, L, D, device=device, dtype=dtype)
    
    # beta: (H, D) -> (B*H, D, 1) for broadcasting over (B*H, D, D)
    gamma = beta.unsqueeze(-1).repeat(B, 1, 1).contiguous()

    for t in range(L):
        # (B*H, D)
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        
        # v_pred = kt @ state -> (B*H, D)
        # Using element-wise multiplication and sum to avoid cuBLAS entirely
        v_pred = torch.sum(kt.unsqueeze(-1) * state, dim=1)
        
        # update = outer product (kt.T @ delta) -> (B*H, D, D)
        delta = vt - v_pred
        update = kt.unsqueeze(-1) * delta.unsqueeze(-2)
        
        # state update
        state = state + gamma * update
        
        # read: (B*H, D)
        yt = torch.sum(qt.unsqueeze(-1) * state, dim=1)
        y[:, t] = yt
        
    # Reshape back to (B, L, H, D)
    y = y.reshape(B, H, L, D).transpose(1, 2).contiguous()
    return y



def delta_rule_parallel(q, k, v, beta, chunk_size=64, initial_state=None):
    """Parallel implementation of the Delta Rule using associative chunked scan.
    
    Formula: S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    """
    B, L, H, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # 1. Reshape and setup
    q_orig = q
    target_H, target_D = H, D
    # Flatten batch and heads for simpler indexing inside
    q = q.transpose(1, 2).reshape(B * H, L, D)  # (N, L, D)
    k = k.transpose(1, 2).reshape(B * H, L, D)
    v = v.transpose(1, 2).reshape(B * H, L, D)
    
    # beta: (H, D) -> (B*H, D, 1) to match (N, D, D) state broadcasting
    gamma = beta.unsqueeze(-1).repeat(B, 1, 1).contiguous()

    # Padding if L is not divisible by chunk_size
    pad_len = (chunk_size - (L % chunk_size)) % chunk_size
    if pad_len > 0:
        q = torch.cat([q, q.new_zeros(B * H, pad_len, D)], dim=1)
        k = torch.cat([k, k.new_zeros(B * H, pad_len, D)], dim=1)
        v = torch.cat([v, v.new_zeros(B * H, pad_len, D)], dim=1)
    
    total_L = L + pad_len
    num_chunks = total_L // chunk_size
    
    q_chunks = q.reshape(B * H, num_chunks, chunk_size, D)
    k_chunks = k.reshape(B * H, num_chunks, chunk_size, D)
    v_chunks = v.reshape(B * H, num_chunks, chunk_size, D)
    
    # 2. Compute Chunk Parameters (Transitions and Final States)
    all_final_states = torch.zeros(B * H, num_chunks, D, D, device=device, dtype=dtype)
    all_transitions = torch.zeros(B * H, num_chunks, D, D, device=device, dtype=dtype)
    
    # Sequence of chunk parameters
    for c in range(num_chunks):
        s = torch.zeros(B * H, D, D, device=device, dtype=dtype)
        a = torch.eye(D, device=device, dtype=dtype).expand(B * H, D, D).contiguous()
        
        kc = k_chunks[:, c]
        vc = v_chunks[:, c]
        
        # Inner sequential loop (short, fixed length)
        for t in range(chunk_size):
            ki = kc[:, t].unsqueeze(-1)  # (N, D, 1)
            vi = vc[:, t].unsqueeze(-2)  # (N, 1, D)
            g_k = gamma * ki             # (N, D, 1)
            
            # A_new = A_old - (gamma * k) @ (k^T @ A_old)
            # Use element-wise sum to avoid cuBLAS: (N, 1, D)
            k_T_a = torch.sum(ki * a, dim=1, keepdim=True)
            a = a - g_k * k_T_a
            
            # S_new = S_old - (gamma * k) @ (k^T @ S_old) + (gamma * k) @ v^T
            k_T_s = torch.sum(ki * s, dim=1, keepdim=True)
            s = s - g_k * k_T_s + g_k * vi
            
        all_final_states[:, c] = s
        all_transitions[:, c] = a
        
    # 3. Parallel Scan (Prefix Sum) over chunk parameters
    start_states = torch.zeros(B * H, num_chunks, D, D, device=device, dtype=dtype)
    
    if initial_state is not None:
        curr_s = initial_state
    else:
        curr_s = torch.zeros(B * H, D, D, device=device, dtype=dtype)
    
    for c in range(num_chunks):
        start_states[:, c] = curr_s
        # S_next = A_chunk @ S_curr + B_chunk
        # Full matrix-matrix multiplication via broadcasting
        # (N, D, D) @ (N, D, D) -> (N, D, D)
        trans = all_transitions[:, c]
        # (N, D, D, 1) * (N, 1, D, D) -> (N, D, D, D) -> sum over k -> (N, D, D)
        curr_s = torch.sum(trans.unsqueeze(-1) * curr_s.unsqueeze(-3), dim=-2) + all_final_states[:, c]
        
    final_state = curr_s # Return the state after all chunks
        
    # 4. Compute Final Outputs
    y = torch.empty(B * H, total_L, D, device=device, dtype=dtype)
    for c in range(num_chunks):
        state = start_states[:, c]
        qc = q_chunks[:, c]
        kc = k_chunks[:, c]
        vc = v_chunks[:, c]
        
        for t in range(chunk_size):
            qi = qc[:, t].unsqueeze(-2)  # (N, 1, D)
            ki = kc[:, t].unsqueeze(-1)  # (N, D, 1)
            vi = vc[:, t].unsqueeze(-2)  # (N, 1, D)
            g_k = gamma * ki             # (N, D, 1)
            
            # Update state: v_pred = k^T @ S_old
            v_pred = torch.sum(ki * state, dim=1, keepdim=True)
            state = state + g_k * (vi - v_pred)
            
            # Read state: yt = q^T @ S
            yt = torch.sum(qi.transpose(-1, -2) * state, dim=1)
            y[:, c * chunk_size + t] = yt
            
    y = y[:, :L]
    y = y.reshape(B, target_H, L, target_D).transpose(1, 2).contiguous()
    return y, final_state


def delta_rule_scan(q, k, v, beta):
    """Wrapper for Delta Rule, uses parallel version for speed if L is large."""
    B, L, H, D = q.shape
    if L > 128:
        y, _ = delta_rule_parallel(q, k, v, beta)
        return y
    else:
        return delta_rule_recursive(q, k, v, beta)


def reasoning_delta_rule_scan(q, k, v, beta, initial_state=None):
    """Wrapper for Reasoning Delta Rule, uses parallel version for speed if L is large."""
    B, L, H, D = q.shape
    if L > 128:
        return delta_rule_parallel(q, k, v, beta, initial_state=initial_state)
    else:
        return reasoning_delta_rule_recursive(q, k, v, beta, initial_state=initial_state)


def reasoning_delta_rule_recursive(q, k, v, beta, initial_state=None):
    """Recursive implementation of the Delta Rule update with persistent state.
    
    Args:
        q: (B, L, H, D) - Query tokens
        k: (B, L, H, D) - Key tokens (already normalized)
        v: (B, L, H, D) - Value tokens
        beta: (H, D) or (H, 1) - Learning rate
        initial_state: (B*H, D, D) - Optional initial memory state
        
    Returns:
        y: (B, L, H, D) - Output tokens
        state: (B*H, D, D) - Final memory state
    """
    B, L, H, D = q.shape
    device = q.device
    dtype = q.dtype
    
    # Flatten batch and heads for simpler indexing
    q = q.transpose(1, 2).reshape(B * H, L, D)
    k = k.transpose(1, 2).reshape(B * H, L, D)
    v = v.transpose(1, 2).reshape(B * H, L, D)
    
    # State: (B*H, D, D)
    if initial_state is not None:
        state = initial_state
    else:
        state = torch.zeros(B * H, D, D, device=device, dtype=dtype)
        
    y = torch.empty(B * H, L, D, device=device, dtype=dtype)
    
    # beta: (H, D) -> (B*H, D, 1) for broadcasting over (B*H, D, D)
    gamma = beta.unsqueeze(-1).repeat(B, 1, 1).contiguous()

    for t in range(L):
        # (B*H, D)
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        
        # v_pred = kt @ state -> (B*H, D)
        v_pred = torch.sum(kt.unsqueeze(-1) * state, dim=1)
        
        # update = outer product (kt.T @ delta) -> (B*H, D, D)
        delta = vt - v_pred
        update = kt.unsqueeze(-1) * delta.unsqueeze(-2)
        
        # state update
        state = state + gamma * update
        
        # read: (B*H, D)
        yt = torch.sum(qt.unsqueeze(-1) * state, dim=1)
        y[:, t] = yt
        
    # Reshape back
    y = y.reshape(B, H, L, D).transpose(1, 2).contiguous()
    return y, state
