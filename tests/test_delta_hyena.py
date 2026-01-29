# TODO: Add license header here

import torch
import pytest
from nvsubquadratic.modules.delta_hyena import DeltaHyena
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND


def test_delta_hyena_forward():
    # Setup
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    height, width = 8, 8
    hidden_dim = 64
    num_heads = 4
    
    # Simple CKConv config for the value filter
    global_conv_cfg = LazyConfig(CKConvND)(
        data_dim=2,
        hidden_dim=hidden_dim,
        kernel_cfg=LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=hidden_dim,
            mlp_hidden_dim=16,
            num_layers=2,
            embedding_dim=16,
            omega_0=10.0,
            L_cache=8,
            use_bias=True,
        ),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type="double",
        fft_padding="zero",
    )
    
    # New DeltaHyena constructor requires more arguments due to inheritance
    model = DeltaHyena(
        global_conv_cfg=global_conv_cfg,
        short_conv_cfg=LazyConfig(torch.nn.Identity)(),
        gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
        pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
        apply_qk_norm=True,
        use_rope=False,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        gamma_init=0.1,
    ).to(device)
    
    # Inputs
    q = torch.randn(batch_size, height, width, hidden_dim).to(device)
    k = torch.randn(batch_size, height, width, hidden_dim).to(device)
    v = torch.randn(batch_size, height, width, hidden_dim).to(device)
    
    # Forward pass
    out = model(q, k, v)
    
    # Checks
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    print("Forward pass successful!")


def test_delta_hyena_memory_recall():
    # A simple memory recall test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    seq_len = 16
    hidden_dim = 8
    num_heads = 1 # Single head for simplicity
    
    model = DeltaHyena(
        global_conv_cfg=LazyConfig(torch.nn.Identity)(),
        short_conv_cfg=LazyConfig(torch.nn.Identity)(),
        gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
        pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
        apply_qk_norm=False,
        use_rope=False,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        gamma_init=1.0 # High learning rate for fast update
    ).to(device)
    
    # Let's say we want to store a specific vector 'pattern' at index 0
    pattern = torch.zeros(hidden_dim).to(device)
    pattern[0] = 1.0
    
    # Keys will be one-hot to select memory slots
    k = torch.zeros(batch_size, seq_len, hidden_dim).to(device)
    k[:, 0, 0] = 1.0 # Key for the first slot
    
    v = torch.zeros(batch_size, seq_len, hidden_dim).to(device)
    v[:, 0] = pattern # Value to store at the first slot
    
    # Queries: at index 5, we query for the first slot
    q = torch.zeros(batch_size, seq_len, hidden_dim).to(device)
    q[:, 5, 0] = 1.0
    
    # Forward
    out = model(q, k, v)
    
    # The output at index 5 should be close to 'pattern' if the delta rule worked
    recall_error = torch.norm(out[:, 5] - pattern)
    print(f"Recall error: {recall_error.item()}")
    
    # The error should be extremely low
    assert recall_error < 1e-4
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_delta_hyena_forward()
    test_delta_hyena_memory_recall()
