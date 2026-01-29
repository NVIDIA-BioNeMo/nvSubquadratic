
import torch
from nvsubquadratic.modules.delta_hyena import ReasoningDeltaHyena
from nvsubquadratic.lazy_config import LazyConfig

def test_reasoning_delta_hyena():
    print("Testing ReasoningDeltaHyena...")
    
    B, L, H, D = 2, 64, 8, 160
    hidden_dim = D
    
    # Mock configs
    global_conv_cfg = LazyConfig("torch.nn.Identity")()
    short_conv_cfg = LazyConfig("torch.nn.Identity")()
    gate_nonlinear_cfg = LazyConfig("torch.nn.Identity")()
    pixelhyena_norm_cfg = LazyConfig("torch.nn.Identity")()
    output_norm_cfg = LazyConfig("torch.nn.Identity")()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReasoningDeltaHyena(
        global_conv_cfg=global_conv_cfg,
        short_conv_cfg=short_conv_cfg,
        gate_nonlinear_cfg=gate_nonlinear_cfg,
        pixelhyena_norm_cfg=pixelhyena_norm_cfg,
        apply_qk_norm=True,
        use_rope=True,
        hidden_dim=hidden_dim,
        num_heads=H,
        num_recurrence=3
    ).to(device)
    
    q = torch.randn(B, L, D).to(device)
    k = torch.randn(B, L, D).to(device)
    v = torch.randn(B, L, D).to(device)
    
    # Test forward
    model.eval()
    with torch.no_grad():
        out = model(q, k, v)
        
    print(f"Forward pass successful! Output shape: {out.shape}")
    assert out.shape == (B, L, D)
    
    # Test gradients through recurrences
    model.train()
    out = model(q, k, v)
    loss = out.sum()
    loss.backward()
    
    print("Backward pass successful!")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item():.4e}")
            assert not torch.isnan(param.grad).any()
            
    print("Reasoning Delta-Hyena test passed!")

if __name__ == "__main__":
    test_reasoning_delta_hyena()
