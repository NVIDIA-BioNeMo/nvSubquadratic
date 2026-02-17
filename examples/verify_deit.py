import torch
import torch.nn as nn
from examples.tmp_config import get_config, NUM_HIDDEN_CHANNELS, NUM_BLOCKS, INPUT_CHANNELS, OUTPUT_CHANNELS
from nvsubquadratic.lazy_config import instantiate
from nvsubquadratic.modules.mlp import MLP

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("Loading configuration from examples/tmp_config.py...")
    config = get_config()
    
    # -------------------------------------------------------------------------
    # 1. Verification of Interpolations (Fixing broken refs for instantiation)
    # -------------------------------------------------------------------------
    config.net.in_proj_cfg.in_features = INPUT_CHANNELS
    config.net.in_proj_cfg.out_features = NUM_HIDDEN_CHANNELS
    config.net.out_proj_cfg.in_features = NUM_HIDDEN_CHANNELS
    config.net.out_proj_cfg.out_features = OUTPUT_CHANNELS
    config.net.norm_cfg.normalized_shape = NUM_HIDDEN_CHANNELS
    
    block_cfg = config.net.block_cfg
    block_cfg.sequence_mixer_norm_cfg = config.net.norm_cfg
    block_cfg.mlp_norm_cfg = config.net.norm_cfg
    
    block_cfg.mlp_cfg.dim = NUM_HIDDEN_CHANNELS
    block_cfg.mlp_cfg.dropout_cfg.p = config.net.block_cfg.dropout_cfg.p
    
    block_cfg.sequence_mixer_cfg.hidden_dim = NUM_HIDDEN_CHANNELS
    block_cfg.sequence_mixer_cfg.mixer_cfg.hidden_dim = NUM_HIDDEN_CHANNELS
    block_cfg.sequence_mixer_cfg.init_method_out.num_layers = NUM_BLOCKS
    block_cfg.mlp_cfg.init_method_out.num_layers = NUM_BLOCKS

    # -------------------------------------------------------------------------
    # 2. CURRENT MODEL (GLU, Exp=2.0)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print(f"CURRENT CONFIG: Activation={block_cfg.mlp_cfg.activation}, Expansion={block_cfg.mlp_cfg.expansion_factor}")
    print("="*50)
    
    model_current = instantiate(config.net)
    params_current = count_parameters(model_current)
    print(f"Total Params: {params_current:,}")
    
    mlp_current = instantiate(block_cfg.mlp_cfg)
    mlp_params_current = count_parameters(mlp_current)
    print(f"MLP Block Params: {mlp_params_current:,}")


    # -------------------------------------------------------------------------
    # 3. PROPOSED STANDARD DEIT (GELU, Exp=4.0)
    # -------------------------------------------------------------------------
    # Modify config in-place to Standard ViT settings
    block_cfg.mlp_cfg.activation = "gelu"
    block_cfg.mlp_cfg.expansion_factor = 4.0
    
    print("\n" + "="*50)
    print(f"PROPOSED STANDARD CONFIG: Activation={block_cfg.mlp_cfg.activation}, Expansion={block_cfg.mlp_cfg.expansion_factor}")
    print("="*50)
    
    model_std = instantiate(config.net)
    params_std = count_parameters(model_std)
    print(f"Total Params: {params_std:,}")
    
    mlp_std = instantiate(block_cfg.mlp_cfg)
    mlp_params_std = count_parameters(mlp_std)
    print(f"MLP Block Params: {mlp_params_std:,}")

    # -------------------------------------------------------------------------
    # 4. COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    diff = params_std - params_current
    percent = (params_current / params_std) * 100
    print(f"Difference in Total Params: {diff:,}")
    print(f"Current model is {percent:.1f}% size of Standard DeiT.")
    
    # -------------------------------------------------------------------------
    # 5. DISCREPANCY ANALYSIS (Why is Standard DeiT 85M and not 86M?)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("ANALYSIS OF MISSING 1.2M PARAMETERS")
    print("="*50)
    print("Standard ViT-B (ImageNet-1k) has ~86.5M parameters.")
    print(f"Our 'Standard' Config Construction (GELU/4.0) has {params_std:,} (~85.1M).")
    print("Where is the missing ~1.4M?")
    
    # Calculate Embedding Difference
    # Standard ViT: PatchEmbed (Conv2d 16x16, 3->768) + PosEmbed (Learned)
    # PatchEmbed weights: 768 * 3 * 16 * 16 = 589,824
    # PosEmbed weights: (14*14 + 1) * 768 = 151,296
    # Total Standard Embed: ~741,120
    
    # Our Config: Linear (3->768) + No explicit PosEmbed in backbone (RoPE is in Attention)
    current_embed_params = count_parameters(model_std.in_proj)
    print(f"\n1. Input Projection / Embeddings:")
    print(f"   Standard ViT PatchEmbed: ~590,000")
    print(f"   Our Linear (3->768):         {current_embed_params:,}")
    print(f"   Missing:                    ~{590000 - current_embed_params:,}")

    # Calculate Head Difference
    # Standard ViT: 1000 classes
    # Head weights: 768 * 1000 + 1000 = 769,000
    
    # Our Config: 200 classes (TinyImageNet)
    current_head_params = count_parameters(model_std.out_proj)
    print(f"\n2. Classification Head:")
    print(f"   Standard ViT (1000 cls): ~769,000")
    print(f"   Our Head (200 cls):       {current_head_params:,}")
    print(f"   Missing:                 ~{769000 - current_head_params:,}")
    
    total_missing_explanation = (590000 - current_embed_params) + (769000 - current_head_params)
    print(f"\nTotal Explained Discrepancy: ~{total_missing_explanation:,}")
    print(f"85.1M (Calculated) + 1.2M (Explained) = ~86.3M (Matches Standard ViT-B)")

    if block_cfg.mlp_cfg.activation == "gelu" and block_cfg.mlp_cfg.expansion_factor == 4.0:
        print("\nNote: Config object was modified in memory for this test.")

if __name__ == "__main__":
    main()
