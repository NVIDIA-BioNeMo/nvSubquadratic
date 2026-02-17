import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from nvsubquadratic.lazy_config import instantiate
from nvsubquadratic.modules.mlp import MLP

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_config(config_module_name, label):
    print(f"\n--- Verifying {label} ---")
    try:
        module = __import__(config_module_name, fromlist=['get_config'])
        config = module.get_config()
        
        # Instantiate net
        # We might need to handle interpolations if they exist, but let's try direct instantiation first
        # Usually lazy_config.instantiate handles the interpolations if the config is structured correctly
        # However, our previous script showed we might need manual help if interpolation uses 'net.*' 
        # But get_config() usually returns a clean config.
        # The previous failure was because get_config() wasn't fully resolved or something?
        # Actually, let's just inspect the config values directly first.
        
        mlp_cfg = config.net.block_cfg.mlp_cfg
        print(f"Activation: {mlp_cfg.activation}")
        print(f"Expansion Factor: {mlp_cfg.expansion_factor}")
        
        if mlp_cfg.activation == 'gelu' and mlp_cfg.expansion_factor == 4.0:
            print("STATUS: PASS (Aligned with DeiT Standard)")
        else:
            print("STATUS: FAIL (Mismatch)")
            
    except Exception as e:
        print(f"STATUS: ERROR ({e})")

def main():
    configs_to_check = [
        ("examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.attention", "Attention (TinyImageNet)"),
        ("examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.attention_patchify", "Attention Patchify (TinyImageNet)"),
        ("examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.imagenette_attention_patchify", "Attention Patchify (Imagenette)"),
    ]
    
    for module, label in configs_to_check:
        verify_config(module, label)

if __name__ == "__main__":
    main()
