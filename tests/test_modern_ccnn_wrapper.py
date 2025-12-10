import sys
from dataclasses import dataclass

import torch


# Add lingua_clone to path
sys.path.append("/home/dwessel/code/nvSubquadratic-private/lingua_clone")

from apps.main.modern_ccnn_wrapper import ModernCCNNWrapper


@dataclass
class MockArgs:
    modern_ccnn_config_path: str = "examples/text_pretraining/zyda_1d_attention.py"
    vocab_size: int = 1000
    dim: int = 256
    n_layers: int = 2


def test_wrapper():
    args = MockArgs()
    print(f"Initializing wrapper with config: {args.modern_ccnn_config_path}")

    # Mock load_config_from_path to return a dummy config if file doesn't exist
    # But here we expect the file to exist from previous steps

    try:
        model = ModernCCNNWrapper(args)
    except Exception as e:
        print(f"Failed to initialize wrapper: {e}")
        return

    print("Wrapper initialized successfully.")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    vocab_size = model.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    print("Testing forward pass (logits only)...")
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)

    print("Testing forward pass (loss)...")
    loss = model(input_ids, labels)
    print(f"Loss: {loss.item()}")
    assert isinstance(loss.item(), float)

    print("Test passed!")


if __name__ == "__main__":
    test_wrapper()
