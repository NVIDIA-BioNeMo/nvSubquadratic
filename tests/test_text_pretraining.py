"""Test TextPretrainingWrapper."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from experiments.lightning_wrappers.text_pretraining_wrapper import TextPretrainingWrapper

class DummyModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        return self.proj(x)

def test_text_pretraining_wrapper():
    """Test TextPretrainingWrapper."""
    vocab_size = 100
    model = DummyModel(vocab_size=vocab_size)
    
    # Mock config
    cfg = OmegaConf.create({
        "model": {"vocab_size": vocab_size},
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 0.1, "__target__": "torch.optim.AdamW"},
        "scheduler": {"name": "cosine", "total_iterations": 100, "warmup_iterations_percentage": 0.1},
        "train": {"track_grad_norm": -1}
    })
    
    wrapper = TextPretrainingWrapper(model, cfg)
    
    # Mock batch
    batch_size = 2
    seq_len = 10
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len))
    }
    
    # Test training step
    loss = wrapper.training_step(batch, 0)
    print("Training loss:", loss.item())
    
    # Test validation step
    val_loss = wrapper.validation_step(batch, 0)
    print("Validation loss:", val_loss.item())
    
    assert loss > 0
    assert val_loss > 0

if __name__ == "__main__":
    test_text_pretraining_wrapper()
