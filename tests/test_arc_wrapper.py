import pytest
import torch
import torch.nn as nn

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper


class MockNet(nn.Module):
    def __init__(self, num_colors=12):
        super().__init__()
        self.num_colors = num_colors

    def forward(self, batch):
        b, h, w = batch["input"].shape
        # Return a fixed logit prediction
        # Let's say all logits favor class 5
        logits = torch.randn((b, self.num_colors, h, w))
        # Zero out and set class 5 to high value
        logits.zero_()
        logits[:, 5, :, :] = 100.0
        return {"logits": logits}


def test_arc_wrapper_metrics():
    cfg = ExperimentConfig()
    net = MockNet(num_colors=12)
    wrapper = ARCWrapper(network=net, cfg=cfg)
    wrapper.eval()

    b, h, w = 2, 32, 32

    # 1. Test exact match logic
    input_tensor = torch.zeros((b, h, w))
    # Batch 0: all labels are 5 (will be predicted correctly by MockNet)
    # Batch 1: one label is 4 (will be wrong)
    labels = torch.ones((b, h, w), dtype=torch.long) * 5
    labels[1, 0, 0] = 4

    batch = {"input": input_tensor, "label": labels}

    with torch.no_grad():
        out = wrapper._step(batch)

    assert "loss" in out
    assert "pixel_acc" in out
    assert "exact_match" in out

    # Exact match: batch 0 is correct, batch 1 is wrong -> exact_match = 0.5
    assert torch.isclose(out["exact_match"], torch.tensor(0.5))

    # Pixel accuracy: (32*32*2 - 1) / (32*32*2) is correct
    correct_pixels = (h * w * 2) - 1
    total_pixels = h * w * 2
    assert torch.isclose(out["pixel_acc"], torch.tensor(correct_pixels / total_pixels))


def test_arc_wrapper_ignore_indices():
    cfg = ExperimentConfig()
    net = MockNet(num_colors=12)
    wrapper = ARCWrapper(network=net, cfg=cfg)
    wrapper.eval()

    b, h, w = 1, 4, 4
    input_tensor = torch.zeros((b, h, w))

    # Labels with 10(IGNORE) and 11(PAD)
    # 8 pixels of class 5 (correct)
    # 4 pixels of class 10 (ignored by loss and metrics)
    # 4 pixels of class 11 (ignored by loss and metrics)
    labels = torch.tensor([[5, 5, 5, 5], [5, 5, 5, 5], [10, 10, 10, 10], [11, 11, 11, 11]]).unsqueeze(0)

    batch = {"input": input_tensor, "label": labels}

    with torch.no_grad():
        out = wrapper._step(batch)

    # Since all valid pixels (5) are correct, exactly match should be 1.0
    assert torch.isclose(out["exact_match"], torch.tensor(1.0))
    # Pixel accuracy on valid pixels is 1.0
    assert torch.isclose(out["pixel_acc"], torch.tensor(1.0))
    # Loss should be ~0.0
    assert out["loss"] < 1e-4


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_arc_wrapper.py"])
