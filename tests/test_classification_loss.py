import os
import sys
import warnings

import torch


# Suppress PyTorch Lightning warnings for cleaner output
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

from experiments.default_cfg import ExperimentConfig  # noqa: E402
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper  # noqa: E402


# Mock Network
class MockNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.out_proj = torch.nn.Linear(10, num_classes)

    def forward(self, input_and_condition):
        # Extract input from the dictionary passed by ClassificationWrapper
        x = input_and_condition["input"]
        return {"logits": self.out_proj(x)}


def test_bce_loss():
    print("Testing BCE Loss implementation...")
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)

    # ---------------------------------------------------------
    # Test 1: Initialize with use_bce_loss=True
    # ---------------------------------------------------------
    wrapper_bce = ClassificationWrapper(net, cfg, use_bce_loss=True)

    # Verify the loss metric choice
    assert isinstance(wrapper_bce.loss_metric, torch.nn.BCEWithLogitsLoss), (
        "Expected BCEWithLogitsLoss when use_bce_loss=True"
    )
    print("[PASS] Loss metric is correctly set to BCEWithLogitsLoss.")

    # ---------------------------------------------------------
    # Test 2: Initialize with use_bce_loss=False (Default)
    # ---------------------------------------------------------
    wrapper_ce = ClassificationWrapper(net, cfg, use_bce_loss=False)

    # Verify the loss metric choice
    assert isinstance(wrapper_ce.loss_metric, torch.nn.CrossEntropyLoss), (
        "Expected CrossEntropyLoss when use_bce_loss=False"
    )
    print("[PASS] Loss metric is correctly set to CrossEntropyLoss.")
    print("All verification steps passed!")


if __name__ == "__main__":
    test_bce_loss()
