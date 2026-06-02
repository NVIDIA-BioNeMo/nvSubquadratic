# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import warnings

import torch

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.classification_wrapper import (
    ClassificationWrapper,
    SoftTargetCrossEntropy,
)


# Suppress PyTorch Lightning warnings for cleaner output
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())


# Mock Network
class MockNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.out_proj = torch.nn.Linear(10, num_classes)

    def forward(self, input_and_condition):
        x = input_and_condition["input"]
        return {"logits": self.out_proj(x)}


def test_bce_loss():
    print("Testing loss= parameter...")
    cfg = ExperimentConfig()
    net = MockNet(num_classes=10)

    # Test 1: loss="bce" → BCEWithLogitsLoss
    wrapper_bce = ClassificationWrapper(net, cfg, loss="bce")
    assert isinstance(wrapper_bce.loss_metric, torch.nn.BCEWithLogitsLoss), (
        "Expected BCEWithLogitsLoss when loss='bce'"
    )
    print("[PASS] loss='bce' → BCEWithLogitsLoss")

    # Test 2: loss="soft_target_ce" → SoftTargetCrossEntropy
    wrapper_soft = ClassificationWrapper(net, cfg, loss="soft_target_ce")
    assert isinstance(wrapper_soft.loss_metric, SoftTargetCrossEntropy), (
        "Expected SoftTargetCrossEntropy when loss='soft_target_ce'"
    )
    print("[PASS] loss='soft_target_ce' → SoftTargetCrossEntropy")

    # Test 3: loss="cross_entropy" (default) → CrossEntropyLoss
    wrapper_ce = ClassificationWrapper(net, cfg, loss="cross_entropy")
    assert isinstance(wrapper_ce.loss_metric, torch.nn.CrossEntropyLoss), (
        "Expected CrossEntropyLoss when loss='cross_entropy'"
    )
    print("[PASS] loss='cross_entropy' → CrossEntropyLoss")

    # Test 4: invalid loss raises ValueError
    try:
        ClassificationWrapper(net, cfg, loss="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[PASS] Invalid loss raises ValueError")

    print("All verification steps passed!")


if __name__ == "__main__":
    test_bce_loss()
