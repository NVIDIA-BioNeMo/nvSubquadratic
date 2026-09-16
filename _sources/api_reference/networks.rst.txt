.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Networks
========

End-to-end classification and general-purpose networks composing the
modules above, plus the UNet-ConvNeXt baselines used in benchmark
comparisons.

Classification & general-purpose
--------------------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.classification_resnet.ClassificationResNet
   ~networks.general_purpose_resnet.ResidualNetwork
   ~networks.vit5_classification.ViT5ClassificationNet

Baselines
---------

UNet-ConvNeXt baselines ported from
`The Well <https://github.com/PolymathicAI/the_well>`_, used as PDE-task
reference points.  :class:`UNetConvNextV2` fixes the upstream
finest-skip bug; see the module docstring for the diff.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.baselines.unet_convnext.UNetConvNext
   ~networks.baselines.unet_convnext.WellUNetConvNext
   ~networks.baselines.unet_convnext_v2.UNetConvNextV2
   ~networks.baselines.unet_convnext_v2.WellUNetConvNextV2
