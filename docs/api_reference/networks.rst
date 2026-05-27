.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Networks
========

End-to-end classification and general-purpose networks composing the
modules above, plus the diffusion backbones and the UNet-ConvNeXt
baselines used in benchmark comparisons.

Classification & general-purpose
--------------------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.classification_resnet.ClassificationResNet
   ~networks.general_purpose_resnet.ResidualNetwork
   ~networks.vit5_classification.ViT5ClassificationNet

Diffusion — Hugging Face adapters
---------------------------------

Wrappers that expose :class:`diffusers.DiTTransformer2DModel` and
:class:`diffusers.UVit2DModel` to the diffusion Lightning wrapper.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.huggingface_diffusers.HuggingFaceDiTConfig
   ~networks.huggingface_diffusers.HuggingFaceUVitConfig
   ~networks.huggingface_diffusers.DiffusersDiTWrapper
   ~networks.huggingface_diffusers.DiffusersUVitWrapper

Diffusion — JiT backbone
------------------------

Port of the JiT diffusion model (`LTH14/JiT <https://github.com/LTH14/JiT>`_) —
patch-embedding, transformer blocks with RoPE and SwiGLU FFN, and the
factory functions for the published model sizes.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.jit.JiT
   ~networks.jit.JiTBlock
   ~networks.jit.BottleneckPatchEmbed
   ~networks.jit.TimestepEmbedder
   ~networks.jit.LabelEmbedder
   ~networks.jit.Attention
   ~networks.jit.SwiGLUFFN
   ~networks.jit.FinalLayer

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~networks.jit.modulate
   ~networks.jit.JiT_B_4
   ~networks.jit.JiT_B_16
   ~networks.jit.JiT_B_32
   ~networks.jit.JiT_L_16
   ~networks.jit.JiT_L_32
   ~networks.jit.JiT_H_16
   ~networks.jit.JiT_H_32

JiT helpers (rotary embeddings, RMSNorm, sin-cos position embeddings):

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.jit_utils.VisionRotaryEmbedding
   ~networks.jit_utils.VisionRotaryEmbeddingFast
   ~networks.jit_utils.RMSNorm

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~networks.jit_utils.broadcat
   ~networks.jit_utils.rotate_half
   ~networks.jit_utils.get_1d_sincos_pos_embed_from_grid
   ~networks.jit_utils.get_2d_sincos_pos_embed
   ~networks.jit_utils.get_2d_sincos_pos_embed_from_grid

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
