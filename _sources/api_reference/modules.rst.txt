.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. currentmodule:: nvsubquadratic

Modules
=======

High-level ``torch.nn.Module`` building blocks that compose the ops above
into sequence and spatial mixers, plus the kernels, norms, gates, and
residual blocks they rely on.

Mixers
------

Sequence/spatial mixers: Hyena, Mamba, attention variants.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.hyena_nd.Hyena
   ~modules.mamba_nd.Mamba
   ~modules.attention.Attention
   ~modules.vit5_attention.ViT5Attention
   ~modules.vit5_hyena_adapter.ViT5HyenaAdapter
   ~modules.sequence_mixer.QKVSequenceMixer

Convolutions
------------

Depthwise, multi-head, and continuous-kernel convolutions plus their
context-parallel counterparts.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.causal_conv1d.CausalConv1D
   ~modules.subq_ops_causal_conv1d.SubqOpsCausalConv1d
   ~modules.ckconv_nd.CKConvND
   ~modules.distributed_depthwise_conv_nd.DistributedDepthwiseConv1d
   ~modules.distributed_depthwise_conv_nd.DistributedDepthwiseConv2d
   ~modules.distributed_depthwise_conv_nd.DistributedDepthwiseConv3d

Kernels & filters
-----------------

Learned kernel parametrisations (SIREN, random Fourier features) and
masks that produce the filters consumed by the FFT ops.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.kernels_nd.SIRENKernelND
   ~modules.kernels_nd.SIRENPositionalEmbeddingND
   ~modules.kernels_nd.MultiOmegaSIRENKernelND
   ~modules.kernels_nd.MultiOmegaSIRENPositionalEmbeddingND
   ~modules.kernels_nd.BlockDiagonalMultiOmegaSIRENKernelND
   ~modules.kernels_nd.LearnableOmegaSIRENKernelND
   ~modules.kernels_nd.LearnableOmegaSIRENPositionalEmbeddingND
   ~modules.kernels_nd.BlockDiagonalLearnableOmegaSIRENKernelND
   ~modules.kernels_nd.RandomFourierKernelND
   ~modules.kernels_nd.RandomFourierPositionalEmbeddingND
   ~modules.kernels_nd.Sine
   ~modules.masks_nd.ExponentialModulationND
   ~modules.masks_nd.GaussianModulationND
   ~modules.masks_nd.BlockAlignedGaussianModulationND

Normalization
-------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.rms_norm.RMSNorm
   ~modules.rms_norm.PerHeadRMSNorm
   ~modules.rms_norm_channel_first.RMSNormChannelFirst
   ~modules.grn.GlobalResponseNorm
   ~modules.layer_scale.LayerScale

Position encoding & patching
----------------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.position_encoding.PositionEmbeddingND
   ~modules.patchify.Patchify
   ~modules.patchify.Unpatchify
   ~modules.mlp.MLP

Gating & conditioning
---------------------

Drop-path, FiLM-style conditioning, and the QKV conditioning mixer that
feeds Hyena's per-sample kernels.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.drop_path.DropPath
   ~modules.condition_mixer.QKVConditionMixer
   ~modules.film.KernelFiLMGenerator
   ~modules.film.RegisterPooling
   ~modules.film.RegisterCompressConcat

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~modules.drop_path.drop_path

Residual blocks
---------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.residual_block.ResidualBlock
   ~modules.residual_block.AdaLNZeroResidualBlock
   ~modules.vit5_residual_block.ViT5ResidualBlock

Schedulers
----------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.schedulers.ResumableSequentialLR
