.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

.. module:: nvsubquadratic
.. currentmodule:: nvsubquadratic

API Reference
=============

The reference is organised bottom-up: low-level FFT convolution primitives
first, then the mixer modules that compose them. See [`docs/ops/README.md`](ops/README.md)
for the math motivation behind the FFT-based ops, and
``docs-tracker.md`` at the repo root for the documentation coverage plan.

Ops — FFT convolutions (reference fp32)
---------------------------------------

Reference implementations in pure PyTorch FFT. Use these for correctness
and as the spec the CUDA kernels must match.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv.fftconv1d_fp32_blh
   ~ops.fftconv.fftconv2d_fp32_blh
   ~ops.fftconv.fftconv3d_fp32_blh
   ~ops.fftconv.causal_fftconv1d_fp32_blh
   ~ops.fftconv.fftconv1d_fp32_bhl
   ~ops.fftconv.fftconv2d_fp32_bhl
   ~ops.fftconv.fftconv3d_fp32_bhl
   ~ops.fftconv.causal_fftconv1d_fp32_bhl

Ops — FFT convolutions (CUDA-accelerated)
-----------------------------------------

Drop-in wrappers around the ``subquadratic_ops_torch`` fused CUDA kernels.
2D non-causal and 1D causal long-conv variants share the same API as the
fp32 reference ops above.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_custom.fftconv2d_blh
   ~ops.fftconv_custom.fftconv2d_bhl
   ~ops.fftconv_custom.fftconv2d_bhl_w_reshape
   ~ops.fftconv_custom.causal_fftconv1d_blh
   ~ops.fftconv_custom.causal_fftconv1d_bhl
   ~ops.fftconv_custom.causal_fftconv1d_bhl_w_reshape

Ops — Direct 1D causal convolutions (CUDA-accelerated)
------------------------------------------------------

Non-FFT CUDA kernels for short and fused 1D causal convolutions. Useful for
small kernel sizes (where FFT overhead dominates) and as building blocks
for fused Hyena variants.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.causal_conv1d_custom.causal_conv1d
   ~ops.causal_conv1d_custom.b2b_causal_conv1d

Ops — Circular FFT convolutions
-------------------------------

Periodic-boundary FFT convolutions for global mixing without zero padding.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.circular_fftconv.circular_fftconv1d_fp32_bhl
   ~ops.circular_fftconv.circular_fftconv2d_fp32_bhl
   ~ops.circular_fftconv.circular_fftconv3d_fp32_bhl

Ops — Multi-head FFT convolutions
---------------------------------

Multi-head variants used by Hyena-style mixers, including low-rank
factorizations.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_multihead.fftconv2d_multihead_bhl
   ~ops.fftconv_multihead.fftconv2d_multihead_lowrank_bhl
   ~ops.fftconv_multihead.fftconv2d_multihead_circular_bhl
   ~ops.fftconv_multihead.fftconv2d_multihead_lowrank_circular_bhl

Ops — Chunking utilities
------------------------

Helpers to bound the FFT working-set memory by processing along the
sequence axis in chunks.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.fftconv_chunked.enable_chunking
   ~ops.fftconv_chunked.chunking_enabled
   ~ops.fftconv_chunked.set_default_chunk_size
   ~ops.fftconv_chunked.get_default_chunk_size

Ops — Mixed-precision FFT convolutions
--------------------------------------

FFT convolutions that switch internal precision per-axis (e.g. fp16 on
power-of-2 dims, fp32 on others).  See the
`FP16 Circular FFT Convolution: Derivation <ops/FP16_FFTCONV_DERIVATION.html>`_
for the numerical-stability background.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~ops.mixed_fftconv.mixed_fftconv1d_fp32_bhl
   ~ops.mixed_fftconv.mixed_fftconv2d_fp32_bhl
   ~ops.mixed_fftconv.mixed_fftconv3d_fp32_bhl
   ~ops.mixed_fftconv.mixed_fftconv1d_fp32_bhl_w_reshape
   ~ops.mixed_fftconv.mixed_fftconv2d_fp32_bhl_w_reshape
   ~ops.mixed_fftconv.mixed_fftconv3d_fp32_bhl_w_reshape
   ~ops.mixed_fftconv.mixed_fftconv1d_fp32_bhl_chunked
   ~ops.mixed_fftconv.mixed_fftconv2d_fp32_bhl_chunked
   ~ops.mixed_fftconv.mixed_fftconv3d_fp32_bhl_chunked

Modules — Mixers
----------------

High-level PyTorch ``nn.Module`` sequence/spatial mixers.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.hyena_nd.Hyena
   ~modules.mamba_nd.Mamba
   ~modules.attention.Attention
   ~modules.vit5_attention.ViT5Attention
   ~modules.vit5_hyena_adapter.ViT5HyenaAdapter
   ~modules.sequence_mixer.QKVSequenceMixer

Modules — Convolutions
----------------------

Depthwise, multi-head, and continuous-kernel convolutions plus their
context-parallel counterparts.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.causal_conv1d.CausalConv1D
   ~modules.subq_ops_causal_conv1d.SubqOpsCausalConv1d
   ~modules.ckconv_nd.CKConvND
   ~modules.ckconv_multihead_nd.CKConvMultiheadND
   ~modules.distributed_depthwise_conv_nd.DistributedDepthwiseConv1d
   ~modules.distributed_depthwise_conv_nd.DistributedDepthwiseConv2d
   ~modules.distributed_depthwise_conv_nd.DistributedDepthwiseConv3d

Modules — Kernels & filters
---------------------------

Learned kernel parametrisations (SIREN, random Fourier features) and
masks that produce the filters consumed by the FFT ops above.

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

Modules — Normalization
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.rms_norm.RMSNorm
   ~modules.rms_norm.PerHeadRMSNorm
   ~modules.rms_norm_channel_first.RMSNormChannelFirst
   ~modules.grn.GlobalResponseNorm
   ~modules.layer_scale.LayerScale

Modules — Position encoding & patching
--------------------------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.position_encoding.PositionEmbeddingND
   ~modules.patchify.Patchify
   ~modules.patchify.Unpatchify
   ~modules.mlp.MLP

Modules — Gating & conditioning
-------------------------------

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

Modules — Residual blocks
-------------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.residual_block.ResidualBlock
   ~modules.residual_block.AdaLNZeroResidualBlock
   ~modules.vit5_residual_block.ViT5ResidualBlock

Modules — Schedulers
--------------------

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~modules.schedulers.ResumableSequentialLR

Networks
--------

End-to-end classification / general-purpose networks composing the
modules above.

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~networks.classification_resnet.ClassificationResNet
   ~networks.general_purpose_resnet.ResidualNetwork
   ~networks.vit5_classification.ViT5ClassificationNet

Parallel
--------

Context-parallel communication primitives (zigzag splits / all-to-all).

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~parallel.utils.init_parallel_state
   ~parallel.utils.zigzag_split_across_group_ranks
   ~parallel.utils.zigzag_gather_from_group_ranks
   ~parallel.utils.setup_rank0_logging

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~parallel.a2a_comms.AllToAllSingleFunction

Utilities
---------

QK normalization, rotary position embeddings, and weight-init helpers
shared across mixers.

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~utils.qk_norm.apply_qk_norm
   ~utils.quack_utils.cuda_supports_quack
   ~utils.rope.apply_rope_1d_bhl
   ~utils.rope.apply_rope_2d_bhl
   ~utils.rope.apply_rope_3d_bhl
   ~utils.rope.apply_rope_1d_blh
   ~utils.rope.apply_rope_2d_blh
   ~utils.rope.apply_rope_3d_blh
   ~utils.rope.construct_rope_1d_cache_bhl
   ~utils.rope.construct_rope_2d_cache_bhl
   ~utils.rope.construct_rope_3d_cache_bhl

.. autosummary::
   :toctree: generated/
   :template: class_template.rst

   ~utils.qk_norm.L2Norm

Metrics
-------

.. autosummary::
   :toctree: generated/
   :template: function_template.rst

   ~metrics.cleanfid.compute_folder_fid
