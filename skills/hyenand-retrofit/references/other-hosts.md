# Other HyenaND hosts

Use the checked-out APIs and matching examples. The vision asset is 2D and bidirectional; changing only `data_dim` is not a complete general retrofit.

## Feature maps and volumes

For `[B,C,*spatial]`, move channels to the last axis for `QKVSequenceMixer`, apply the mixer, and restore the layout. Match `data_dim` on kernel/global convolution and use `Conv1d`, `Conv2d`, or `Conv3d` for the short convolution. Preserve projections, residual paths, conditioning, downsampling, and return type.

Boundary conditions are independent:

- Zero-padded noncausal: `fft_padding="zero", grid_type="double"`.
- Periodic: `fft_padding="circular", grid_type="single"`.
- Per-axis: for example `fft_padding=["circular", "zero"], grid_type=None`, using `torch_fft`.

Start with `torch_fft` for 3D. Choose SIREN `L_cache` deliberately: it sets coordinate spacing and allocates a coordinate cache whose storage grows rapidly with spatial rank. The library can extend it, but repeated extensions in the current implementation can change coordinates according to input history. Prefer a declared supported extent with a bound check, preserve that setting across checkpoint reloads, and test shape-history invariance before relying on dynamic extension. Avoid an unnecessarily large maximum for 3D. Large volumes also stress short-convolution intermediates; profile the actual dtype/backend/grid before selecting patch merging or chunking. Batch chunking cannot fix an operator that already fails for one sample; spatial chunking of the short convolution needs halos. Small halos do not preserve global Hyena context: exact global-convolution chunking needs full kernel support or appropriate global communication.

Source examples:

- `examples/well/v2/active_matter/hyena.py`: 2D periodic feature maps.
- `examples/well/v2/supernova_explosion_64/hyena.py`: 3D zero padding.
- `examples/well/v2/MHD_64/hyena.py`: 3D periodic feature maps.
- `examples/vit5_imagenet/v5_patchmerge/`: hierarchical layouts.

## Causal 1D

Autoregression requires causality throughout the block:

- `CKConvND(data_dim=1, is_causal=True, fft_padding="zero", grid_type="double", ...)`.
- Use `nvsubquadratic.modules.causal_conv1d.CausalConv1D(is_causal=True, ...)` for the short convolution. Symmetric `Conv1d(padding=1)` leaks future tokens.
- Normalize over channels at each position, not future positions. Hyena's internal layout is channels-first; `RMSNormChannelFirst` and `L2Norm(dim=1)` are appropriate references.
- Do not use whole-sequence pooling for FiLM in autoregressive models.
- Replace a conventional triangular mask with a verified causal operator at the call site. Reject arbitrary, padding, or segment-isolation masks unless their semantics are implemented; do not silently discard them.
- `Hyena` has no RoPE constructor knobs. Adapt host positional transforms explicitly.

See `examples/spatial_recall_v2/mixer_defaults.py` and `simple_copy_1d/hyena.py` for composition patterns; resolve experiment interpolations to concrete dimensions. `subq_ops` has a 1D causal path but does not support batched FiLM kernels. Start with `torch_fft` and test the intended path.

To check causality, hold sequence length fixed, change only tokens after position `t`, and compare outputs through `t`. Test the whole model to catch short-convolution, normalization, and conditioning leakage.

## Diffusion and conditional networks

Replace self-attention where a spatial lattice is defined. Preserve timestep/class conditioning and the host's AdaLN or other modulation. SIREN FiLM is an additional option, not a substitute for host conditioning. Cross-attention over text/external keys has a different contract and should remain unless an architectural redesign is intended. See `examples/imagenet_diffusion/ccnn_12_768_hyena_qknorm.py` for composition ideas, checking current signatures and host semantics.
