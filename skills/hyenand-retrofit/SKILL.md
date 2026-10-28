---
name: hyenand-retrofit
description: Integrate nvSubquadratic HyenaND into PyTorch attention models, including vision encoders with optional FiLM, 1D sequence models, and 3D feature maps. Use for attention-to-Hyena replacements, hybrid architectures, and adapting host layouts, projections, conditioning, or checkpoints.
---

# hyenand-retrofit

Produce a working integration with the user's model and installed nvSubquadratic revision. For conceptual requests, answer in chat. For implementation, prefer a sibling variant unless the user requests an in-place change or the host already has a mixer-selection configuration. Preserve the surrounding objective, patch embedding, residual/MLP structure, and output contract unless changing them is part of the request.

## Inspect the host before replacing attention

Trace the actual encoder construction and forward call, not only the attention class. Establish from code:

- Spatial rank, tensor layout, actual patch-grid dimensions at each stage, and token ordering. Image pixels, patch tokens, and biological cells are different quantities.
- Full attention module versus just its mixing operation: which module owns QKV and output projections, normalization, positional transforms, and residual additions?
- Dense batches versus packed/ragged inputs, prefix tokens, dropped/shuffled patches, masks, and cross-attention.
- Tensor or tuple return, downstream dense features versus CLS/pooling readout, and pretrained-loading order.

Ask only about choices the code cannot establish. A pure or hybrid layer pattern and FiLM on/off are experiment choices; use the user's preference or state a starting assumption. Preserve attention where an unsupported mask or cross-attention contract is essential.

## Select the integration path

- **Vision encoder:** read [references/vision-encoders.md](references/vision-encoders.md). Use [assets/vision_encoder.py](assets/vision_encoder.py) as a copyable starting point for dense, complete 2D patch grids with no prefixes. It includes learned projections, optional patch-conditioned FiLM, and an explicit runtime grid within a declared supported bound. Prefix/register and packed-batch integrations need the additional handling described in the reference.
- **Native nvSubquadratic config:** reuse the matching builder and inspect its instantiated module graph. The current `examples/vit5_imagenet/vit5_hybrid/_base_config.py::build_hybrid_net` accepts a layer pattern; all `H` gives pure Hyena. Plain hybrid construction does not enable FiLM; see the vision reference for `apply_film_overrides`. Historical configs are examples, not API guarantees: the older `v5_patch` builder includes FiLM but also uses legacy `prepend_registers`/prefix-layout assumptions. Check constructor signatures and padding/readout indices before reusing it with the current network. Keep config siblings as config shims; instantiate and exercise the graph in a separate check.
- **1D, 3D, feature-map, or diffusion host:** read [references/other-hosts.md](references/other-hosts.md). Retain the host's conditioning and causal semantics. The dense 2D asset is not an autoregressive or cross-attention adapter.

## Current library contracts

Verify these against the checked-out revision before adapting code:

- `Hyena` consumes separate Q, K, V tensors in channels-last layout `[B, *spatial, C]`; `QKVSequenceMixer` supplies learned QKV and output projections around it. Replacing a full attention module requires equivalent projections. Replacing only an attention kernel must not duplicate them.
- `Hyena` has no `use_rope` or `rope_base` constructor arguments. Positional operations in the host require an explicit integration decision.
- `CKConvND` requires `grid_type`: usually `"double"` with zero padding, `"single"` with circular padding, or `None` with a per-axis padding list. It owns `is_causal`, `fft_padding`, and `fft_backend`; there is no `use_fp16_fft` constructor argument.
- Start functional checks with `fft_backend="torch_fft"`. Select an accelerated backend only after checking its installed package, supported dimensions/dtypes/extents, and forward/backward behavior. See `nvsubquadratic/modules/ckconv_nd.py`; CPU correctness does not establish CUDA speed.
- An adapter must reject unsupported masks, cross-attention, causal requests, and requests for attention weights. Do not absorb them with `**ignored_kwargs`. An MHA-compatible wrapper may return `(out, None)` when `need_weights=False`, but must validate its other arguments.
- Construct and register modules before optimizer, EMA, DDP/FSDP, and compilation setup. Do not create trainable layers during `forward`.

## Verify behavior, not only shape

Run a small forward/backward through the actual adapted encoder. Confirm the intended blocks were replaced and gradients reach new Hyena parameters and image/patch inputs. Preserve output shape and dtype; output values are expected to change when changing the mixer.

For CLS readout, show image-dependent logits and a nonzero path from logits to patches. For FiLM, verify conditioning reaches the kernel and changes it after the identity-initialized output head moves. For causal models, perturb future tokens and verify earlier outputs do not change. For packed inputs, compare batched/bucketed output with independently evaluated samples.

Check that revisiting a grid after other image sizes preserves the output. Check trained-checkpoint save/load, the intended training precision, and representative geometry before larger runs. Report which checks actually ran and any remaining GPU, distributed, or performance validation. The asset's CPU regression tests run with:

```bash
python -m pytest -o addopts='' skills/hyenand-retrofit/evals/test_vision_encoder.py -q
```

See [evals/README.md](evals/README.md) for independent agent scenarios. Deliver the integration, its selected layer pattern/conditioning, validation results, and a concrete next experiment; do not claim task-quality or throughput gains from a successful smoke test.
