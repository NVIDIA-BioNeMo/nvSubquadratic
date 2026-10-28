# Vision-encoder integration

## Dense patch encoder: first working comparison

For an encoder returning one feature per image patch, keep the patch embedding, positional embeddings, residual/MLP layers, feature ordering, and downstream head. Obtain `(H, W)` from the patch embedding's output before flattening; do not infer a square grid from token count. A 600-by-600 image with non-overlapping 20-by-20 patches has 900 tokens, not 360,000. Smaller patches or larger fields of view are separate experiments that also affect positional embeddings and pretrained compatibility.

Copy [../assets/vision_encoder.py](../assets/vision_encoder.py) into the host project and import `Hyena2DTokenMixer`. It operates on complete row-major patch grids and returns dense tokens in the same order:

```python
# Construct once in the host block's __init__.
self.token_mixer = Hyena2DTokenMixer(
    hidden_dim,
    max_grid_hw=(30, 30),  # configure the supported patch-grid bound before training
    film_conditioning="patch_mean",  # explicit ablation; "none" disables FiLM
    fft_backend="torch_fft",  # correctness baseline
    qkv_bias=True,  # match the host's projections
    out_proj_bias=True,
)

# In forward: grid_hw comes from patch embedding, before flattening.
x = x + self.token_mixer(self.norm1(x), grid_hw=grid_hw)
x = x + self.mlp(self.norm2(x))
```

Thread `grid_hw` through encoder/block calls. The constructor requires `max_grid_hw`; choose the supported patch-grid bound before training and retain it in the saved model configuration. This fixes SIREN coordinate spacing as well as cache capacity, so changing it is an architecture change, not a transparent memory optimization. The asset rejects larger grids. In the current library, repeated coordinate-cache extensions can make outputs depend on the history of input sizes; the bounded asset avoids that path. Verify identical-input outputs before/after other rectangular sizes and after checkpoint reload. Do not remove the bound check without verifying the library's cache behavior.

Preserve the host's residual scaling, dropout, stochastic depth, and output projection contract; this schematic is a pre-norm example, not a replacement for those details. Do not replace only `scaled_dot_product_attention` with this whole mixer: it already owns QKV and output projections. Retained attention blocks keep their own interface.

The asset intentionally does not accept MHA's three-input/tuple contract, prefixes, masks, packed tokens, or cross-attention. Adapt the call site, or write a narrow wrapper that validates the supported contract. Exact `H*W` validation catches extra tokens but cannot detect a permutation: verify row-major ordering in host code. Keep the complete lattice when using masked input embeddings; sparse token removal needs grid reconstruction and an explicit missing-token policy.

For channels-first feature maps, move `[B,C,H,W]` to `[B,H,W,C]`, flatten to tokens for the asset, then restore the original layout. Preserve downsampling. Hierarchical encoders use each stage's current width, height, and channel count.

## FiLM needs an end-to-end conditioning path

FiLM conditions the SIREN kernel generator on a per-image vector; it is not an attention-style retrieval operation. Registers or a FiLM module alone do not activate it. Wire the entire path:

`observed features -> conditioning [B,Ccond] -> QKVSequenceMixer -> Hyena -> CKConvND -> SIRENKernelND -> KernelFiLMGenerator`

The kernel needs `film_cfg`, and the forward call needs non-`None` `conditioning`. Otherwise it can silently use the unmodulated kernel. Choose the conditioning source explicitly:

- **No registers:** the asset's `patch_mean` uses the mean incoming patch features independently for each image. In a pre-norm block these are normalized features. This is a simple retrofit/ablation, not the register-conditioned ImageNet recipe.
- **Existing registers:** use `RegisterPooling` on incoming normalized registers, excluding CLS and padding. Confirm indices and ensure registers exchange information with patches. Learned registers alone are image-independent at the first block; later updated registers can carry image information.
- **No FiLM:** `film_conditioning="none"` is a useful baseline. Do not insert registers or change readout merely to turn FiLM on.

`KernelFiLMGenerator.cond_dim` must match the conditioning width; `kernel_hidden_dim` must match the SIREN MLP width. Set `num_film_layers = num_layers - 1 + int(film_after_pos_embed)`. Positional-embedding FiLM additionally requires `embedding_dim == mlp_hidden_dim`. The asset uses three SIREN layers, positional-embedding FiLM, and three gamma/beta pairs.

Identity initialization (`gamma=1`, `beta=0`) gives a controlled start. Different conditioning vectors deliberately yield the same modulation initially. Check gradients on the FiLM output head, then take a small update or perturb that head before testing conditioning sensitivity. Check that host-wide initialization or parameter reparametrization has not overwritten this initialization. Include FiLM parameters in the optimizer; library `_no_weight_decay` metadata only works if the host optimizer-group builder honors it.

For masked-expression or conditional objectives, pool only features available to the predictor. Do not introduce hidden targets into conditioning. A bidirectional image encoder can coexist with an autoregressive downstream decoder without changing the decoder's objective.

**Native config route:** `examples/vit5_imagenet/vit5_hybrid/_film.py::apply_film_overrides(config)` adds register pooling and kernel FiLM after `build_hybrid_net` and any kernel-replacement override. Its defaults include positional-embedding FiLM and eight registers. These are configurable example choices, not requirements for an external encoder. Inspect the instantiated graph/layout and preserve the host's register count when appropriate.

## CLS and register communication

Do not peel off CLS/register tokens, mix only patches, and return unchanged prefixes as the mixer output. Full replacement can disconnect a CLS classifier from its image; residual addition can also incorrectly add prefixes to themselves.

Choose a communication strategy that fits the host:

- Fold special tokens into separate auxiliary grid rows, mix patches and auxiliary rows, then restore the host's sequence order. For `[prefixes, patches]`, split first; arrange `[patches reshaped HxW; prefixes plus padding reshaped ceil(P/W)xW]`. Directly reshaping `[prefixes, patches]` to width `W` shifts patch coordinates. Recover prefixes from auxiliary rows and patches from their original rows. Decide whether auxiliary padding persists or is recreated each block; resetting it changes the model.
- Keep attention blocks that exchange information between patches and prefixes, and define the Hyena branch's prefix update explicitly (for example, a zero residual update). Validate the whole hybrid.
- Use patch pooling as readout only when that architecture change is intended; it changes a pretrained CLS model's contract.

`ViT5HyenaAdapter` only reshapes. Its caller must arrange/pad tokens and supply projections through the inner mixer. Current `ViT5ClassificationNet` uses `[patches, CLS, registers, padding]`; do not copy another host's prefix indices. When using the asset for this folded layout, `max_grid_hw` must include the auxiliary rows. They can push FFT sizes over a padding boundary; include their cost in measurements.

## Packed batches and resolution

Each image needs its own convolution domain. This does **not** require a sequential GPU call per image:

- Stack same-`(H,W)` images as `[B,H,W,C]` and process together, including separate FiLM conditioning `[B,Ccond]`.
- Group varying shapes by `(H,W)` and compatible prefix layout, gather complete grids, process each group as a batch, and scatter back to packed order. Preserve sample boundaries and inverse token permutations. Pointwise operations can stay packed where supported.
- Padding to a common rectangle requires a defined policy and validation: it changes the coordinate grid, convolution domain, pooling, and possibly auxiliary-token layout. Masking final outputs alone does not guarantee equivalence.

`cu_seqlens` supplies boundaries, not geometry. Window reordering, shuffled/dropped tokens, and sparse coordinates need reconstruction. Never reshape different images into one grid; that mixes samples. The asset handles uniform dense batches, not a ragged kernel or bucketing wrapper.

Measure the actual size distribution, gather/scatter, kernel generation, FiLM, forward/backward, and peak memory. Large total packed length is not large per-image context: packed attention cost depends on the sum of per-image squared lengths. Demonstrate high-resolution speedups at the intended per-image token counts.

## Pretrained initialization and trained checkpoints

For an attention checkpoint, load the original model, construct replacements on the same device with compatible projection dtypes, and transfer compatible projections before replacing modules. The asset's `copy_projections_from(qkv, out_proj)` handles same-width `nn.Linear(C,3C)` and `nn.Linear(C,C)` with matching biases. Separate Q/K/V, MHA's `in_proj_weight`, unequal dimensions, or extra QK normalization need explicit mapping. Reuse compatible embedding/MLP/norm weights; new kernels need training.

For a **trained Hyena checkpoint**, construct the matching Hyena architecture/conditioning before loading it. Reconstruct with the same `max_grid_hw` and FiLM configuration: those constructor choices are not stored in `state_dict`. Require trained Hyena/FiLM/projection parameters to load; missing/unexpected keys need an explained mapping, not blanket `strict=False`. Verify save/load output agreement on a small input.

Finish replacements before optimizer, EMA, distributed wrappers, and compilation. Confirm the encoder being trained/evaluated contains the intended Hyena modules; testing an unchanged attention model proves nothing about the integration.

## Backend and experiment progression

Use `torch_fft` for the CPU reference. The asset includes a small `CKConvND` subclass that casts the local convolution shortcut to the activation dtype under autocast; this preserves the original FP32 parameter and its gradients. Without that cast, the current reference FFT path rejects BF16 activations with an FP32 shortcut. CPU BF16 autocast is covered by the asset tests; CUDA precision/backend validation is still required. Keep SIREN coordinate buffers in FP32: prefer FP32 model parameters with autocast. Blanket `.half()`/`.bfloat16()` also casts those buffers and is not supported by this template without additional buffer handling.

Current `subq_ops` 2D zero-padded noncausal convolution supports batched FiLM kernels. `subq_ops_fused` also supports them but currently caps spatial extents at 64 per axis, including auxiliary rows. Check `CKConvND` and the installed CUDA extension before selecting either, especially at high resolution. Consult the repository's current CUDA extra and getting-started instructions for installation.

First compare attention with a chosen Hyena/hybrid variant at unchanged patch grid and training objective. Treat FiLM on/off as a separate knob. Then evaluate larger fields of view or finer patches, handling positional embeddings and controlling for task quality. Record task quality, encoder and end-to-end training throughput, and memory. CPU tests or unrelated benchmarks do not establish downstream gains. Context parallelism requires separate end-to-end validation of the intended 2D layout and conditioning path.
