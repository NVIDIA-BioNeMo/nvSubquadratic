# Meta-Kernel: Future Directions

Tracker for ideas to improve the centralized kernel generator (`MetaSIRENKernelND`).

## Implemented

- **Single SIREN MLP for all layers**: shared positional embedding + hidden layers + single output linear projecting to `num_layers * out_dim`, split into per-layer kernels. Mask applied once on full tensor before split.

## Ideas

### FiLM Conditioning
Per-layer scale+shift vectors on the shared hidden features, applied before the output projection. This lets the model modulate per-layer kernel complexity with minimal extra parameters (~2 x hidden_dim per layer). The shared backbone learns the spatial structure; FiLM controls how much each layer "listens" to different spatial frequencies.

**Potential benefit**: increases input-dependency of kernel generation — the same spatial features are weighted differently per layer, allowing the model to decide which layers need more complex kernels.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.

### Layer Embeddings
Learnable per-layer embedding concatenated or added to the positional encoding before entering the shared backbone. This makes the backbone itself layer-aware — it sees `(x, y, layer_id)` instead of just `(x, y)`.

**Trade-off**: changes the backbone's input dimensionality (if concatenated) or requires matching dimensions (if added). More expressive than FiLM but increases backbone compute.

### Per-Layer Output Heads
Replace the single large output linear with N smaller independent heads. Each layer gets a dedicated `Linear(hidden_dim, out_dim)` projection. The backbone is still shared, but the output is specialized.

**Trade-off**: same total output params as the single large linear, but breaks the cross-layer structure in the output weight matrix. Could be useful if layers need very different kernel patterns.

### Dynamic Capacity Allocation
Explore mechanisms (e.g., soft routing, mixture-of-experts on output columns) where the generator learns to allocate more expressive kernels to certain layers. The key insight: not all layers need equally complex spatial kernels.

### Batched FFT Convolutions
Since all N kernels are produced at once by MetaSIRENKernelND, explore batching the downstream FFT convolutions across layers. Instead of 12 sequential FFT conv calls, batch them into fewer (or one) operation.

### Input-Dependent FiLM via Register Tokens
Apply FiLM conditioning in individual `CKConvND`/`SIRENKernelND` modules (not meta-kernel) using the mean of register tokens as the conditioning signal. This provides attention-free, input-dependent modulation of kernel generation. ~14 register tokens → mean-pool → linear projection → per-layer (scale, shift).

## Implemented (Experiments)

### FP16 FFT Convolutions (`use_fp16_fft`)

**Status**: Implemented and validated.  
**Files**: `nvsubquadratic/ops/fftconv_fp16.py`, `CKConvND(use_fp16_fft=True)`

Replaces the f32 FFT convolution with fp16 using three key techniques:
1. **Power-of-2 padding**: cuFFT requires power-of-2 sizes for half-precision (28→32 for 14×14 input + 27×27 kernel)
2. **Ortho normalization**: `norm="ortho"` divides both forward/inverse FFT by `sqrt(N)`, keeping intermediate complex products within fp16 range (max ~2.8 vs 52064 in standard norm). A `sqrt(N)` correction factor restores the correct convolution scale.
3. **No f32 upcasting**: Inputs cast directly to fp16 instead of f32, halving intermediate memory.

**Results** (v2-hyena-gap checkpoint, epoch 607, H100):
- ImageNet-1k validation: **80.10% (f32) vs 80.10% (fp16)** — zero accuracy loss
- Top-1 prediction match: 98.5% on random inputs (cosine similarity 0.999987)
- Isolated FFT memory: 36% savings per convolution (1002→638 MB at batch 256)
- Isolated FFT speed: ~9% slower (32×32 vs 28×28 padding overhead)
- End-to-end model: ~3% total memory savings, ~0% speed change (FFT is small fraction of compute)

**Limitations**:
- 2D zero-padded convolutions only (no 1D, 3D, circular, or causal support yet)
- "ComplexHalf support is experimental" warning from PyTorch
- Power-of-2 constraint adds padding overhead that varies with input/kernel size
