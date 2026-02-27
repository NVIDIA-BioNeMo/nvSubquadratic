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
