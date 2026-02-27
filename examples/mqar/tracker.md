# MQAR Experiment Tracker

**Goal**: Evaluate associative recall capabilities of Hyena vs Attention.
**Metric**: Accuracy (exact match on query-answer pairs).

## Planned Experiments

### 1. Scaling `num_kv_pairs` (Primary)

**Why**: This is the core diagnostic. Attention should maintain perfect recall as pairs increase (up to context limit), while Hyena/SSMs typically degrade.
**Config**: `seq_len=256`, `vocab=8192`.
**Variations**: `num_kv_pairs` ∈ {4, 8, 16, 32, 64}

### 2. Scaling `seq_len`

**Why**: Test recall over longer distances.
**Variations**: `seq_len` ∈ {256, 512, 1024, 2048} (with fixed `num_kv_pairs=8` or proportional)

### 3. Power Law Distribution (`power_a`)

**Why**: Test robustness to query distribution shifts (clustered vs uniform).
**Variations**: `power_a` ∈ {0.01 (default), 0.1, 1.0 (uniform)}

### 4. Hyena Ablations — Closing the Associative Recall Gap

> \[!NOTE\]
> **Baseline**: `hidden_dim=128`, `num_blocks=4`, `MLP expansion=2.0`, `mask_cfg=Identity()` (no filter decay), `num_kv_pairs=8`

#### 4a. Increasing Hidden Dimension (`d_model`)

**Why**: Each channel acts as a storage slot. More channels = more capacity to "store" KV pairs in the convolutional state.
**Parameter**: `hidden_dim` in `base_config.py`
**Variations**: `hidden_dim` ∈ {128 (baseline), 256, 512}

#### 4b. Increasing MLP Expansion Factor

**Why**: Creates a larger "internal scratchpad" for gating without increasing `d_model` across all layers.
**Parameter**: `expansion_factor` in `base_config.py → block_cfg → mlp_cfg`
**Variations**: `expansion_factor` ∈ {2.0 (baseline), 4.0, 8.0}

#### 4c. Filter Decay / Windowing

**Why**: The current Hyena config uses `mask_cfg=Identity()` (no decay). Adding `ExponentialModulationND` controls how quickly the filter "forgets" distant tokens. Too aggressive decay truncates long-range recall.
**Parameter**: `mask_cfg` in `mixer_defaults.py → global_conv_cfg`
**Variations**:

- No decay (baseline, `Identity`)
- Slow decay: `ExponentialModulationND(slow_decay_pct=0.5, fast_decay_pct=2.0)`
- Default decay: `ExponentialModulationND(slow_decay_pct=2.3, fast_decay_pct=13.81)`

#### 4d. Hybrid Architecture (Hyena + Attention) — *Lower Priority*

**Why**: Insert a single Attention layer to handle exact recall while Hyena handles bulk sequence processing.
**Implementation**: Requires modifying `base_config.py` to allow per-block mixer selection.
**Variations**:

- 4 Hyena blocks (baseline)
- 3 Hyena + 1 Attention (last block)
- 2 Hyena + 1 Attention + 1 Hyena (middle block)

______________________________________________________________________

## Results

### 1. Accuracy vs Number of Pairs (`seq_len=256`)

| `num_kv_pairs` | Hyena | Attention | Run IDs (Hyena / Attn) |
| :------------- | :---- | :-------- | :--------------------- |
| 4              | 0.961 | 0.988     | 132724 / 132728        |
| 8              | 0.598 | 0.965     | 132725 / 132729        |
| 16             | 0.062 | 0.858     | 132726 / 132730        |
| 32             | 0.031 | 0.711     | 132727 / 132731        |
| 64             | 0.015 | 0.531     | 132964 / 132965        |

### 2. Accuracy vs Sequence Length (`num_kv_pairs=8`)

| `seq_len` | Hyena | Attention | Run IDs         |
| :-------- | :---- | :-------- | :-------------- |
| 256       | -     | -         | (See Table 1)   |
| 512       | -     | -         | 133178 / 133179 |
| 1024      | 0.109 | 0.949     | 133180 / 133181 |
| 2048      | 0.425 | 0.863     | 133967 / 133968 |

### 3. Hyena Ablations (`num_kv_pairs=8`, `seq_len=256`)

#### 3a. Width Scaling

| `hidden_dim`   | Hyena Accuracy   | Params | Run ID        |
| :------------- | :--------------- | :----- | :------------ |
| 128 (baseline) | 0.598            | ~0.8M  | (See Table 1) |
| 256            | 0.121            | ~3.2M  | 134274        |
| 512            | 0.097 (Timeout?) | ~12.6M | 134275        |

#### 3b. MLP Expansion Factor

| `expansion_factor`  | Hyena Accuracy           | Params | Run ID             |
| :------------------ | :----------------------- | :----- | :----------------- |
| 2.0 (baseline)      | 0.598                    | ~0.8M  | (See Table 1)      |
| 4.0                 | 0.635                    | ~1.1M  | 134279             |
| 8.0                 | 0.889                    | ~1.7M  | 134276             |
| 16.0                | 0.191 (Poor convergence) | ~3.0M  | 134717             |
| 32.0                | 0.904                    | ~5.7M  | 134718             |
| \<\<\<\<\<\<\< HEAD |                          |        |                    |
| 64.0                | 0.536                    | ~10.4M | 134875 (Completed) |
| =======             |                          |        |                    |
| 64.0                | -                        | ~10.4M | 134875             |

> > > > > > > 3a93d71 (New setup lm experiments)

#### 3c. Filter Decay (`mask_cfg`)

| Decay Config                     | Hyena Accuracy | Run ID        |
| :------------------------------- | :------------- | :------------ |
| None / `Identity` (baseline)     | 0.598          | (See Table 1) |
| Slow (`slow=0.5, fast=2.0`)      | 0.241          | 134277        |
| Default (`slow=2.3, fast=13.81`) | 0.121          | 134278        |

#### 3d. Hybrid Architecture

| Architecture                  | Accuracy | Run ID        |
| :---------------------------- | :------- | :------------ |
| 4× Hyena (baseline)           | 0.598    | (See Table 1) |
| 3× Hyena + 1× Attn (last)     | -        |               |
| 2× Hyena + 1× Attn + 1× Hyena | -        |               |

______________________________________________________________________

## Notes & Observations

- **2024-02-11**: Initial setup complete. Default `num_kv_pairs=8`.
- **Note**: Validation accuracy is logged every 1000 steps.

### Observations: The Associative Recall Gap

**Phenomenon**: Hyena performance collapses as `num_kv_pairs` increases, while Attention remains robust.

**Technical Explanation**:

1. **Addressing Mechanism**:

   - **Attention (Content-based)**: Uses similarity search (`DotProduct(Query, Key)`). It effectively acts as a differentiable hash map (`Value = Dictionary[Key]`), allowing direct lookups regardless of distance.
   - **Hyena (Position-based)**: Uses convolutions (`Output = Input * Filter`). To retrieve a value, it must generate a filter with a precise spike at the exact relative distance of the key. This requires the model to "measure" distance based on content.

1. **The "Dynamic Filter" Bottleneck**:

   - As `num_kv_pairs` increases, the sequence becomes more crowded.
   - Hyena's gating/projection layers must compress the context into a filter specification.
   - Generating a filter with sufficiently high resolution to distinguish, for example, "Distance 48" from "Distance 49" solely from compressed state becomes increasingly difficult (Vanishing Resolution).
   - The model fails to maintain the precise positional arithmetic needed for retrieval in dense sequences.

**Conclusion**: This fundamental difference—Attention "cheating" with $O(N^2)$ visibility vs Hyena compressing into $O(N \\log N)$ convolutions—explains the gap. It confirms the benchmark is correctly functioning as a diagnostic for this known SSM/Convolutional weakness.
