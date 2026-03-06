# ViT-5 Reference-Matching Tracker

Goal: match the ViT-5-reference implementation (82.2% top-1 on ImageNet-1k) by fixing
architectural and initialization discrepancies.

W&B project: [`implicit-long-convs/nvsubquadratic`](https://wandb.ai/implicit-long-convs/nvsubquadratic)

## Discrepancies identified

| #   | Discrepancy                                                                                              | Severity | Fix                                                                                                                                                                                                                                                             | Status |
| --- | -------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1   | **Weight init**: Kaiming uniform (PyTorch default) vs `trunc_normal_(std=0.02)` for all Linear layers    | High     | Added `init_fn_qkv_proj`, `init_fn_out_proj` to `ViT5Attention`; `init_method_in`, `init_method_out` to `MLP`; `_init_weights()` in `ViT5ClassificationNet` for embeddings/head. Created `utils/init.py` with `trunc_normal_init`, `trunc_normal_init_factory`. | Done   |
| 2   | **Bias init**: Linear biases not zero-initialized                                                        | High     | Added `nn.init.zeros_` for biases in both `ViT5Attention` and `MLP` when init functions are provided.                                                                                                                                                           | Done   |
| 3   | **MLP bias**: `bias=False` (hardcoded assert) vs reference `bias=True` (timm `Mlp` default)              | High     | Removed assert, made `bias` configurable (default `False`).                                                                                                                                                                                                     | Done   |
| 4   | **Attn output proj bias**: `bias=False` vs reference `bias=True` (`nn.Linear(dim, dim)` default)         | High     | Added `out_proj_bias` parameter (default `True` in module, `False` in v2 config — overridable via CLI).                                                                                                                                                         | Done   |
| 5   | **Attn scale with QK-norm**: `scale=1.0` when `qk_norm=True` vs reference always uses `head_dim ** -0.5` | High     | Made `scale` a configurable parameter (`Optional[float]`, defaults to `head_dim ** -0.5`). Removed the `1.0 if qk_norm` conditional.                                                                                                                            | Done   |

## Files changed

- `nvsubquadratic/utils/init.py` — new file: `trunc_normal_init`, `trunc_normal_init_factory`, `small_init`, `wang_init`, `partial_wang_init_fn_with_num_layers`
- `nvsubquadratic/modules/init_functions.py` — re-export shim pointing to `utils/init.py`
- `nvsubquadratic/modules/vit5_attention.py` — added `init_fn_qkv_proj`, `init_fn_out_proj`, `out_proj_bias`, `scale`; zero-init for biases
- `nvsubquadratic/modules/mlp.py` — removed `assert bias is False`; zero-init for biases when init methods provided
- `nvsubquadratic/networks/vit5_classification.py` — `_init_weights()` for cls_token, pos_embed, reg_token, patch_embed, out_proj
- `examples/vit5_imagenet/v2/vit5_small_pretrain_apex_dali_fused.py` — wired init fns, explicit `out_proj_bias=False`, `bias=False` (CLI-overridable)
- `examples/vit5_imagenet/v1/vit5_small_pretrain_apex_dali_fused.py` — same + `out_proj_bias=True`, `bias=True`
- `examples/vit5_imagenet/wsd_ft_ablation/_base.py` — same

## Active pretraining runs (2026-03-04)

All runs use: v2 EMA config (`v2/vit5_small_pretrain_attention_ema.py`), EMA decay=0.99996,
trunc_normal init, `scale=None` (defaults to `1/sqrt(d_k)`), 800 epochs, LAMB lr=4e-3.

| Job ID | Job name                  | Loss           | MLP bias | out_proj_bias | Status  | Notes                        |
| ------ | ------------------------- | -------------- | -------- | ------------- | ------- | ---------------------------- |
| 33421  | `vit5-pt-softce-ema`      | soft_target_ce | False    | False         | Running | Init-only fix                |
| 33422  | `vit5-pt-bce-ema`         | bce            | False    | False         | Running | Init-only fix                |
| 33434  | `vit5-pt-softce-ema-bias` | soft_target_ce | True     | True          | Running | Init + bias (full ref match) |
| 33437  | `vit5-pt-bce-ema-bias`    | bce            | True     | True          | Running | Init + bias (full ref match) |

### Early checkpoints (epoch ~87–119, jobs 33421/33422)

| Job                     | Metric                  | Value |
| ----------------------- | ----------------------- | ----- |
| 33421 (softce, no bias) | val/acc_ema @ epoch 87  | 64.4% |
| 33422 (bce, no bias)    | val/acc_ema @ epoch 119 | 50.4% |

## Previous pretraining baselines (before init fix)

| Run                       | Loss           | val/acc (pretrain) | val/acc (finetune) | Notes                                        |
| ------------------------- | -------------- | ------------------ | ------------------ | -------------------------------------------- |
| `vit5-apex` (30923)       | BCE            | ~81.7%             | —                  | Kaiming init, no EMA, scale=1.0 with QK-norm |
| `vit5-dali-fused` (32158) | soft_target_ce | ~82%               | —                  | Kaiming init, no EMA                         |
