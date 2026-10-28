# Skill verification

Two complementary checks live here:

- `test_vision_encoder.py` is a runnable CPU regression suite for the copyable dense vision asset. It checks rectangular geometry and shape-history invariance, batch independence, real FiLM conditioning, FP32 and CPU BF16-autocast gradients, registered parameters, projection transfer, strict trained-checkpoint reload, unsupported arguments, and actual encoder replacement.
- `evals.json` describes independent agent scenarios. These are behavioral acceptance criteria, **not an executable agent harness**. They cover native configs, CLS classifiers, dense vision with FiLM, 3D feature maps, causal 1D models, and packed-batch design. There is no automated model invocation or scheduled evaluation in this directory.

## CPU asset checks

Use an environment compatible with the repository's current torch requirement, plus `einops`, `omegaconf`, `numpy`, and `pytest`. From the repository root:

```bash
python -m pytest -o addopts='' skills/hyenand-retrofit/evals/test_vision_encoder.py -q
```

The empty `addopts` avoids requiring the repository's optional coverage plugin for this standalone suite. The tests use `torch_fft`; they do not validate CUDA backends, full precision-mode coverage, distributed execution, real-model quality, or speed. Run those checks in the intended host environment before drawing those conclusions.

## Independent agent evaluation

Give a fresh agent the skill, one scenario prompt, and its raw input files. Have it work on copies in an isolated directory with the repository on `PYTHONPATH`. Do not supply the proposed solution or prior failure analysis. Run and inspect its generated artifact against the behavioral criteria; text matching alone cannot prove an encoder was actually replaced, preserves causality, or has image-dependent CLS output.

The Python fixtures under `inputs/` are small executable hosts. For the vision scenarios, use small hidden dimensions/grids for CPU checks, then validate realistic shapes on the intended device. Preserve the original fixtures. A generated integration is a separate artifact from this regression suite; do not claim all scenarios passed merely because the asset tests passed.
