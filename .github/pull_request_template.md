## Summary

<!-- What does this PR change and why? -->

## Environment setup

Create the conda environment (required to run tests):

```bash
bash setup_conda_env.sh
conda activate nvsubquadratic
```

## Test plan

- [ ] `pre-commit run --all-files` passes (`pre-commit install` if not yet set up).
- [ ] Existing tests pass (`pytest tests/`).
- [ ] New tests added, or explain why not needed:

## Documentation checklist

For every new or modified public symbol in `nvsubquadratic/` or `experiments/`:

- [ ] Every new **module** has a module-level docstring explaining what it contains and why.
- [ ] Every new **public class** has a class docstring covering purpose, math/motivation, and key attributes.
- [ ] Every new **public method / function** has `Args:` and `Returns:` blocks with tensor shapes where applicable.
- [ ] Math notation is consistent with the paper (or a comment explains any deviation).
- [ ] Docstrings containing backslashes use `r"""..."""` (required by ruff D301).
- [ ] If a new file was added, a row has been added to [`docs-tracker.md`](../docs-tracker.md) with status `[x]`.

> See [`CONVENTIONS.md`](../CONVENTIONS.md) for the full style guide.
