# Documentation Conventions

## Goal

External collaborators should be able to read any module in `nvsubquadratic/`
or `experiments/` alongside the research paper and understand:

1. **What** the class/function does and **why** it exists.
1. The **math** it implements (notation from the paper where applicable).
1. The **shape contract** of every tensor argument.
1. How it fits into the larger architecture.

______________________________________________________________________

## Docstring style

All public classes, functions, and methods use **Google-style docstrings**.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply RMS normalisation over the last dimension.

    Args:
        x: Input tensor of shape ``[*leading, D]``.

    Returns:
        torch.Tensor: Normalised tensor, same shape and dtype as ``x``.

    Raises:
        RuntimeError: If ``x.shape[-1]`` does not equal ``self.dim``.
    """
```

### Rules

| What                                    | Where                                                      |
| --------------------------------------- | ---------------------------------------------------------- |
| Math / motivation / paper context       | **Module docstring** or **class docstring**                |
| Parameter descriptions + shapes         | `Args:` block on the method                                |
| Return shape + dtype                    | `Returns:` block                                           |
| Docstrings containing `\` (backslashes) | Use `r"""..."""` (required by ruff D301)                   |
| Single-line `__init__` docstrings       | Allowed only when the class docstring covers all arguments |

______________________________________________________________________

## PR convention

> **Any PR that adds or modifies a public class or function in `nvsubquadratic/`
> or `experiments/` must update the docstring of every touched symbol.**

This means:

- New file → full module docstring + docstrings on all public classes/functions.
- Existing file touched → update only the affected symbols; do not leave
  neighbouring docstrings stale.
- Renamed parameter → rename in the docstring too (never paper over a
  misleading name with a docstring explanation).

______________________________________________________________________

## Automated enforcement

### 1. ruff (already active)

The pre-commit hook runs `ruff` with the `D` rule-set enabled.  It catches:

- `D100` Missing docstring in public module
- `D101` Missing docstring in public class
- `D102` Missing docstring in public method
- `D103` Missing docstring in public function
- `D301` Use `r"""` if backslashes appear in a docstring

If your commit is rejected by ruff with a `D` error, add or fix the docstring
before pushing.

### 2. CI diff-check (recommended addition)

Add the following job to `.github/workflows/ci.yml` (or your equivalent):

```yaml
- name: Docstring coverage on changed files
  run: |
    # Collect Python files touched by this PR
    git diff --name-only origin/main...HEAD \
      | grep -E '^(nvsubquadratic|experiments)/.*\.py$' \
      > changed.txt

    # Fail if any changed file has missing public docstrings
    if [ -s changed.txt ]; then
      xargs ruff check --select D100,D101,D102,D103,D417 < changed.txt
    fi
```

This restricts the `D` check to **files actually changed in the PR**, so it
never blocks unrelated legacy files.

### 3. CONVENTIONS.md review checklist

Add a pull-request template (`.github/pull_request_template.md`) that
reminds authors:

```markdown
## Documentation checklist

- [ ] Every new public class has a module-level or class-level docstring
      explaining *what it does* and *why*.
- [ ] Every new public method/function has Args: and Returns: blocks with
      tensor shapes.
- [ ] Math notation is consistent with the paper (or a comment explains any
      difference).
- [ ] Docstrings containing backslashes use `r"""..."""`.
```

### 4. Updating `docs-tracker.md`

When a PR introduces a **new file** in scope (`nvsubquadratic/` or
`experiments/`), add a row to the relevant table in `docs-tracker.md` with
status `[x]` and a one-line note.  This keeps the tracker current without a
separate documentation PR.

______________________________________________________________________

## What does *not* need a docstring

- Private helpers prefixed with `_` (encouraged but not required).
- `__repr__` / `extra_repr` methods — a brief one-liner is enough.
- Test files under `tests/` — descriptive test names suffice.
- Config files under `experiments/` that contain only a `cfg = ...` assignment.
