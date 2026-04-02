## Summary

<!-- What does this PR change and why? -->

## Environment setup

Create the conda environment (required to run tests):

```bash
bash setup_conda_env.sh
conda activate nvsubquadratic
```

## Pre-commit

Run locally before pushing (hooks also run on `git push` if installed):

```bash
pre-commit install
pre-commit install --hook-type pre-push
# Optional: run pre-push checks without pushing
pre-commit run --hook-stage pre-push --all-files
```

**Required for CI:** Keep the following line in your PR description (the PR Description Check workflow looks for it):

> Pre-commit checks passed

_(Replace with accurate wording after you run hooks; e.g. if something is intentionally skipped, coordinate with reviewers.)_

## Testing

<!-- How did you test this? -->
