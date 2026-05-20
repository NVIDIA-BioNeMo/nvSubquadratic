# Skill evals — future CI integration

This directory holds eval cases for the `hyenand-retrofit` skill. As of writing
they function as a **spec** for the skill's intended behavior, not a runnable
test suite — there is no harness in this repo that executes them. This README
is the design for wiring that up later.

## What's here

| File          | Purpose                                                                                                                                                                                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evals.json`  | Five eval cases (native pure swap, native hybrid, foreign 2D ViT, foreign 3D feature-map U-Net, foreign 1D causal LM). Each case has a prompt, the input files the agent should see, an `expected_output` description, and a list of grep-based assertions against the file the agent writes. |
| `inputs/*.py` | Standalone test fixtures the agent retrofits. Each runs end-to-end (`python <file>` produces a forward-pass shape) so the harness can sanity-check the inputs themselves before evaluating the agent's output.                                                                                |

The evals together cover the four-axis grid from `SKILL.md`:

| Eval                     | data_dim | causal | host layout | prefix tokens |
| ------------------------ | -------- | ------ | ----------- | ------------- |
| `foreign-timm-vit`       | 2        | False  | tokens      | 1 (CLS)       |
| `foreign-3d-feature-map` | 3        | False  | feature_map | 0             |
| `foreign-1d-causal-lm`   | 1        | True   | tokens      | 0             |

`native-pure-swap` and `native-hybrid` cover the native path where the host
already uses nvSubquadratic builders.

## Why CI for these

Three reasons, in order of value:

1. **Regression guard.** When SKILL.md is edited, the description, the four-axis
   decision tree, or the adapter skeleton may drift in a way that breaks a
   retrofit pattern that used to work. Running the evals on every skill change
   catches that early.
1. **Model-drift catch.** As Claude models update (4.7 → 4.8 → ...), the same
   skill text may be interpreted slightly differently. A weekly cron picks this
   up before the next user does.
1. **Documented behavior.** The asserts encode "what counts as a correct
   retrofit" in a machine-checkable form. Reviewers don't have to read SKILL.md
   to know what the skill claims to do — they can read evals.json.

## Recommended shape

Separate workflow, not bolted onto the main GPU CI in
[../../../.github/workflows/ci.yml](../../../.github/workflows/ci.yml). Reasons:

- The skill evals don't need a GPU — they only produce code, then grep it.
  `ubuntu-latest` is enough. No reason to bloat the Colossus-runner queue.
- The skill evals are LLM-driven and inherently flakier than the existing
  pytest suite. Keeping them in a separate workflow means a flake here doesn't
  block a code PR.
- The triggers are different: the GPU pipeline runs on every code change; the
  skill evals only need to run when the skill itself changes.

### Triggers

```yaml
on:
  pull_request:
    paths: ['.claude/skills/hyenand-retrofit/**']
  workflow_dispatch:          # manual button for ad-hoc runs
  schedule:
    - cron: '0 6 * * 1'       # weekly Monday 06:00 UTC — catches model drift
```

The `paths:` filter keeps cost down: skill changes are infrequent, code-only
PRs don't pay for skill evals.

### Two-layer cheap-first design

| Layer                                                                                              | Cost                            | When to run                                                 | Tool                                                  |
| -------------------------------------------------------------------------------------------------- | ------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| **Trigger eval** — does the skill description cause Claude to load the skill for the eval prompts? | ~5 sec, ~$0.001 per query       | Every PR touching the skill                                 | skill-creator's `run_eval.py` (already exists; reuse) |
| **Full eval** — does the agent produce a file that passes the asserts?                             | ~1-5 min, ~$0.05-$0.20 per eval | `pull_request` to main and `workflow_dispatch`; weekly cron | Custom harness, see sketch below                      |

Trigger eval is cheap enough to run on every skill PR. Full eval is reserved
for main-PR and manual triggers because a single run is ~$0.50-$1 total and
takes 5-25 minutes serial (or ~1-5 min with a GHA matrix).

### Parallelization

Run the five evals as a GHA matrix:

```yaml
strategy:
  fail-fast: false            # report all failures, not just the first
  matrix:
    eval: [native-pure-swap, native-hybrid, foreign-timm-vit, foreign-3d-feature-map, foreign-1d-causal-lm]
```

Each matrix job runs one eval. Wall-clock is bounded by the slowest single eval
(~5 min) at the cost of 5× concurrent ubuntu-latest runners.

### Flake handling

LLM outputs are not deterministic. The grep patterns in `evals.json` are
deliberately tolerant of surface-level variation (`data_dim\s*=\s*3`, not
`data_dim=3` exactly), but flakes will still happen. Two mitigations:

- **Retry once on failure.** A retry that still fails is a real failure;
  a retry that passes is a flake (note it for tracking, don't block the PR).
- **Treat the workflow as advisory at first.** Mark it non-blocking
  (`continue-on-error: true` or a separate non-required check) until you have
  a few weeks of data on its flake rate. Promote to required once stable.

## Open questions to resolve before building

1. **API key as a GHA secret.** Does the NVIDIA org policy allow
   `ANTHROPIC_API_KEY` in GitHub Actions secrets? If not, the full-eval layer
   is dead in the water and only trigger evals (which can run via the local
   `claude` CLI configured with a personal token) are viable.
1. **Cost budget.** At ~$1/full-run × N PRs/week + cron, is the budget
   acceptable? If not, drop the per-PR trigger and run only on
   `workflow_dispatch` + weekly cron.
1. **Blocking vs advisory.** Should a failing eval block a PR merge?
   Recommend: advisory for the first few months, then promote to required if
   the flake rate is low.
1. **Skill maintenance frequency.** If the skill is touched ~once a quarter,
   the CI overhead may not be worth it vs. running `python run_evals.py`
   manually as part of any skill edit.

## Minimal harness sketch

The full eval harness is ~50 lines of Python. Core loop:

```python
# .claude/skills/hyenand-retrofit/evals/run_evals.py
import json, re, subprocess, pathlib, sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
EVALS = json.load(open(pathlib.Path(__file__).parent / "evals.json"))["evals"]


def run_one(eval_case):
    # Spawn claude -p with the skill auto-loaded via .claude/ discovery
    prompt = eval_case["prompt"]
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Find the file the agent wrote (delta vs git HEAD).
    written = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout
    new_files = [
        line.split()[-1] for line in written.splitlines() if line.startswith("?? ")
    ]
    output_path = REPO_ROOT / new_files[0]
    text = output_path.read_text()

    fails = []
    for a in eval_case["assertions"]:
        kind, pattern = (
            a["check"].split(":", 1) if ":" in a["check"] else (a["check"], "")
        )
        if kind == "grep_output" and not re.search(pattern, text):
            fails.append(a["text"])
        elif kind == "grep_output_negative" and re.search(pattern, text):
            fails.append(f"(negative) {a['text']}")
    return fails, output_path


if __name__ == "__main__":
    failed = 0
    for e in EVALS:
        fails, path = run_one(e)
        status = "PASS" if not fails else "FAIL"
        print(f"[{status}] #{e['id']} {e['name']}  ({path.name})")
        for f in fails:
            print(f"    - {f}")
        failed += bool(fails)
    sys.exit(1 if failed else 0)
```

Caveats this sketch glosses over:

- Cleaning up the agent's written file between evals (so the next eval's
  "files written since HEAD" doesn't include the previous output)
- Handling the `original_unmodified` and `sibling_location` assertion kinds
  (which need filesystem inspection, not grep)
- Per-eval working-directory isolation if evals are matrix-parallelized
- Passing the eval's `files` input list to the agent so it knows what to read

The skill-creator plugin's
[scripts/run_eval.py](../../../../.claude/plugins/marketplaces/claude-plugins-official/plugins/skill-creator/skills/skill-creator/scripts/run_eval.py)
already handles the `claude -p` spawn correctly (uuid-named command file for
skill discovery, partial-message stream parsing for fast trigger detection).
Cannibalize it rather than re-implementing.

## Does this interfere with the skill itself?

No. The skill loader (Claude Code's skill discovery) reads `SKILL.md` matched
by its frontmatter. README files, eval definitions, and test fixtures in
`evals/` are inert from the skill's perspective. You can freely add docs,
helper scripts, and CI config under this directory without affecting how the
skill triggers or executes.

The only files in this directory that the skill *might* surface to a running
agent are those the agent itself navigates to (e.g., reading
`inputs/tiny_vit_attention.py` because the eval prompt named it). That's
intended — the inputs are part of the eval's surface area.
