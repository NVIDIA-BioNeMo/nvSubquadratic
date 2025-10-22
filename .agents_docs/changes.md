# Change Tickets

## Queued

_None_

## Work in Progress

#### [TICKET T20251022-1926-diffusion-smoketest]
- created_at: 2025-10-22T19:26Z
- updated_at: 2025-10-22T19:45Z
- status: work_in_progress
- owner: codex-agent
- summary: Provision env `nvsq` and execute diffusion experiment smoketest.
- rationale: Validates that the diffusion pipeline is runnable with documented dependencies.
- dependencies: none
- validation_plan: Create `nvsq` environment via mamba from `pyproject.toml`; verify dataset availability; run diffusion smoketest command; capture outputs.

**Log**
- 2025-10-22T19:26Z queued — Goal captured from user request; awaiting planning and execution.
- 2025-10-22T19:27Z work_in_progress — Branch `agent/diffusion-smoketest/20251022-1925` created; beginning discovery and environment setup.
- 2025-10-22T19:29Z work_in_progress — ImageNet dataset confirmed at `~/project_dir/huggingface`; diffusion entry point still unclear from repo layout.
- 2025-10-22T19:44Z work_in_progress — Provisioned mamba env `nvsq` (Python 3.11) and installed project dependencies with `pip install -e .`.
- 2025-10-22T19:45Z work_in_progress — `subquadratic_ops` import missing and no diffusion entry script found; smoketest blocked pending user guidance (next check-in once instructions arrive).

## Review / Test

_None_

## Done

_None_
