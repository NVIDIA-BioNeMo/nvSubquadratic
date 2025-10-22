# Overview

- **Goal**: Prepare reproducible support for the diffusion experiment by provisioning mamba env `nvsq` from `pyproject.toml` dependencies and running a smoke test.
- **Stakeholders**: dknigge (requester), codex-agent (executor).
- **Constraints**: Follow Autonomous Agent Playbook, operate on dedicated feature branch, document every step, leverage dataset staged at `~/project_dir/huggingface`.
- **Active Branch**: `agent/diffusion-smoketest/20251022-1925`.
- **Active Environment**: mamba env `nvsq` (`/gpfs/home1/dknigge/miniforge3/envs/nvsq`, Python 3.11.14).
- **Key Files / Directories**:
  - `pyproject.toml` — canonical dependency list.
  - `examples/` — experiment entry points and configs (diffusion target to confirm).
  - `nvsubquadratic/` — library modules consumed by experiments.
- **Open Questions**:
  - Confirm exact diffusion experiment script/config to execute for smoketest.
  - Validate access path for ImageNet dataset under `~/project_dir/huggingface`.
