# Experiment: Diffusion Smoketest
- timestamp_started: 2025-10-22T19:25Z
- owner: codex-agent
- related_ticket: T20251022-1926-diffusion-smoketest
- hypothesis: With the provided ImageNet dataset and project dependencies installed, the diffusion experiment script should execute at least a short smoketest run.

## Plan
- locate diffusion experiment entry point within the repository or associated scripts.
- validate dataset availability at `~/project_dir/huggingface/imagenet`.
- run the diffusion smoketest via `mamba run -n nvsq python <entry-point> --smoke` (exact command pending discovery).

## Commands Executed
- `ls ~/project_dir/huggingface`
- `find . -iname '*diffusion*'`
- `rg "diffusion"`
- `mamba run -n nvsq python -c "import subquadratic_ops"`

## Observations
- ImageNet dataset directory exists as `~/project_dir/huggingface/imagenet/imagenet-1k`.
- Repository lacks any obvious diffusion experiment scripts or configs; searches for "diffusion" returned nothing.
- `subquadratic_ops` is unavailable in the `nvsq` environment; installation appears to require private credentials.

## Outcome
- status: blocked
- timestamp_completed: 2025-10-22T19:45Z
- notes: Unable to execute a diffusion smoketest without a defined entry point or the private `subquadratic_ops` dependency. Awaiting guidance on the correct script and installation credentials.
