# Changes

## Queued
- None

## Work in Progress
- None
#### [T20251021-2259-diffusion-transfer]
- created_at: 2025-10-21T22:58:13Z
- updated_at: 2025-10-21T23:16:30Z
- status: done
- owner: codex-agent
- summary: Build repository knowledge base and define diffusion pipeline transfer plan.
- rationale: Need clear understanding of `nvSubquadratic` before porting diffusion features from `../ccnn_v2`.
- dependencies: none
- validation_plan: Document architecture insights in `.agents_docs`, outline transfer steps with required modules/tests.

**Log**
- 2025-10-21T22:58:13Z queued — Captured goal to evaluate repo and plan diffusion transfer.
- 2025-10-21T22:58:30Z work_in_progress — Created branch `agent/diffusion-transfer/20251021-225813` and initialized agent documentation.
- 2025-10-21T23:05:00Z work_in_progress — Documented nvSubquadratic architecture and experiment tooling in `.agents_docs`.
- 2025-10-21T23:12:00Z work_in_progress — Reviewed `../ccnn_v2` diffusion network and Lightning wrapper to inform transfer strategy.
- 2025-10-21T23:16:30Z done — Captured transfer plan in `.agents_docs/findings.md`; ready for implementation tickets.

## Review / Test
- None

## Done
#### [T20251021-2316-diffusion-imagenet-port]
- created_at: 2025-10-21T23:16:30Z
- updated_at: 2025-10-21T23:46:30Z
- status: done
- owner: codex-agent
- summary: Port ImageNet diffusion pipeline (datamodule, network, Lightning wrapper, configs) from `../ccnn_v2`.
- rationale: Enable nvSubquadratic to run the proven simple diffusion experiment while reusing shared HF cache.
- dependencies: none
- validation_plan: Ensure new modules import correctly, configs resolve via `examples/run.py`, and document HF cache usage.

**Log**
- 2025-10-21T23:16:30Z queued — Captured implementation work items for ImageNet diffusion transfer.
- 2025-10-21T23:17:00Z work_in_progress — Beginning code port of diffusion components into `nvsubquadratic`.
- 2025-10-21T23:28:00Z work_in_progress — Added diffusion network, Lightning wrapper, ImageNet datamodule, and experiment config leveraging shared HF cache.
- 2025-10-21T23:30:00Z work_in_progress — Updated `pyproject.toml` to include `datasets` dependency required by the ImageNet datamodule.
- 2025-10-21T23:32:00Z review_test — Import smoke test blocked by missing `torch`; awaiting dependency install before rerun.
- 2025-10-21T23:46:30Z done — Installed dependencies in `nvsq` and validated `get_config()` + instantiation via `conda run -n nvsq python -c "..."`.

## Done
#### [T20251021-2316-diffusion-imagenet-port]
- created_at: 2025-10-21T23:16:30Z
- updated_at: 2025-10-21T23:16:30Z
- status: queued
- owner: codex-agent
- summary: Port ImageNet diffusion pipeline (datamodule, network, Lightning wrapper, configs) from `../ccnn_v2`.
- rationale: Enable nvSubquadratic to run the proven simple diffusion experiment while reusing shared HF cache.
- dependencies: none
- validation_plan: Ensure new modules import correctly, configs resolve via `examples/run.py`, and document HF cache usage.

**Log**
- 2025-10-21T23:16:30Z queued — Captured implementation work items for ImageNet diffusion transfer.
