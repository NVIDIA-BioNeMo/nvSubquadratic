# Findings Log

- 2025-10-22T19:26Z — Initialized log; no technical findings yet.
- 2025-10-22T19:29Z — Verified dataset directory `~/project_dir/huggingface/imagenet/imagenet-1k` is present.
- 2025-10-22T19:29Z — Diffusion experiment entry point not yet found in repository; need to identify correct script/config.
- 2025-10-22T19:44Z — Created mamba environment `nvsq` with Python 3.11.14.
- 2025-10-22T19:44Z — Installed project editable package and pyproject dependencies inside `nvsq`.
- 2025-10-22T19:45Z — `subquadratic_ops` import fails in `nvsq`; dependency likely requires internal index credentials.
