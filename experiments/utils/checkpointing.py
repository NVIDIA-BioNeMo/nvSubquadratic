# TODO: Add licence header

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Utility helpers for checkpointing operations (loading, downloading, uploading, pruning, etc.)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Literal

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch

import wandb


def _select_artifact_with_alias(artifacts, alias: Literal["best", "latest"]):
    """Return the first artifact that has the provided alias.

    Prefers artifacts of type "model"; falls back to any artifact with the alias.

    Args:
        artifacts: The list of artifacts to search through.
        alias: The alias to search for.

    Returns:
        The artifact with the given alias.

    Raises:
        ValueError: If no artifact with the given alias is found.
    """
    assert alias in ["best", "latest"], "Alias must be either best or latest"

    # Prefer artifacts of type "model"
    model_artifacts = [a for a in artifacts if getattr(a, "type", None) == "model"]
    for a in model_artifacts:
        if alias in getattr(a, "aliases", []):
            return a

    # Fallback: any artifact with the alias
    for a in artifacts:
        if alias in getattr(a, "aliases", []):
            return a
    return None


def download_checkpoint(run_path: str, alias: Literal["best", "latest"] = "best") -> str:
    """Download the checkpoint files from the Weights & Biases artifact marked with a given alias (default: "best").

    Args:
        run_path: The W&B run path in the form "entity/project/run_id".
        alias: The artifact alias to download (e.g., "best", "latest"). Defaults to "best".

    Returns:
        The path to the checkpoint file (.ckpt) within the downloaded artifact. It will be stored in the .artifacts/{run_id}/{alias} directory.

    Raises:
        ValueError: If the run has no matching artifact or no .ckpt file is found in the artifact.
    """
    api = wandb.Api()
    run = api.run(run_path)

    artifacts = list(run.logged_artifacts())
    if not artifacts:
        raise ValueError(f"No artifacts found for run '{run_path}'.")

    artifact = _select_artifact_with_alias(artifacts, alias=alias)
    if artifact is None:
        raise ValueError(
            f"No artifact with alias '{alias}' found for run '{run_path}'. "
            f"Available: {[a.name + ':' + ','.join(a.aliases) for a in artifacts]}"
        )

    # Compute output directory
    run_id = run.id
    target_root = Path(f".artifacts/{run_id}/{alias}")
    os.makedirs(target_root, exist_ok=True)

    # Download and locate the .ckpt file
    artifact_dir = Path(artifact.download(root=str(target_root)))

    # Find checkpoint file(s)
    ckpt_files = sorted(artifact_dir.rglob("*.ckpt"))
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in artifact '{artifact.name}:{alias}' at {artifact_dir}.")

    # Heuristic: pick the most recent file by modification time
    ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpt_files[0])


def load_checkpoint_state_dict(ckpt_path: str) -> dict:
    """Load a .ckpt file and return a flat state_dict-like mapping.

    Supports both Lightning checkpoints (with a 'state_dict' key) and plain
    torch.save(state_dict) style checkpoints.
    """
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format for {ckpt_path}: type={type(obj)}")


def align_compiled_keys(
    state_dict: dict[str, torch.Tensor],
    model_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Remap ``state_dict`` keys to match *model_keys*, handling ``_orig_mod`` mismatches.

    ``torch.compile`` wraps parameters under ``_orig_mod``, so a checkpoint
    saved with/without compile may have different key prefixes.  This strips
    the ``._orig_mod.`` segment from both sides and maps by the canonical
    (stripped) name.  Works in both directions (compiled->plain and
    plain->compiled) in a single pass.

    Returns *state_dict* unchanged if the keys already match.
    """
    if set(state_dict.keys()) == model_keys:
        return state_dict

    def _strip(key: str) -> str:
        return key.replace("._orig_mod.", ".")

    model_stripped = {_strip(k): k for k in model_keys}
    return {model_stripped.get(_strip(k), k): v for k, v in state_dict.items()}


class StripCompiledPrefix:
    """Callback for ``start_from_checkpoint`` that aligns ``_orig_mod.`` prefixes.

    Delegates to :func:`align_compiled_keys`.
    """

    def __call__(self, state_dict, model=None, **_kwargs):
        """Align ``_orig_mod.`` prefixes between *state_dict* and *model*."""
        if model is None:
            return state_dict
        return align_compiled_keys(state_dict, set(model.state_dict().keys()))


def _compute_overlapping_slices(target_shape, source_shape):
    """Return a tuple of slice objects selecting overlapping extents for target and source tensors.

    Overlaps are computed per-dimension as min(target_dim, source_dim), starting at 0.
    """
    common_sizes = [min(t, s) for t, s in zip(target_shape, source_shape)]
    target_slices = tuple(slice(0, n) for n in common_sizes)
    source_slices = tuple(slice(0, n) for n in common_sizes)
    return target_slices, source_slices


def load_state_dict_partially(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load checkpoint tensors into model with partial overlap support.

    Behavior:
    - For matching keys with identical shapes: copy directly.
    - For matching keys with different shapes: copy overlapping leading slices on each dimension.
    - Keys in checkpoint not present in model are ignored.
    - Model parameters/buffers without a checkpoint entry are left as-initialized.

    Notes:
    - This function does in-place copies on the model's tensors; it does not replace the state_dict mapping.
    - It supports both parameters and buffers.
    """
    model_state = model.state_dict()

    # Report missing and unexpected keys (summary)
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())
    missing_in_ckpt = sorted(model_keys - ckpt_keys)
    unexpected_in_ckpt = sorted(ckpt_keys - model_keys)
    if missing_in_ckpt:
        print(f"[resume/partial] Missing model keys (not found in checkpoint): {len(missing_in_ckpt)}")
        for k in missing_in_ckpt:
            print(f"  - {k}")
    if unexpected_in_ckpt:
        print(f"[resume/partial] Unexpected checkpoint keys (ignored): {len(unexpected_in_ckpt)}")
        for k in unexpected_in_ckpt:
            print(f"  - {k}")

    loaded_exact = 0
    loaded_partial = 0
    ignored_missing_in_model = 0
    ignored_non_tensor = 0

    for key, source_tensor in state_dict.items():
        if key not in model_state:
            ignored_missing_in_model += 1
            continue

        target_tensor = model_state[key]

        if not isinstance(source_tensor, torch.Tensor) or not isinstance(target_tensor, torch.Tensor):
            ignored_non_tensor += 1
            continue

        # Ensure dtype/device compatibility for copy; cast source to target dtype
        source_tensor = source_tensor.to(dtype=target_tensor.dtype, device=target_tensor.device)

        if target_tensor.shape == source_tensor.shape:
            # Direct copy
            target_tensor.copy_(source_tensor)
            loaded_exact += 1
        else:
            # Partial overlapping copy
            target_slices, source_slices = _compute_overlapping_slices(target_tensor.shape, source_tensor.shape)
            if len(target_slices) != len(target_tensor.shape) or len(source_slices) != len(source_tensor.shape):
                # Safety: skip malformed slicing
                continue
            # Report what slice will be copied
            common_sizes = [s.stop for s in target_slices]
            dim_desc = ", ".join(str(n) for n in common_sizes)
            print(
                (
                    f"[resume/partial] Partially loaded '{key}': "
                    f"source{tuple(source_tensor.shape)} -> target{tuple(target_tensor.shape)}; "
                    f"copied [:{dim_desc}]"
                )
            )
            target_tensor[target_slices].copy_(source_tensor[source_slices])
            loaded_partial += 1
    # After in-place copies, no need to call load_state_dict again; parameters/buffers are updated.
    total_ckpt_tensors = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
    print(
        f"[resume/partial] Summary: exact_loaded={loaded_exact}, partial_loaded={loaded_partial}, "
        f"ignored_missing_in_model={ignored_missing_in_model}, ignored_non_tensor={ignored_non_tensor}, "
        f"ckpt_tensor_entries={total_ckpt_tensors}"
    )


def print_state_dict_summary(state_dict: dict) -> None:
    """Print a summary of the state dict.

    Args:
        state_dict: The state dict to print.
    """
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            # Non-tensor entries (e.g., scalars)
            print(f"{key}: type={type(value).__name__}")


def preview_state_dict_compatibility(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], *, max_list: int = 20
) -> None:
    """Print a detailed preview of how a checkpoint matches the model.

    Reports:
    - Missing model keys (not found in checkpoint)
    - Unexpected checkpoint keys (not present in model)
    - Among intersecting keys, counts of exact shape matches vs mismatches, and examples
    """
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())

    missing_in_ckpt = sorted(model_keys - ckpt_keys)
    unexpected_in_ckpt = sorted(ckpt_keys - model_keys)

    print(f"[resume/compat] Model params/buffers: {len(model_keys)} | Checkpoint entries: {len(ckpt_keys)}")
    if missing_in_ckpt:
        print(f"[resume/compat] Missing in checkpoint: {len(missing_in_ckpt)}")
        for k in missing_in_ckpt[:max_list]:
            print(f"  - {k}")
        if len(missing_in_ckpt) > max_list:
            print(f"  ... (+{len(missing_in_ckpt) - max_list} more)")
    if unexpected_in_ckpt:
        print(f"[resume/compat] Unexpected in checkpoint: {len(unexpected_in_ckpt)}")
        for k in unexpected_in_ckpt[:max_list]:
            print(f"  - {k}")
        if len(unexpected_in_ckpt) > max_list:
            print(f"  ... (+{len(unexpected_in_ckpt) - max_list} more)")

    intersect = sorted(model_keys & ckpt_keys)
    exact = 0
    mismatch = []
    for k in intersect:
        v_model = model_state[k]
        v_ckpt = state_dict[k]
        if not isinstance(v_model, torch.Tensor) or not isinstance(v_ckpt, torch.Tensor):
            continue
        if tuple(v_model.shape) == tuple(v_ckpt.shape):
            exact += 1
        else:
            mismatch.append((k, tuple(v_ckpt.shape), tuple(v_model.shape)))
    print(
        f"[resume/compat] Intersect keys: {len(intersect)} | exact shape matches: {exact} | mismatches: {len(mismatch)}"
    )
    if mismatch:
        print("[resume/compat] Shape mismatches (ckpt -> model):")
        for k, s_ckpt, s_model in mismatch[:max_list]:
            print(f"  - {k}: {s_ckpt} -> {s_model}")
        if len(mismatch) > max_list:
            print(f"  ... (+{len(mismatch) - max_list} more)")


class WandbSelectiveCheckpointUploader(pl_callbacks.Callback):
    """Upload checkpoints to W&B with per-scheduler-phase best/latest tracking.

    In addition to the global ``best`` and ``latest`` aliases, this callback
    uploads ``last.ckpt`` with per-phase aliases (e.g. ``stable-best``,
    ``decay-latest``) so that the best and most recent checkpoint for each
    scheduler regime are preserved in W&B artifacts.

    Phase boundaries are derived from the scheduler config and passed as
    ``phase_boundaries``.  When empty, only global best/latest are uploaded
    (backward-compatible behavior).
    """

    def __init__(
        self,
        upload_best: bool = True,
        upload_last: bool = True,
        remove_local_after_upload: bool = False,
        keep_last_k_versions: int = 2,
        phase_boundaries: dict[str, tuple[int, int]] | None = None,
        mode: str = "max",
    ):
        """Configure selective checkpoint uploads to W&B.

        Args:
            upload_best: Upload the global-best checkpoint (alias ``"best"``).
            upload_last: Upload the latest checkpoint (alias ``"latest"``).
            remove_local_after_upload: Delete local file after successful upload.
            keep_last_k_versions: Minimum unaliased artifact versions to retain
                when pruning.  Aliased versions are **always** protected.
            phase_boundaries: Maps phase name to ``(start_step, end_step)``
                inclusive/exclusive.  E.g. ``{"stable": (1000, 8000), "decay":
                (8000, 10000)}``.  Empty or ``None`` disables per-phase tracking.
            mode: ``"max"`` or ``"min"`` — whether higher or lower metric is better.
        """
        super().__init__()
        self.upload_best = upload_best
        self.upload_last = upload_last
        self.remove_local_after_upload = remove_local_after_upload
        self.keep_last_k_versions = max(int(keep_last_k_versions), 1)
        self.phase_boundaries: dict[str, tuple[int, int]] = phase_boundaries or {}
        self.mode = mode
        # Track per-alias content hashes to avoid re-uploading identical content
        # even if the file path is reused (e.g., latest checkpoint overwrites).
        # Structure: {alias: last_sha256_hex}
        self._uploaded_hashes: dict[str, str] = {}
        # Per-phase best metric tracking: {phase_name: best_score}
        self._phase_best: dict[str, float] = {
            name: float("-inf") if mode == "max" else float("inf") for name in self.phase_boundaries
        }

    def _current_phase(self, global_step: int) -> str | None:
        """Return the phase name for *global_step*, or ``None`` if no phase matches."""
        for name, (start, end) in self.phase_boundaries.items():
            if start <= global_step < end:
                return name
        return None

    def _is_phase_improvement(self, phase: str, score: float) -> bool:
        """Return ``True`` if *score* improves the per-phase best for *phase*."""
        prev = self._phase_best[phase]
        if self.mode == "max":
            return score > prev
        return score < prev

    def _file_sha256(self, path: str, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _maybe_upload(self, run: "wandb.sdk.wandb_run.Run", path: str, alias: str):
        if not path:
            msg = f"[checkpoint/upload][error] No path provided for alias='{alias}'"
            print(msg)
            raise RuntimeError(msg)
        if not os.path.isfile(path):
            msg = f"[checkpoint/upload][warn] Path is not a file (maybe pruned already): '{path}' for alias='{alias}'"
            print(msg)
            return
        # Compute hash and skip if identical content already uploaded for this alias
        try:
            file_hash = self._file_sha256(path)
        except Exception as e:
            print(f"[checkpoint/upload][warn] Failed to hash '{path}' for alias='{alias}': {e}; proceeding to upload")
            file_hash = None

        last_hash = self._uploaded_hashes.get(alias)
        if file_hash is not None and last_hash == file_hash:
            print(f"[checkpoint/upload] Content unchanged for alias='{alias}', skipping upload of '{path}'")
            return
        art = wandb.Artifact(name=f"model-{run.id}", type="model")
        art.add_file(path)
        run.log_artifact(art, aliases=[alias])
        if file_hash is not None:
            self._uploaded_hashes[alias] = file_hash
        if self.remove_local_after_upload:
            try:
                os.remove(path)
            except OSError:
                pass

    def _prune_old_versions(self, run: "wandb.sdk.wandb_run.Run", artifact_name: str):
        try:
            # Prefer robust pruning via API collection (covers all versions)
            api = wandb.Api()
            entity = getattr(run, "entity", None)
            project = getattr(run, "project", None)
            versions = []
            if entity and project:
                locator = f"{entity}/{project}/{artifact_name}"
                try:
                    versions = list(api.artifact_versions("model", locator))
                    print(f"[checkpoint/prune] Using artifact_versions for {locator} → {len(versions)} versions")
                except Exception as e:
                    print(f"[checkpoint/prune][warn] artifact_versions lookup failed for {locator}: {e}.")

            # Fallback: use artifacts visible from this run only
            if not versions:
                api_run = None
                if entity and project:
                    try:
                        api_run = api.run(f"{entity}/{project}/{run.id}")
                    except Exception as e:
                        print(f"[checkpoint/prune][warn] api.run lookup failed for {run.id}: {e}.")
                if api_run is not None:
                    versions = [
                        a
                        for a in api_run.logged_artifacts()
                        if isinstance(getattr(a, "name", None), str) and a.name.split(":", 1)[0] == artifact_name
                    ]
                print(f"[checkpoint/prune] Fallback to logged_artifacts() → {len(versions)} versions")

            if len(versions) <= self.keep_last_k_versions:
                print(f"[checkpoint/prune] Nothing to prune (have {len(versions)} ≤ keep {self.keep_last_k_versions})")
                return

            def _ver_num(a):
                try:
                    return int(str(getattr(a, "version", "v0")).lstrip("v"))
                except Exception:
                    return -1

            # Newest first
            versions.sort(key=_ver_num, reverse=True)

            # Preserve every version that carries at least one alias
            alias_protected = set()
            for a in versions:
                aliases = set(getattr(a, "aliases", []) or [])
                if aliases:
                    alias_protected.add(getattr(a, "id", a))
            if alias_protected:
                print(f"[checkpoint/prune] Protected by alias: {len(alias_protected)}")

            to_keep = []
            for a in versions:
                aid = getattr(a, "id", a)
                if aid in alias_protected:
                    to_keep.append(a)
                if len(to_keep) >= self.keep_last_k_versions:
                    break

            # If protected set already exceeds keep budget, keep all protected and skip deletion
            keep_ids = {getattr(a, "id", a) for a in to_keep} | alias_protected
            if len(keep_ids) < self.keep_last_k_versions:
                # Fill remaining keep slots with newest others
                for a in versions:
                    aid = getattr(a, "id", a)
                    if aid in keep_ids:
                        continue
                    keep_ids.add(aid)
                    if len(keep_ids) >= self.keep_last_k_versions:
                        break

            kept, deleted = [], []
            for a in versions:
                aid = getattr(a, "id", a)
                if aid in keep_ids:
                    kept.append(a)
                    continue
                try:
                    a.delete()
                    deleted.append(a)
                except Exception:
                    # Ignore failures (e.g., permissions, race conditions)
                    print(
                        f"[checkpoint/prune][warn] Failed to delete artifact id={aid} version={getattr(a, 'version', '?')}"
                    )

            def _fmt(a):
                return f"id={getattr(a, 'id', '?')} ver={getattr(a, 'version', '?')} aliases={','.join(getattr(a, 'aliases', []) or [])}"

            print(f"[checkpoint/prune] Keep count={len(kept)} (budget={self.keep_last_k_versions})")
            for a in kept[:5]:
                print(f"  keep: {_fmt(a)}")
            if len(kept) > 5:
                print(f"  ... (+{len(kept) - 5} more kept)")

            print(f"[checkpoint/prune] Deleted count={len(deleted)}")
            for a in deleted[:5]:
                print(f"  delete: {_fmt(a)}")
            if len(deleted) > 5:
                print(f"  ... (+{len(deleted) - 5} more deleted)")
        except Exception as e:
            # Report errors explicitly but do not crash training
            print(f"[checkpoint/prune][error] {type(e).__name__}: {e}")
        finally:
            # Explicitly release the API client to free HTTP sessions/caches
            try:
                del api
            except UnboundLocalError:
                pass

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Upload selected checkpoints at the end of validation if appropriate."""
        # Only upload during actual training runs to avoid alias churn during post-training validate
        if getattr(trainer, "sanity_checking", False):
            return
        # Lightning 1.x/2.x compatible detection of fitting state
        state = getattr(trainer, "state", None)
        fn = getattr(state, "fn", None)
        fn_str = str(fn).lower() if fn is not None else ""
        if ("fit" not in fn_str) and ("fitting" not in fn_str):
            return

        if not trainer.is_global_zero:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        run = trainer.logger.experiment
        ckpt_cb = next((cb for cb in trainer.callbacks if isinstance(cb, pl_callbacks.ModelCheckpoint)), None)
        if ckpt_cb is None:
            print("[checkpoint/upload][warn] No ModelCheckpoint callback found; skipping upload")
            return

        # --- Global best / latest (backward-compatible) -----------------------
        if self.upload_best:
            best_path = getattr(ckpt_cb, "best_model_path", None)
            if best_path:
                self._maybe_upload(run, best_path, alias="best")
            else:
                print("[checkpoint/upload][warn] upload_best=True but best_model_path is empty; will retry later")
        if self.upload_last:
            last_path = getattr(ckpt_cb, "last_model_path", None) or getattr(ckpt_cb, "last_model", None)
            if last_path:
                self._maybe_upload(run, last_path, alias="latest")
            else:
                print("[checkpoint/upload][warn] upload_last=True but last_model_path is empty; will retry later")

        # --- Per-phase best / latest ------------------------------------------
        if self.phase_boundaries:
            phase = self._current_phase(trainer.global_step)
            if phase is not None:
                last_path = getattr(ckpt_cb, "last_model_path", None) or getattr(ckpt_cb, "last_model", None)
                if last_path and os.path.isfile(last_path):
                    self._maybe_upload(run, last_path, alias=f"{phase}-latest")

                    score = getattr(ckpt_cb, "current_score", None)
                    if score is not None:
                        score_val = float(score)
                        if self._is_phase_improvement(phase, score_val):
                            self._phase_best[phase] = score_val
                            self._maybe_upload(run, last_path, alias=f"{phase}-best")
                            print(
                                f"[checkpoint/upload] New {phase}-best: {score_val:.4f} (step {trainer.global_step})"
                            )

        # Prune older remote versions to keep storage bounded
        self._prune_old_versions(run, artifact_name=f"model-{run.id}")
