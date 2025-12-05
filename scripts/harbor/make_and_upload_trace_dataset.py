#!/usr/bin/env python3
"""
Export traces from a Harbor job directory into a Hugging Face dataset and upload it.

This mirrors the end-of-run trace export/upload behavior used by HPC Launch
datagen flows, but works as a simple standalone script.

Examples:

  # Export from a completed Harbor job directory and push to HF
  python -m scripts.harbor.make_and_upload_trace_dataset \
    --job_dir /path/to/jobs/codecontests_glm46 \
    --repo_id DCAgent/code_contests-GLM-4.6-traces \
    --episodes last \
    --filter success

Notes:
- Requires Harbor to be installed/importable and a completed Harbor job dir.
- Auth to HF via HF_TOKEN env var or `huggingface-cli login`.
- Optional Supabase registration can be skipped with --skip_register.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _install_safe_episode_guard() -> None:
    """Patch Harbor's episode discovery to skip invalid agent directories."""

    try:
        from harbor.utils import traces_utils  # type: ignore
    except Exception:
        return

    original_find = getattr(traces_utils, "find_episode_dirs", None)
    if original_find is None:
        return
    if getattr(original_find, "__dcagent_safe__", False):
        return

    def safe_find_episode_dirs(trial_dir: Path):
        episodes_root = Path(trial_dir) / "agent"
        if episodes_root.exists() and not episodes_root.is_dir():
            print(
                f"[trace-export] Skipping trial {trial_dir} because {episodes_root} is not a directory."
            )
            return []
        try:
            return original_find(trial_dir)
        except NotADirectoryError as exc:
            print(f"[trace-export] Skipping trial {trial_dir}: {exc}")
            return []

    safe_find_episode_dirs.__dcagent_safe__ = True  # type: ignore[attr-defined]
    traces_utils.find_episode_dirs = safe_find_episode_dirs


def _import_export_traces():
    """Resolve the export_traces helper, preferring the dc-agent wrapper."""
    try:
        from database.unified_db.utils import export_traces  # type: ignore
        return export_traces
    except Exception:
        pass
    try:
        from harbor.utils.traces_utils import export_traces  # type: ignore
        return export_traces
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Harbor is not available. Install it (pip install -e ../harbor) "
            f"or ensure it's on PYTHONPATH. Import error: {exc}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make and upload a trace dataset from a Harbor job directory",
    )
    p.add_argument("--job_dir", required=True, help="Path to Harbor job directory")
    p.add_argument("--repo_id", required=True, help="Target HF dataset repo (org/name)")
    p.add_argument(
        "--episodes",
        choices=["all", "last"],
        default="last",
        help="Which episodes to export per trial (default: last)",
    )
    p.add_argument(
        "--filter",
        choices=["success", "failure", "none"],
        default="none",
        help="Filter exported episodes (default: none)",
    )
    p.add_argument(
        "--to_sharegpt",
        action="store_true",
        help="Export in ShareGPT-style format where applicable",
    )
    p.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Prepend reasoning_content (if available) to assistant replies inside <think> tags",
    )
    p.add_argument(
        "--dataset_type",
        default="SFT",
        help="Dataset type for registration (SFT or RL). Default: SFT",
    )
    p.add_argument(
        "--skip_register",
        action="store_true",
        help="Skip Supabase registration after upload",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise SystemExit(f"job_dir does not exist or is not a directory: {job_dir}")

    export_traces = _import_export_traces()
    _install_safe_episode_guard()

    # Map filter flag to traces_utils argument
    success_filter = None if args.filter == "none" else args.filter

    print(f"[trace-export] Exporting traces from: {job_dir}")
    ds = export_traces(
        root=job_dir,
        recursive=True,
        episodes=args.episodes,
        to_sharegpt=bool(args.to_sharegpt),
        repo_id=None,
        push=False,
        verbose=True,
        success_filter=success_filter,
        include_reasoning=bool(args.include_reasoning),
    )

    # Push to HF and optionally register in DB
    print(f"[trace-export] Exported {len(ds)} rows. Uploading to {args.repo_id}...")
    try:
        from data.commons import upload_traces_to_hf  # push + optional registration
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Could not import data.commons.upload_traces_to_hf. Ensure project dependencies "
            f"are installed. Import error: {exc}"
        )

    # upload_traces_to_hf handles cleaning empty struct columns and push_to_hub
    upload_traces_to_hf(ds, args.repo_id, args.dataset_type)
    print(f"[trace-export] Upload complete: https://huggingface.co/datasets/{args.repo_id}")

    if args.skip_register:
        print("[trace-export] Note: --skip_register is set; Supabase registration may have been attempted by upload_traces_to_hf.")


if __name__ == "__main__":
    main()
