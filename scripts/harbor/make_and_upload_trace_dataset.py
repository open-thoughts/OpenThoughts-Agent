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
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _install_dataset_sanitizer() -> None:
    """Patch Harbor's rows_to_dataset to sanitize surrogate characters before HF conversion."""
    try:
        from harbor.utils import traces_utils  # type: ignore
        from scripts.harbor.run_and_export_traces import _strip_surrogates  # type: ignore
    except Exception:
        return

    original_rows_to_dataset = getattr(traces_utils, "rows_to_dataset", None)
    if original_rows_to_dataset is None:
        return
    if getattr(original_rows_to_dataset, "__dcagent_surrogate_sanitized__", False):
        return

    def safe_rows_to_dataset(rows, *args, **kwargs):
        cleaned_rows = []
        for row in rows:
            if isinstance(row, dict):
                cleaned_rows.append({k: _strip_surrogates(v) for k, v in row.items()})
            else:
                cleaned_rows.append(row)
        return original_rows_to_dataset(cleaned_rows, *args, **kwargs)

    safe_rows_to_dataset.__dcagent_surrogate_sanitized__ = True  # type: ignore[attr-defined]
    traces_utils.rows_to_dataset = safe_rows_to_dataset


def _install_inline_subagent_merger() -> None:
    """
    Patch Harbor's conversation extraction so subagent trajectories are injected into
    the main agent conversations instead of being exported as standalone rows.
    """

    try:
        from harbor.utils import traces_utils  # type: ignore
    except Exception:
        return

    if getattr(traces_utils, "__dcagent_inline_subagents__", False):
        return

    subagent_extractor = getattr(traces_utils, "_extract_complete_subagent_conversation", None)
    if subagent_extractor is None:
        return

    def _infer_subagent_label(path_fragment: str) -> str:
        name = Path(path_fragment).name
        if name.startswith("trajectory.") and name.endswith(".json"):
            return name[len("trajectory.") : -len(".json")]
        return name

    def _append_conversation_turn(
        messages: List[Dict[str, str]], role: str, content: str
    ) -> None:
        """Append a turn, merging with the previous one if the role matches."""
        if not content:
            return
        role_key = "assistant" if role == "assistant" else "user"
        if messages and messages[-1]["role"] == role_key:
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
        else:
            messages.append({"role": role_key, "content": content})

    def _format_subagent_conversation(
        conv_dict: Dict[str, Any], label: str, summary: Optional[str]
    ) -> List[Dict[str, str]]:
        conversations = conv_dict.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return []

        prefix = f"[subagent:{label}]"
        header_parts = [prefix]
        if summary:
            header_parts.append(summary)
        formatted: List[Dict[str, str]] = []
        formatted.append(
            {
                "role": "user",
                "content": " ".join(part for part in header_parts if part).strip(),
            }
        )

        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = "assistant" if message.get("role") == "assistant" else "user"
            content = message.get("content") or ""
            if not content:
                continue
            tagged_content = f"{prefix} {role.upper()}: {content}"
            formatted.append({"role": role, "content": tagged_content})

        return [msg for msg in formatted if msg.get("content")]

    def _observation_to_text(observation: Any) -> Optional[str]:
        if not isinstance(observation, dict):
            return None
        results = observation.get("results")
        if not isinstance(results, list):
            return None
        observation_contents: List[str] = []
        for result in results:
            if isinstance(result, dict) and "content" in result:
                observation_contents.append(result["content"])
        if observation_contents:
            return "\n".join(observation_contents)
        return None

    def _append_subagent_transcripts(
        messages: List[Dict[str, str]],
        step: Dict[str, Any],
        trajectory_dir: Path,
        cache: Dict[str, Optional[Dict[str, Any]]],
        run_metadata: Dict[str, Any],
    ) -> None:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            return
        results = observation.get("results")
        if not isinstance(results, list):
            return

        for result in results:
            if not isinstance(result, dict):
                continue
            refs = result.get("subagent_trajectory_ref")
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                rel_path = ref.get("trajectory_path")
                if not rel_path:
                    continue
                subagent_path = (trajectory_dir / rel_path).resolve()
                cache_key = str(subagent_path)
                subagent_conv = cache.get(cache_key)
                if subagent_conv is None:
                    if not subagent_path.exists():
                        cache[cache_key] = None
                        continue
                    try:
                        subagent_conv = subagent_extractor(subagent_path, run_metadata)  # type: ignore[misc]
                    except Exception:
                        subagent_conv = None
                    cache[cache_key] = subagent_conv

                if not subagent_conv:
                    continue

                label = _infer_subagent_label(rel_path)
                extra = ref.get("extra")
                summary = extra.get("summary") if isinstance(extra, dict) else None
                formatted_messages = _format_subagent_conversation(
                    subagent_conv, label, summary
                )
                for formatted in formatted_messages:
                    _append_conversation_turn(
                        messages, formatted["role"], formatted["content"]
                    )

    def _extract_episode_with_subagents(
        steps: List[Dict[str, Any]],
        episode_num: int,
        run_metadata: Dict[str, Any],
        trajectory_dir: Path,
        cache: Dict[str, Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        conv: Dict[str, Any] = {
            "conversations": [],
            "agent": run_metadata["agent_name"],
            "model": run_metadata["model_name"],
            "model_provider": run_metadata["model_provider"],
            "date": run_metadata["start_time"],
            "task": None,
            "episode": f"episode-{episode_num}",
            "run_id": run_metadata["run_id"],
            "trial_name": None,
        }

        agent_steps = [idx for idx, step in enumerate(steps) if step.get("source") == "agent"]

        for idx, step in enumerate(steps):
            source = step.get("source")
            message = step.get("message", "")

            if source in {"system", "user"}:
                _append_conversation_turn(conv["conversations"], "user", message)
            elif source == "agent":
                content_parts: List[str] = []
                reasoning_content = step.get("reasoning_content")
                if reasoning_content:
                    content_parts.append(f"<think>{reasoning_content}</think>")
                if message:
                    content_parts.append(message)
                tool_calls = step.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        tool_call_obj = {
                            "name": call.get("function_name"),
                            "arguments": call.get("arguments", {}),
                        }
                        tool_call_json = json.dumps(tool_call_obj, ensure_ascii=False)
                        content_parts.append(f"<tool_call>\n{tool_call_json}\n</tool_call>")
                assistant_content = "\n".join(content_parts) if content_parts else ""
                _append_conversation_turn(
                    conv["conversations"], "assistant", assistant_content
                )

                is_last_agent_step = agent_steps and (idx == agent_steps[-1])
                if not is_last_agent_step:
                    observation_text = _observation_to_text(step.get("observation"))
                    if observation_text:
                        _append_conversation_turn(
                            conv["conversations"], "user", observation_text
                        )

            _append_subagent_transcripts(conv["conversations"], step, trajectory_dir, cache, run_metadata)

        return conv

    def patched_extract_conversations_from_trajectory(
        trajectory_file: Path, run_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        try:
            trajectory_data = json.loads(trajectory_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[traces] Skipping trajectory {trajectory_file}: invalid JSON ({exc})")
            return []

        steps = trajectory_data.get("steps", [])
        agent_info = trajectory_data.get("agent", {})
        trajectory_agent_name = agent_info.get("name") or run_metadata["agent_name"]
        trajectory_model_name = agent_info.get("model_name") or run_metadata["model_name"]
        trajectory_run_metadata = {
            **run_metadata,
            "agent_name": trajectory_agent_name,
            "model_name": trajectory_model_name,
        }

        agent_step_indices: List[int] = []
        for idx, step in enumerate(steps):
            if step.get("source") == "agent" and not step.get("is_copied_context"):
                agent_step_indices.append(idx)

        if not agent_step_indices:
            return []

        trajectory_dir = trajectory_file.parent
        subagent_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        conversations: List[Dict[str, Any]] = []
        for episode_num, agent_step_idx in enumerate(agent_step_indices):
            conv = _extract_episode_with_subagents(
                steps[: agent_step_idx + 1],
                episode_num,
                trajectory_run_metadata,
                trajectory_dir,
                subagent_cache,
            )
            if conv and conv.get("conversations"):
                conversations.append(conv)
        return conversations

    patched_extract_conversations_from_trajectory.__dcagent_inline_subagents__ = True  # type: ignore[attr-defined]
    traces_utils.extract_conversations_from_trajectory = patched_extract_conversations_from_trajectory


def _import_export_traces():
    """Resolve the export_traces helper, preferring the OpenThoughts-Agent wrapper."""
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
    _install_dataset_sanitizer()
    _install_inline_subagent_merger()

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
        export_subagents=False,
        include_reasoning=bool(args.include_reasoning),
    )
    try:
        from scripts.harbor.run_and_export_traces import _finalize_trace_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"Failed to import Harbor trace finalizer: {exc}")

    ds = _finalize_trace_dataset(ds)

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
