#!/usr/bin/env python3
"""
Convert OpenAI Codex rollout logs into ot-agent style trace datasets.

Each rollout JSONL file (as produced by codex-sessions) is reconstructed into
a Harbor-compatible trace row so it can be mixed with other ot-agent training
datasets. Rows include the standard trace fields plus Codex-specific metadata.

Example:
    python data/codex_chats/generate.py \
        --sessions-root ~/Documents/codex-sessions \
        --save-dir ~/Documents/datasets/codex_chats_traces \
        --repo-id DCAgent/codex-chats-traces
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset

try:
    from data.commons import upload_traces_to_hf
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    upload_traces_to_hf = None  # type: ignore[assignment]
    _UPLOAD_IMPORT_ERROR = exc
else:
    _UPLOAD_IMPORT_ERROR = None


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _format_blocks(blocks: Any) -> str:
    """Normalize Codex message content into text."""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        parts: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                parts.append(json.dumps(block, ensure_ascii=False))
                continue
            btype = block.get("type")
            text = block.get("text")
            if btype in {"input_text", "output_text", "text", "reasoning"} and isinstance(text, str):
                parts.append(text)
                continue
            if btype == "code":
                language = block.get("language") or ""
                code = block.get("text") or ""
                fence = f"```{language}\n{code}\n```" if code else ""
                if fence:
                    parts.append(fence)
                continue
            if btype == "image_url":
                parts.append(f"[image_url] {block.get('image_url')}")
                continue
            parts.append(json.dumps(block, ensure_ascii=False))
        return "\n\n".join(p for p in parts if p).strip()
    try:
        return json.dumps(blocks, ensure_ascii=False)
    except TypeError:
        return str(blocks)


def _format_reasoning(payload: Dict[str, Any]) -> str:
    """Convert a reasoning response item into a textual <think> section."""
    content = _format_blocks(payload.get("content"))
    if not content:
        summary_segments = payload.get("summary") or []
        texts = [
            seg.get("text")
            for seg in summary_segments
            if isinstance(seg, dict) and isinstance(seg.get("text"), str)
        ]
        content = "\n".join(t for t in texts if t)
    if not content:
        content = "[encrypted reasoning unavailable]"
    return f"<think>\n{content}\n</think>"


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _format_function_call(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    name = payload.get("name") or "unknown_tool"
    call_id = payload.get("call_id")
    args_raw = payload.get("arguments")
    args = _safe_json_loads(args_raw)
    if isinstance(args, (dict, list)):
        args_text = json.dumps(args, ensure_ascii=False, indent=2)
        serialized_args = json.dumps(args, ensure_ascii=False)
    else:
        args_text = str(args)
        serialized_args = args_text
    text = f"[tool-call] {name}\n{args_text}"
    return text, {
        "call_id": call_id,
        "name": name,
        "arguments": serialized_args,
        "outputs": [],
    }


def _format_function_output(payload: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    call_id = payload.get("call_id")
    output_raw = payload.get("output")
    output_payload = _safe_json_loads(output_raw)
    content = ""
    metadata = None
    if isinstance(output_payload, dict):
        content = output_payload.get("output") or ""
        metadata = output_payload.get("metadata")
        if not content:
            content = json.dumps(output_payload, ensure_ascii=False)
    else:
        content = str(output_payload or "")
    text = content if content else "[tool-output] (empty)"
    serialized_meta = json.dumps(metadata, ensure_ascii=False) if isinstance(metadata, (dict, list)) else metadata
    return text, call_id, {"call_id": call_id, "output": content, "metadata": serialized_meta}


def _build_system_message(meta: Dict[str, Any]) -> Optional[str]:
    if not meta:
        return None
    origin = meta.get("originator") or meta.get("source")
    cli_version = meta.get("cli_version")
    cwd = meta.get("cwd")
    parts = ["Session ingested from Codex rollouts."]
    if origin:
        parts.append(f"Originator: {origin}")
    if cli_version:
        parts.append(f"Codex CLI version: {cli_version}")
    if cwd:
        parts.append(f"User workspace: {cwd}")
    instructions = meta.get("instructions")
    if instructions:
        parts.append(f"Pinned instructions: {instructions}")
    return "\n".join(parts)


def parse_rollout(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    conversations: List[Dict[str, str]] = []
    session_meta: Dict[str, Any] = {}
    last_turn_context: Dict[str, Any] = {}
    tool_calls: Dict[str, Dict[str, Any]] = {}
    collected_tool_calls: List[Dict[str, Any]] = []
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            first_ts = first_ts or entry.get("timestamp")
            last_ts = entry.get("timestamp") or last_ts
            etype = entry.get("type")
            payload = entry.get("payload")
            if payload is None:
                payload = entry

            if etype == "session_meta" or (
                etype is None and isinstance(payload, dict) and payload.get("id")
            ):
                if isinstance(payload, dict):
                    session_meta = payload
                continue

            if etype == "turn_context":
                if isinstance(payload, dict):
                    last_turn_context = payload
                continue

            if etype == "response_item":
                payload_type = payload.get("type")

                if payload_type == "message":
                    role = payload.get("role") or "assistant"
                    content = _format_blocks(payload.get("content"))
                    if content:
                        conversations.append({"role": role, "content": content})
                    continue

                if payload_type == "reasoning":
                    conversations.append({"role": "assistant", "content": _format_reasoning(payload)})
                    continue

                if payload_type == "function_call":
                    text, call = _format_function_call(payload)
                    conversations.append({"role": "assistant", "content": text})
                    call_id = call.get("call_id")
                    if call_id:
                        tool_calls[call_id] = call
                        collected_tool_calls.append(call)
                    continue

                if payload_type == "function_call_output":
                    text, call_id, output_meta = _format_function_output(payload)
                    conversations.append({"role": "tool", "content": text})
                    if call_id and call_id in tool_calls:
                        tool_calls[call_id]["outputs"].append(output_meta)
                    continue

                role = payload.get("role") or "assistant"
                conversations.append(
                    {
                        "role": role,
                        "content": f"[{payload_type}] {json.dumps(payload, ensure_ascii=False)}",
                    }
                )
                continue

            if etype in {"message", "reasoning", "function_call", "function_call_output"}:
                payload_type = etype
                if payload_type == "message":
                    role = payload.get("role") or "assistant"
                    content = _format_blocks(payload.get("content"))
                    if content:
                        conversations.append({"role": role, "content": content})
                    continue

                if payload_type == "reasoning":
                    conversations.append({"role": "assistant", "content": _format_reasoning(payload)})
                    continue

                if payload_type == "function_call":
                    text, call = _format_function_call(payload)
                    conversations.append({"role": "assistant", "content": text})
                    call_id = call.get("call_id")
                    if call_id:
                        tool_calls[call_id] = call
                        collected_tool_calls.append(call)
                    continue

                if payload_type == "function_call_output":
                    text, call_id, output_meta = _format_function_output(payload)
                    conversations.append({"role": "tool", "content": text})
                    if call_id and call_id in tool_calls:
                        tool_calls[call_id]["outputs"].append(output_meta)
                    continue

            if entry.get("record_type"):
                continue

            if etype is None:
                continue

            role = payload.get("role") or "assistant"
            conversations.append(
                {"role": role, "content": f"[{etype}] {json.dumps(payload, ensure_ascii=False)}"}
            )

    if not conversations:
        return None

    system_message = _build_system_message(session_meta)
    if system_message:
        if not conversations or conversations[0]["role"] != "system":
            conversations.insert(0, {"role": "system", "content": system_message})

    session_id = session_meta.get("id") or path.stem
    rel_path = str(path.relative_to(root))
    date_str = session_meta.get("timestamp") or first_ts or last_ts

    row: Dict[str, Any] = {
        "conversations": conversations,
        "agent": session_meta.get("originator") or "codex",
        "model": last_turn_context.get("model") or session_meta.get("model") or "unknown",
        "model_provider": session_meta.get("model_provider") or "openai",
        "date": date_str,
        "task": f"codex_session::{session_id}",
        "episode": "episode-0",
        "run_id": session_id,
        "trial_name": rel_path,
        "session_path": rel_path,
        "session_meta": session_meta,
        "turn_context": last_turn_context,
        "tool_calls": collected_tool_calls,
        "first_event_ts": first_ts,
        "last_event_ts": last_ts,
        "num_messages": len(conversations),
    }

    start_dt = _parse_iso(first_ts)
    end_dt = _parse_iso(last_ts)
    if start_dt and end_dt and end_dt >= start_dt:
        row["duration_seconds"] = (end_dt - start_dt).total_seconds()

    return row


def collect_rollouts(root: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    files = sorted(root.rglob("rollout-*.jsonl"))
    if limit is not None:
        files = files[:limit]

    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(files, start=1):
        record = parse_rollout(path, root)
        if record:
            record["episode"] = f"episode-{idx:05d}"
            rows.append(record)
    return rows


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert codex-sessions rollouts into trace datasets."
    )
    parser.add_argument(
        "--sessions-root",
        default=str(Path.home() / "Documents" / "codex-sessions"),
        help="Directory containing Codex rollout JSONL files (default: %(default)s).",
    )
    parser.add_argument(
        "--save-dir",
        help="Optional directory to save the dataset via Dataset.save_to_disk().",
    )
    parser.add_argument(
        "--repo-id",
        help="Optional Hugging Face dataset repo (org/name) for upload_traces_to_hf.",
    )
    parser.add_argument(
        "--dataset-type",
        default="SFT",
        help="Dataset type used when registering with upload_traces_to_hf (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N rollout files (useful for smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    sessions_root = Path(args.sessions_root).expanduser().resolve()
    if not sessions_root.exists():
        raise SystemExit(f"sessions root does not exist: {sessions_root}")

    rows = collect_rollouts(sessions_root, args.limit)
    if not rows:
        raise SystemExit("No conversations were extracted from the provided sessions root.")

    dataset = Dataset.from_list(rows)
    print(f"[codex-chats] Built dataset with {len(dataset)} rows from {sessions_root}")

    if args.save_dir:
        save_path = Path(args.save_dir).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"[codex-chats] Saved dataset to {save_path}")

    if args.repo_id:
        if upload_traces_to_hf is None:
            raise SystemExit(
                "upload_traces_to_hf is unavailable because optional dependencies are missing. "
                "Install ot-agent with its data extras or ensure google-cloud-storage is installed. "
                f"Original import error: {_UPLOAD_IMPORT_ERROR}"
            )
        print(f"[codex-chats] Uploading dataset to Hugging Face: {args.repo_id}")
        upload_traces_to_hf(dataset, args.repo_id, args.dataset_type)


if __name__ == "__main__":
    main()
