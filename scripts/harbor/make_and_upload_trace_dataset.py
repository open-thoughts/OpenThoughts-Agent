#!/usr/bin/env python3
"""
Export traces from a Harbor job directory into a Hugging Face dataset and upload it.

Single streaming path: enumerates trial directories non-recursively (via Harbor's
pruning ``iter_trial_dirs`` os.walk), processes trials in chunks, writes each chunk
as a parquet shard, uploads that shard to the HF dataset repo IMMEDIATELY via an
additive ``HfApi.upload_file`` commit, then DROPS the shard from disk and frees
the in-process Arrow/datasets buffers. Peak RAM is bounded to roughly one chunk.

Each shard is an additive hub commit, so a mid-run failure leaves prior shards
intact + resumable. Registration is a separate, ORTHOGONAL post-upload step
controlled by ``--skip_register``:

- No flag (default)  -> stream-upload, THEN register the finished HF repo in Supabase.
- ``--skip_register``  -> stream-upload only, no Supabase registration (RL path).

Examples:

  # Datagen: export, upload (streaming), AND register in Supabase
  python -m scripts.harbor.make_and_upload_trace_dataset \
    --job_dir /path/to/jobs/codecontests_glm46 \
    --repo_id DCAgent/code_contests-GLM-4.6-traces \
    --episodes last \
    --filter success

  # RL cleanup: upload only (streaming), do NOT register
  python -m scripts.harbor.make_and_upload_trace_dataset \
    --job_dir "$EXPERIMENTS_DIR/<job>/<job>" \
    --repo_id penfever/<job> \
    --episodes last \
    --skip_register

Notes:
- Requires Harbor to be installed/importable and a completed Harbor job dir.
- Auth to HF via HF_TOKEN env var or `huggingface-cli login`.
- Datasets are created PUBLIC by default. Pass --private for private repos.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from hpc.artifact_store import mounted, resolve_trials_root


# ---------------------------------------------------------------------------
# Harbor monkeypatches (preserved from the previous implementation). These make
# Harbor's per-trial collectors robust against malformed agent dirs, surrogate
# characters, and inline subagent merging. They are installed before any
# collection runs.
# ---------------------------------------------------------------------------


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
    """Patch Harbor rows before HF conversion to redact credentials and sanitize surrogates."""
    try:
        from harbor.utils import traces_utils  # type: ignore
        from scripts.harbor.run_and_export_traces import _strip_surrogates  # type: ignore
        from scripts.harbor.secret_redaction import redact_record
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
                redacted, findings = redact_record(row)
                if findings:
                    print(
                        f"[trace-export] Redacted {len(findings)} credential-shaped values."
                    )
                cleaned_rows.append(
                    {k: _strip_surrogates(v) for k, v in redacted.items()}
                )
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

    subagent_extractor = getattr(
        traces_utils, "_extract_complete_subagent_conversation", None
    )
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

        agent_steps = [
            idx for idx, step in enumerate(steps) if step.get("source") == "agent"
        ]

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
                    # Fix orphaned </think> tags: when the generation prompt includes <think>
                    # but vLLM only returns generated tokens, the opening tag is missing.
                    if "</think>" in message and "<think>" not in message:
                        message = "<think>" + message
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
                        content_parts.append(
                            f"<tool_call>\n{tool_call_json}\n</tool_call>"
                        )
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

            _append_subagent_transcripts(
                conv["conversations"], step, trajectory_dir, cache, run_metadata
            )

        return conv

    def patched_extract_conversations_from_trajectory(
        trajectory_file: Path,
        run_metadata: Dict[str, Any],
        embed_tools_in_conversation: bool = True,
        include_literal_tokens: bool = False,
        sft_format: bool = False,
    ) -> List[Dict[str, Any]]:
        # Keep the replacement signature aligned with Harbor's extractor. This
        # inline-subagent-merger emits the legacy conversations shape and does
        # not emit literal-token columns; the literal runtime skips this patch.
        _ = embed_tools_in_conversation, include_literal_tokens, sft_format
        try:
            trajectory_data = json.loads(trajectory_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[traces] Skipping trajectory {trajectory_file}: invalid JSON ({exc})"
            )
            return []

        steps = trajectory_data.get("steps", [])
        agent_info = trajectory_data.get("agent", {})
        trajectory_agent_name = agent_info.get("name") or run_metadata["agent_name"]
        trajectory_model_name = (
            agent_info.get("model_name") or run_metadata["model_name"]
        )
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
    traces_utils.extract_conversations_from_trajectory = (
        patched_extract_conversations_from_trajectory
    )


def _install_harbor_patches(include_literal_tokens: bool = False) -> None:
    """Install the Harbor monkeypatches used by the streaming exporter.

    The inline-subagent-merger patch reimplements ``extract_conversations_from_trajectory``
    and does NOT emit the literal token columns. When ``include_literal_tokens`` is
    requested we SKIP that patch and let Harbor's native (literal-aware) extraction
    run, which lifts ``step.metrics.{prompt_token_ids,completion_token_ids,logprobs}``
    into parallel columns.
    """
    _install_safe_episode_guard()
    _install_dataset_sanitizer()
    if not include_literal_tokens:
        _install_inline_subagent_merger()


# ---------------------------------------------------------------------------
# Streaming export helpers
# ---------------------------------------------------------------------------


def _import_traces_utils():
    """Resolve Harbor's traces_utils module (the per-trial collectors live here)."""
    try:
        from harbor.utils import traces_utils  # type: ignore

        return traces_utils
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Harbor is not available. Install it (pip install -e ../harbor) "
            f"or ensure it's on PYTHONPATH. Import error: {exc}"
        )


def _finalize_chunk(ds):
    """Apply the same sanitize/format cleanup the legacy path applied, per-chunk.

    Mirrors ``scripts.harbor.run_and_export_traces._finalize_trace_dataset`` and
    ``data.commons.clean_empty_structs`` but is applied to a single bounded
    chunk so peak memory stays small.
    """
    try:
        from scripts.harbor.run_and_export_traces import _finalize_trace_dataset  # type: ignore

        ds = _finalize_trace_dataset(ds)
    except Exception:
        pass
    return ds


def _iter_trial_dirs_nonrecursive(traces_utils, root: Path) -> Iterator[Path]:
    """Yield trial dirs using Harbor's PRUNING os.walk enumeration.

    ``iter_trial_dirs`` prunes ``dirnames`` once a trial dir is found, so it
    never descends into a trial's inner files — cost is O(directory skeleton).
    """
    yield from traces_utils.iter_trial_dirs(root, recursive=True)


def _collect_trial_rows(
    traces_utils,
    trial_dir: Path,
    *,
    episodes: str,
    success_filter: Optional[str],
    include_instruction: bool,
    include_verifier_output: bool,
    verbose: bool,
    include_literal_tokens: bool = False,
) -> List[Dict[str, Any]]:
    """Collect conversation rows for a single trial. Returns [] on skip/error."""
    from harbor.models.agent.name import AgentName  # type: ignore
    from harbor.agents.factory import AgentFactory  # type: ignore

    try:
        run_meta = traces_utils.load_run_metadata(trial_dir)
    except (ValueError, FileNotFoundError) as exc:
        if verbose:
            print(f"[traces] Skipping {trial_dir.name}: {exc}")
        return []

    agent_name = run_meta["agent_name"]
    agent_enum = AgentName(agent_name)
    agent_class = AgentFactory._AGENT_MAP.get(agent_enum)
    if agent_class is None or not agent_class.SUPPORTS_ATIF:
        raise NotImplementedError(
            f"{agent_name} does not support Harbor's trajectory format (ATIF), cannot export traces"
        )

    if success_filter in ("success", "failure"):
        succ = traces_utils._trial_is_success(trial_dir)
        if succ is None:
            if verbose:
                print(
                    f"[traces] Trial {trial_dir.name}: missing result.json; skipping due to filter"
                )
            return []
        if success_filter == "success" and not succ:
            return []
        if success_filter == "failure" and succ:
            return []

    try:
        return traces_utils.collect_conversations_from_trial(
            trial_dir,
            run_meta=run_meta,
            episodes=episodes,
            verbose=verbose,
            include_instruction=include_instruction,
            include_verifier_output=include_verifier_output,
            embed_tools_in_conversation=True,
            include_literal_tokens=include_literal_tokens,
        )
    except (UnicodeDecodeError, ValueError, OSError) as e:
        if verbose:
            print(
                f"[traces] Trial {trial_dir.name}: skipping due to decode/read error: {e}"
            )
        return []


def _release_arrow_memory() -> None:
    """Force-free Arrow + Python buffers so per-chunk RSS does not accumulate.

    HF ``datasets`` leaves memory-mapped Arrow tables and writer buffers alive
    that the pyarrow memory pool does NOT return to the OS on its own. Runs a
    GC pass and releases the default pyarrow memory pool's unused blocks.
    """
    gc.collect()
    try:
        import pyarrow as pa  # type: ignore

        pool = pa.default_memory_pool()
        # release_unused() hands freed-but-cached arena blocks back to the OS;
        # available on pyarrow's system/jemalloc/mimalloc pools.
        release = getattr(pool, "release_unused", None)
        if callable(release):
            release()
    except Exception:
        pass


_LITERAL_TOKEN_COLUMNS = ("prompt_token_ids", "completion_token_ids", "logprobs")


def _literal_token_features():
    """Explicit nested feature types for the literal token columns (datasets import deferred)."""
    from datasets import Sequence, Value  # type: ignore

    return {
        "prompt_token_ids": Sequence(Sequence(Value("int64"))),
        "completion_token_ids": Sequence(Sequence(Value("int64"))),
        "logprobs": Sequence(Sequence(Value("float64"))),
    }


def _pin_literal_token_columns(ds, rows: List[Dict[str, Any]]):
    """Rebuild the literal token columns from ``rows`` with an EXPLICIT nested type.

    ``Dataset.from_list`` and the finalize ``.map()`` passes infer feature types
    from each chunk's leading rows, so a chunk whose first rows carry no
    literals infers a null/empty type and silently DROPS the token lists of
    every other row in the chunk. The chunk pipeline never reorders or drops
    rows, so ``ds[i]`` corresponds to ``rows[i]``; we rebuild the three columns
    straight from ``rows`` with a pinned type as the LAST step before writing,
    bypassing inference entirely.
    """
    feats = _literal_token_features()

    def column(key):
        return [v if isinstance((v := row.get(key)), list) else [] for row in rows]

    for name in _LITERAL_TOKEN_COLUMNS:
        if name in ds.column_names:
            ds = ds.remove_columns([name])
        ds = ds.add_column(name, column(name))
    schema = ds.features.copy()
    schema.update(feats)
    return ds.cast(schema)


def _flush_chunk_to_parquet(
    rows: List[Dict[str, Any]],
    to_sharegpt: bool,
    traces_utils,
    shard_path: Path,
    cache_dir: Path,
    include_literal_tokens: bool = False,
) -> int:
    """Convert a chunk of rows to a (finalized) Dataset and write one parquet shard.

    Returns the number of rows written. The intermediate Dataset is dropped and
    the per-chunk HF datasets cache dir is wiped as soon as the parquet file is
    on disk, so peak RAM (and disk-cache) is bounded to one chunk. ``cache_dir``
    isolates this chunk's ``.map`` cache files so they can be deleted eagerly
    instead of piling up under the shared ``~/.cache/huggingface`` tree.

    When ``include_literal_tokens`` is set, the literal token columns are re-pinned from the
    source rows with an explicit nested type as the final step (see
    :func:`_pin_literal_token_columns`) so type inference on a null-leading chunk cannot drop
    them.
    """
    if not rows:
        return 0
    import datasets as _hf_datasets  # type: ignore

    # Pin every intermediate Dataset's on-disk artifacts to this chunk's
    # throwaway cache dir so they are removed with it (the global HF datasets
    # cache otherwise grows one Arrow file per chunk and never shrinks).
    prev_cache = _hf_datasets.config.HF_DATASETS_CACHE
    try:
        _hf_datasets.config.HF_DATASETS_CACHE = cache_dir
        ds = traces_utils.rows_to_dataset(rows)
        if to_sharegpt:
            ds = traces_utils.convert_openai_to_sharegpt(
                ds, "conversations", "conversations_sharegpt"
            )
        ds = _finalize_chunk(ds)
        if include_literal_tokens:
            ds = _pin_literal_token_columns(ds, rows)
        n = len(ds)
        ds.to_parquet(str(shard_path))
        del ds
    finally:
        _hf_datasets.config.HF_DATASETS_CACHE = prev_cache
        shutil.rmtree(cache_dir, ignore_errors=True)
        _release_arrow_memory()
    return n


def _upload_shard(api, repo_id: str, shard_path: Path) -> None:
    """Push ONE shard as its own additive commit, then remove it locally."""
    path_in_repo = f"data/{shard_path.name}"
    # Test seam: when set, copy the shard to a local dir instead of hitting the
    # Hub (lets the subprocess path be exercised offline without real creds).
    fake_dir = os.environ.get("TRACE_EXPORT_FAKE_UPLOAD_DIR")
    if fake_dir:
        dest = Path(fake_dir) / shard_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shard_path, dest)
    else:
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {path_in_repo}",
        )
    print(f"[trace-export] Uploaded {path_in_repo} to {repo_id}.")
    shard_path.unlink(missing_ok=True)


def _process_one_shard_in_subprocess(
    *,
    trial_dirs: List[str],
    shard_idx: int,
    repo_id: str,
    episodes: str,
    success_filter: Optional[str],
    to_sharegpt: bool,
    include_instruction: bool,
    include_verifier_output: bool,
    tmp_root: str,
    verbose: bool,
    include_literal_tokens: bool = False,
) -> int:
    """Worker body: collect this batch's rows, write + upload ONE shard, return row count.

    Runs in a FRESH process (multiprocessing ``spawn``). The collection path
    accumulates per-row native memory that ``gc.collect()`` +
    ``pyarrow.release_unused()`` do NOT reliably return to the OS; doing each
    shard in its own process makes the OS reclaim ALL of that on process exit,
    so parent RSS stays flat at roughly one chunk. The parent only ever holds a
    list of trial-dir path strings.
    """
    from huggingface_hub import HfApi  # type: ignore

    traces_utils = _import_traces_utils()
    _install_harbor_patches(include_literal_tokens=include_literal_tokens)

    chunk: List[Dict[str, Any]] = []
    for td in trial_dirs:
        rows = _collect_trial_rows(
            traces_utils,
            Path(td),
            episodes=episodes,
            success_filter=success_filter,
            include_instruction=include_instruction,
            include_verifier_output=include_verifier_output,
            verbose=verbose,
            include_literal_tokens=include_literal_tokens,
        )
        if rows:
            chunk.extend(rows)

    if not chunk:
        return 0

    shard_path = Path(tmp_root) / f"train-{shard_idx:05d}.parquet"
    cache_dir = Path(tmp_root) / f"_cache-{shard_idx:05d}"
    written = _flush_chunk_to_parquet(
        chunk,
        to_sharegpt,
        traces_utils,
        shard_path,
        cache_dir,
        include_literal_tokens=include_literal_tokens,
    )
    if not written:
        shard_path.unlink(missing_ok=True)
        return 0

    api = HfApi()
    _upload_shard(api, repo_id, shard_path)
    return written


def _run_shard_subprocess(kwargs: Dict[str, Any]) -> int:
    """Run one shard in a spawned subprocess and return its row count.

    A subprocess crash (including OOM-kill of the child) raises so the caller
    can decide; shards are additive commits, so a crash leaves prior shards intact.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        return pool.apply(_shard_worker_entry, (kwargs,))


def _shard_worker_entry(kwargs: Dict[str, Any]) -> int:
    """Top-level (picklable) entry the spawned worker runs."""
    return _process_one_shard_in_subprocess(**kwargs)


def stream_export_and_upload(
    *,
    job_dir: Path,
    repo_id: str,
    episodes: str,
    success_filter: Optional[str],
    to_sharegpt: bool,
    private: bool,
    include_instruction: bool = True,
    include_verifier_output: bool = True,
    chunk_size: int = 200,
    verbose: bool = False,
    include_literal_tokens: bool = False,
) -> int:
    """Single streaming path: enumerate trials, write parquet shards, upload via HfApi.

    Memory is bounded to roughly one ``chunk_size`` worth of TRIALS at any time.
    The parent only enumerates trial directories and batches their path strings;
    each batch's collect + parquet-write + upload happens in a FRESH spawned
    subprocess that exits afterward, so the OS reclaims the per-chunk
    native-memory leak. Each shard is an additive hub commit, so a mid-run
    failure leaves prior shards intact + resumable. Returns the total row count.
    """
    from huggingface_hub import HfApi  # type: ignore

    traces_utils = _import_traces_utils()

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"[trace-export] Repo {repo_id} ensured (private={private}).")

    # Working root: each shard is written here by its subprocess, uploaded, then
    # deleted. We never keep more than one shard's worth of parquet on disk.
    tmp_root = Path(tempfile.mkdtemp(prefix="trace_shards_"))

    total_rows = 0
    shard_idx = 0

    def _dispatch(batch: List[str]) -> None:
        nonlocal total_rows, shard_idx
        if not batch:
            return
        written = _run_shard_subprocess(
            dict(
                trial_dirs=batch,
                shard_idx=shard_idx,
                repo_id=repo_id,
                episodes=episodes,
                success_filter=success_filter,
                to_sharegpt=to_sharegpt,
                include_instruction=include_instruction,
                include_verifier_output=include_verifier_output,
                tmp_root=str(tmp_root),
                verbose=verbose,
                include_literal_tokens=include_literal_tokens,
            )
        )
        if written:
            total_rows += written
            shard_idx += 1
            print(
                f"[trace-export] Shard {shard_idx - 1:05d} done ({written} rows; running total {total_rows})."
            )

    try:
        n_trials = 0
        batch: List[str] = []
        for trial_dir in _iter_trial_dirs_nonrecursive(traces_utils, job_dir):
            n_trials += 1
            batch.append(str(trial_dir))
            # Batch by TRIAL count — the driver of per-shard memory. (rows/trial
            # varies, but a fixed trial-batch keeps each subprocess bounded.)
            if len(batch) >= chunk_size:
                _dispatch(batch)
                batch = []
        _dispatch(batch)  # final partial batch

        print(
            f"[trace-export] Enumerated {n_trials} trial dirs; collected {total_rows} rows into {shard_idx} shard(s)."
        )

        if shard_idx == 0:
            # Still write+push an empty dataset so downstream consumers see a
            # valid (zero-row) repo rather than nothing — mirrors prior behavior.
            empty_path = tmp_root / "train-00000.parquet"
            empty_ds = traces_utils.rows_to_dataset([])
            empty_ds.to_parquet(str(empty_path))
            del empty_ds
            _release_arrow_memory()
            _upload_shard(api, repo_id, empty_path)
            print("[trace-export] No rows collected; uploaded an empty parquet shard.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return total_rows


def register_trace_dataset_in_supabase(repo_id: str, dataset_type: str = "SFT") -> None:
    """Register an already-uploaded trace dataset repo in Supabase.

    Runs against the FINISHED HF repo, completely decoupled from the streaming
    upload mechanism. Invoked by ``main()`` only when ``--skip_register`` is
    NOT passed.
    """
    try:
        from database.unified_db.utils import register_hf_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Could not import register_hf_dataset for Supabase registration. "
            f"Import error: {exc}"
        )
    register_hf_dataset(repo_name=repo_id, dataset_type=dataset_type)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make and upload a trace dataset from a Harbor job directory (single streaming path)",
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
        "--dataset_type",
        default="SFT",
        help="Dataset type label ('SFT' or 'RL') used by the Supabase registration step. Default: SFT",
    )
    p.add_argument(
        "--skip_register",
        action="store_true",
        help="Upload only; do NOT register the dataset in Supabase. Controls ONLY "
        "registration — the streaming upload path is identical either way.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the HF dataset repo as private (default: public)",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Trials per parquet shard / per subprocess (bounds peak memory). Default: 200",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose per-trial logging",
    )
    p.add_argument(
        "--include_literal_tokens",
        action="store_true",
        help="REQUIRE literal token columns (prompt_token_ids / completion_token_ids / "
        "logprobs). Literals are auto-included whenever a literal.jsonl is "
        "discoverable; this flag FAILS LOUD when literals are expected but none are "
        "found. Never downgrades: a job with no literal.jsonl and no "
        "--include_literal_tokens exports text-only, byte-identical to before.",
    )
    p.add_argument(
        "--no_literal_tokens",
        action="store_true",
        help="Force TEXT-ONLY export even when a literal.jsonl is present (opt out of the "
        "default favor-literals-when-present behavior). Mutually exclusive intent with "
        "--include_literal_tokens.",
    )
    p.add_argument(
        "--literal_log",
        default=None,
        help="URI of the RecordProxy literal.jsonl (gs:// or local) to correlate into "
        "opencode trajectory step metrics before export. Only used with "
        "--include_literal_tokens. If omitted, auto-discovered under "
        "<job_dir>/logs/*_literal.jsonl (searching a few parents).",
    )
    p.add_argument(
        "--served_model",
        default=None,
        help="Authoritative tokenizer/model reference the engine served with (HF repo id "
        "e.g. 'Qwen/Qwen3.5-122B-A10B-FP8', or a gs:// mirror path). Stamped into the "
        "dataset (tokenizer_provenance.json + README) so the literal token columns are "
        "self-service decodable. Only used when literals are included; strongly "
        "recommended for any literal upload (a same-family tokenizer decodes to garbage).",
    )
    p.add_argument(
        "--single_commit",
        action="store_true",
        help="Stage all shards locally, then push them in ONE upload_folder commit instead of "
        "one commit per shard. Use for large datasets (>128 shards) to avoid HF's per-repo "
        "commit rate limit (128/hour). Trade-off: all shards accumulate on local disk before "
        "the single upload (peak disk ~ full dataset), vs one shard at a time for the default path.",
    )
    return p.parse_args()


PROVENANCE_FILENAME = "tokenizer_provenance.json"

# The literal token columns are only decodable with the EXACT tokenizer the
# serving engine used (a same-family tokenizer has a different merge table +
# vocab size, so word tokens decode to garbage). This template records the one
# that works.
_DECODE_RECIPE_TEMPLATE = """\
The `prompt_token_ids` / `completion_token_ids` / `logprobs` columns are the
verbatim tokens the serving engine emitted, stored PER AGENT STEP as a
list-of-lists (one inner list per turn). To turn them back into text you MUST
use the exact tokenizer the model was served with — a generic same-family
tokenizer will decode word tokens to garbage.

Served model / tokenizer source: `{served_model}`

```python
from transformers import AutoTokenizer
# Use the served model's own tokenizer (pull from the ref above; if it is a
# gs:// mirror, copy tokenizer.json/tokenizer_config.json/vocab.json/merges.txt
# locally first and point AutoTokenizer at that dir).
tok = AutoTokenizer.from_pretrained("{served_model}")
# token_ids are list-of-lists (one list per turn) — decode each turn:
text = [tok.decode(turn, skip_special_tokens=False) for turn in completion_token_ids]
```
"""


def read_served_model_name_from_literals(literal_logs: Sequence[str]) -> Optional[str]:
    """Best-effort: the ``literal.model`` the serving engine reported, from the first record.

    Reads only the FIRST non-empty line of the first literal log (streamed, so a
    multi-hundred-MB gs:// log is not slurped) and returns its ``literal.model``
    field. Secondary provenance alongside the authoritative ``served_model`` ref.
    Returns None if unreadable.
    """
    from upath import UPath  # lazy: fsspec/upath is only needed on the literal path

    for uri in literal_logs:
        try:
            with UPath(uri).open("r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    literal = entry.get("literal")
                    if isinstance(literal, dict) and literal.get("model"):
                        return str(literal["model"])
                    break
        except (FileNotFoundError, OSError, json.JSONDecodeError, NotImplementedError):
            continue
    return None


def build_tokenizer_provenance(
    *,
    served_model: Optional[str],
    served_model_name_observed: Optional[str],
) -> Dict[str, Any]:
    """Assemble the machine-readable tokenizer-provenance record for a literal dataset."""
    return {
        "schema": "tokenizer_provenance/v1",
        "served_model": served_model,
        "served_model_name_observed": served_model_name_observed,
        "literal_columns": ["prompt_token_ids", "completion_token_ids", "logprobs"],
        "literal_columns_shape": "list-of-lists (one inner list per agent step/turn)",
        "decode_with": "transformers.AutoTokenizer.from_pretrained(served_model)",
        "note": (
            "Decode with the EXACT served-model tokenizer; a same-family tokenizer "
            "decodes word tokens to garbage. Use skip_special_tokens=False to keep "
            "<|im_end|>/<tool_call>/<think> markers."
        ),
    }


def write_tokenizer_provenance(api, repo_id: str, provenance: Dict[str, Any]) -> None:
    """Upload the provenance JSON + a dataset-card README section to the HF repo.

    Idempotent: re-running overwrites both files with the same deterministic
    content. The README is a minimal auto-generated card (these trace repos carry
    no hand-curated card) whose whole purpose is to make the literal token IDs
    self-service decodable.
    """
    from huggingface_hub import HfApi  # type: ignore

    api = api or HfApi()
    served_model = (
        provenance.get("served_model")
        or provenance.get("served_model_name_observed")
        or "<unknown>"
    )

    prov_bytes = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    api.upload_file(
        path_or_fileobj=prov_bytes,
        path_in_repo=PROVENANCE_FILENAME,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add tokenizer provenance for literal token columns",
    )

    recipe = _DECODE_RECIPE_TEMPLATE.format(served_model=served_model)
    observed = provenance.get("served_model_name_observed")
    observed_line = (
        f"\nEngine-reported served model name: `{observed}`\n" if observed else ""
    )
    readme = (
        "---\n"
        "tags:\n"
        "- agent-traces\n"
        "- literal-tokens\n"
        "---\n\n"
        "# Agent trace dataset\n\n"
        "## Decoding the literal token IDs\n\n"
        f"{recipe}{observed_line}\n"
        f"See `{PROVENANCE_FILENAME}` for a machine-readable version.\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document literal-token decoding (tokenizer provenance)",
    )
    print(
        f"[trace-export] Stamped tokenizer provenance ({served_model}) into "
        f"{repo_id} ({PROVENANCE_FILENAME} + README)."
    )


def count_populated_literal_rows(table) -> int:
    """Number of rows whose ``prompt_token_ids`` column is non-empty (a pyarrow Table).

    Post-upload sanity check: a job that HAD a ``literal.jsonl`` must land >0
    such rows. Returns 0 when the column is absent entirely.
    """
    if "prompt_token_ids" not in table.column_names:
        return 0
    return sum(1 for v in table.column("prompt_token_ids").to_pylist() if v)


def resolve_literal_inclusion(
    job_dir: str,
    *,
    literal_log: Optional[str],
    include_literal_tokens: bool,
    no_literal_tokens: bool,
) -> tuple[bool, list[str]]:
    """Decide whether to include literal token columns, FAVORING them when present.

    Returns ``(include, literal_log_uris)`` — a LIST so a preempted job's multiple
    per-serve ``*_literal.jsonl`` files are ALL correlated (never just one):
      - ``--no_literal_tokens`` -> ``(False, [])`` (forced text-only).
      - else resolve ``[literal_log]`` (explicit) or auto-discover ALL sibling
        ``<job_dir>/logs/*_literal.jsonl``:
          - found -> ``(True, [<uris>])``.
          - not found and ``--include_literal_tokens`` (REQUIRE) -> SystemExit.
          - not found otherwise -> ``(False, [])`` (text-only, byte-identical to pre-literal).
    """
    from scripts.harbor.literal_correlator import discover_literal_logs

    if no_literal_tokens:
        return (False, [])
    resolved = [literal_log] if literal_log else discover_literal_logs(job_dir)
    if resolved:
        return (True, resolved)
    if include_literal_tokens:
        raise SystemExit(
            "--include_literal_tokens requires a literal.jsonl but none was found "
            f"under {job_dir}/logs/*_literal.jsonl (nor --literal_log). This job either "
            "was not run with --record_literal or its durable literal log is missing; "
            "refusing to silently export a literal-less dataset. Pass --no_literal_tokens "
            "to export text-only intentionally."
        )
    return (False, [])


def _run_export(args: argparse.Namespace) -> None:
    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise SystemExit(f"job_dir does not exist or is not a directory: {job_dir}")

    success_filter = None if args.filter == "none" else args.filter

    # Step 0 — literal-token correlation (opencode + --record_literal). The
    # RecordProxy captured token IDs / logprobs into a job-global literal.jsonl on
    # the worker; populate them into the per-trial trajectory step metrics here,
    # BEFORE export. Verify-or-skip: only steps whose token COUNTS match a
    # correlated record are enriched. No-op for non-literal jobs.
    include_literal_tokens, literal_logs = resolve_literal_inclusion(
        getattr(args, "literal_discovery_root", str(job_dir)),
        literal_log=args.literal_log,
        include_literal_tokens=bool(args.include_literal_tokens),
        no_literal_tokens=bool(args.no_literal_tokens),
    )
    if include_literal_tokens:
        from scripts.harbor.literal_correlator import enrich_trajectories_with_literals

        traces_utils = _import_traces_utils()
        print(
            f"[trace-export] Correlating literal tokens from {len(literal_logs)} "
            f"file(s): {literal_logs} ..."
        )
        stats = enrich_trajectories_with_literals(
            str(job_dir),
            literal_logs,
            iter_trial_dirs=lambda root: traces_utils.iter_trial_dirs(
                root, recursive=True
            ),
            verbose=bool(args.verbose),
        )
        yield_pct = (
            (100.0 * stats.trials_enriched / stats.trials) if stats.trials else 0.0
        )
        print(
            f"[trace-export] Literal yield: {stats.trials_enriched}/{stats.trials} trials "
            f"({yield_pct:.1f}%), {stats.steps_enriched} steps enriched, "
            f"{stats.trials_stripped} stripped; {stats.chains} chains "
            f"({stats.ambiguous_chains} ambiguous)."
        )
        # Fail loud: a literal-bearing job that bound ZERO trials is a regression —
        # never silently ship a text-only-looking dataset for a literal job.
        if stats.trials > 0 and stats.trials_enriched == 0:
            raise SystemExit(
                f"[trace-export] Literal correlation bound 0/{stats.trials} trials from "
                f"{literal_logs} ({stats.chains} chains) — refusing to export a "
                "literal-less dataset for a job that HAS a literal.jsonl. Check the "
                "literal log / job-dir pairing, or pass --no_literal_tokens to force "
                "text-only."
            )

    # Step 1 — the single, always-streaming, memory-safe upload mechanism.
    print(f"[trace-export] Streaming export from: {job_dir}")
    # --single_commit: stage every shard on local disk (reuse the per-shard subprocess's
    # TRACE_EXPORT_FAKE_UPLOAD_DIR seam so it writes-to-disk instead of committing to the
    # Hub), then push the whole folder in ONE upload_folder commit below. Sidesteps HF's
    # per-repo 128-commits/hour limit on >128-shard datasets.
    single_commit_dir = None
    if args.single_commit:
        single_commit_dir = tempfile.mkdtemp(prefix="trace_single_commit_")
        os.environ["TRACE_EXPORT_FAKE_UPLOAD_DIR"] = single_commit_dir
        print(
            f"[trace-export] --single_commit: staging shards under {single_commit_dir} (no per-shard commits)."
        )
    total = stream_export_and_upload(
        job_dir=job_dir,
        repo_id=args.repo_id,
        episodes=args.episodes,
        success_filter=success_filter,
        to_sharegpt=bool(args.to_sharegpt),
        private=bool(args.private),
        chunk_size=max(1, int(args.chunk_size)),
        verbose=bool(args.verbose),
        include_literal_tokens=include_literal_tokens,
    )
    if single_commit_dir is not None:
        from huggingface_hub import HfApi  # type: ignore

        os.environ.pop("TRACE_EXPORT_FAKE_UPLOAD_DIR", None)
        print(
            f"[trace-export] --single_commit: pushing {total} rows in ONE upload_folder commit..."
        )
        HfApi().upload_folder(
            folder_path=single_commit_dir,
            path_in_repo="data",
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
            commit_message=f"Upload {total} rows ({total // max(1, int(args.chunk_size))}+ shards) in one commit",
        )
        shutil.rmtree(single_commit_dir, ignore_errors=True)

    print(
        f"[trace-export] Upload complete ({total} rows): "
        f"https://huggingface.co/datasets/{args.repo_id}"
    )

    # Step 1b — stamp tokenizer/model provenance so the literal token columns are
    # self-service decodable. Warn loud if literals shipped without a --served_model
    # ref (the columns are still valid, just harder to decode later).
    if include_literal_tokens:
        from huggingface_hub import HfApi  # type: ignore

        served_model_name_observed = read_served_model_name_from_literals(literal_logs)
        if not args.served_model:
            print(
                "[trace-export] WARNING: literal token columns were uploaded but no "
                "--served_model was given. Stamping only the engine-reported name "
                f"({served_model_name_observed!r}); pass --served_model <hf-repo-or-gs-path> "
                "so consumers can pull the exact tokenizer."
            )
        provenance = build_tokenizer_provenance(
            served_model=args.served_model,
            served_model_name_observed=served_model_name_observed,
        )
        write_tokenizer_provenance(HfApi(), args.repo_id, provenance)

    # Step 2 — ORTHOGONAL post-upload Supabase registration, gated by --skip_register.
    if args.skip_register:
        print(
            "[trace-export] --skip_register set: upload only, NOT registering "
            "in Supabase."
        )
    else:
        print(
            f"[trace-export] Registering {args.repo_id} in Supabase "
            f"(dataset_type={args.dataset_type})..."
        )
        register_trace_dataset_in_supabase(args.repo_id, dataset_type=args.dataset_type)
        print(f"[trace-export] Supabase registration complete for {args.repo_id}.")


def main() -> None:
    """Export a legacy tree directly or mount an inactive image for the read."""
    args = _parse_args()
    requested_root = Path(args.job_dir).expanduser().resolve()
    if not requested_root.is_dir():
        raise SystemExit(
            f"job_dir does not exist or is not a directory: {requested_root}"
        )

    kind, source = resolve_trials_root(requested_root)
    if kind == "bare":
        _run_export(args)
        return

    args.literal_discovery_root = str(requested_root)
    with mounted(source, mode="ro") as image_root:
        args.job_dir = str(image_root)
        _run_export(args)


if __name__ == "__main__":
    main()
