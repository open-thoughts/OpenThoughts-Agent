#!/usr/bin/env python3
"""
Fetch and concatenate every dataset used to train any model in Supabase.

Adds two new columns to every row:
  original_source   — task source extracted from the dataset repo name
  original_teacher  — teacher model extracted from the dataset repo name

Memory-efficient: each source dataset is written to a temp Parquet shard
immediately after loading and freed from RAM. The shards are then merged
via PyArrow streaming (one batch at a time) and uploaded directly to HF Hub
without ever holding the full corpus in memory.

Example:
  python -m scripts.datagen.concatenate_training_datasets \\
    --target mlfoundations-dev/all-training-data \\
    --private \\
    --skip-on-error

  # Inspect what would be fetched (no download):
  python -m scripts.datagen.concatenate_training_datasets \\
    --target x/y --dry-run --unknown-only

Requires: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, HF_TOKEN env vars.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pad
import pyarrow.parquet as pq
from datasets import DatasetDict
from huggingface_hub import HfApi, create_repo


# ── repos to exclude from concatenation ──────────────────────────────────────

_DROP_REPOS: set[str] = {
    "DCAgent/tbench_oracle_solutions_terminus",
    "DCAgent2/taskmaster2-GLM-4.6-32ep-32k",
}

_DROP_PATTERNS: list[re.Pattern] = [
    re.compile(r"dev_set_part1_10k"),
    re.compile(r"nemotron[-_]terminal[-_]debugging"),
    re.compile(r"nemotron[-_]terminal[-_]scientific_computing"),
    re.compile(r"staqc[-_]ot3"),
    re.compile(r"open[-_]thoughts[-_]4"),
]


def _should_drop(repo_id: str) -> bool:
    if repo_id in _DROP_REPOS:
        return True
    name = repo_id.split("/")[-1]
    return any(p.search(name) for p in _DROP_PATTERNS)


# ── teacher / source extraction ───────────────────────────────────────────────

_TEACHER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"nemotron[-_]terminal[-_]scientific", re.I), "GLM-4.7"),
    (re.compile(r"glm[-_]?5|glm5", re.I),                    "GLM-5.0"),
    (re.compile(r"glm[-_]?4\.?7|glm47", re.I),               "GLM-4.7"),
    (re.compile(r"glm[-_]?4\.?6|glm46", re.I),               "GLM-4.6"),
    (re.compile(r"terminus[-_]?2", re.I),                     "GPT 5.1 Nano"),
    (re.compile(r"gpt[-_]?5[-_]?nano", re.I),                "GPT-5-nano"),
    (re.compile(r"gpt[-_]?5[-_]?mini", re.I),                "GPT-5-mini"),
    (re.compile(r"gpt[-_]?5", re.I),                         "GPT-5"),
    (re.compile(r"gpt[-_]?o3", re.I),                        "GPT-o3"),
    (re.compile(r"gpt[-_]?4o", re.I),                        "GPT-4o"),
    (re.compile(r"gpt[-_]?4", re.I),                         "GPT-4"),
    (re.compile(r"kimi[-_]?k2", re.I),                        "Kimi K2.0 Thinking"),
    (re.compile(r"minimax[-_]?m2", re.I),                     "MiniMax M2.0"),
    (re.compile(r"kimi[-_]?2\.5|kimi[-_]?2_5", re.I),        "Kimi-2.5"),
    (re.compile(r"qwen3[-_]?72b", re.I),                     "Qwen3-72B"),
    (re.compile(r"qwen3[-_]?32b", re.I),                     "Qwen3-32B"),
    (re.compile(r"qwen3[-_]?14b", re.I),                     "Qwen3-14B"),
    (re.compile(r"qwen3[-_]?8b", re.I),                      "Qwen3-8B"),
    (re.compile(r"qwen3[-_]?4b", re.I),                      "Qwen3-4B"),
    (re.compile(r"qwen3[-_]?1\.7b|qwen3[-_]?1_7b", re.I),   "Qwen3-1.7B"),
    (re.compile(r"qwen3[-_]?0\.6b|qwen3[-_]?0_6b", re.I),   "Qwen3-0.6B"),
    (re.compile(r"qwen3", re.I),                             "Qwen3"),
    (re.compile(r"gpt[-_]?oss[-_]?120b", re.I),             "GPT-OSS-120B"),
    (re.compile(r"gemini[-_]?2\.?5[-_]?flash", re.I),       "Gemini-2.5-Flash"),
    (re.compile(r"gemini[-_]?2\.?5[-_]?pro", re.I),         "Gemini-2.5-Pro"),
    (re.compile(r"claude[-_]?3\.?7[-_]?sonnet", re.I),      "Claude-3.7-Sonnet"),
    (re.compile(r"claude[-_]?3\.?5[-_]?sonnet", re.I),      "Claude-3.5-Sonnet"),
    (re.compile(r"deepseek[-_]?r1", re.I),                   "DeepSeek-R1"),
    (re.compile(r"\bo1[-_]mini\b", re.I),                    "o1-mini"),
]

_SOURCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"swesmith", re.I),                        "swesmith"),
    (re.compile(r"swe[-_]?bench", re.I),                   "swebench"),
    (re.compile(r"r2egym", re.I),                          "r2egym"),
    (re.compile(r"codeforces", re.I),                      "codeforces"),
    (re.compile(r"nl2bash", re.I),                         "nl2bash"),
    (re.compile(r"code[-_]?contests", re.I),               "code_contests"),
    (re.compile(r"codenet", re.I),                         "codenet"),
    (re.compile(r"taskmaster", re.I),                      "taskmaster"),
    (re.compile(r"staqc", re.I),                           "staqc"),
    (re.compile(r"puzzles", re.I),                         "puzzles"),
    (re.compile(r"stack[-_]?exchange", re.I),              "stack_exchange"),
    (re.compile(r"stackoverflow", re.I),                   "stackoverflow"),
    (re.compile(r"stack[-_]?ruby", re.I),                  "stack_ruby"),
    (re.compile(r"stack[-_]?tezos", re.I),                 "stack_tezos"),
    (re.compile(r"terminal[-_]bench", re.I),               "terminal_bench"),
    (re.compile(r"gsm8k", re.I),                           "gsm8k"),
    (re.compile(r"\bmath\b", re.I),                        "math"),
    (re.compile(r"exp[-_]rpt", re.I),                      "exp_rpt"),
    (re.compile(r"exp[-_]tas", re.I),                      "exp_tas"),
    (re.compile(r"exp[-_]rle", re.I),                      "exp_rle"),
    (re.compile(r"exp[-_]san", re.I),                      "exp_san"),
    (re.compile(r"codeelo", re.I),                         "codeelo"),
    (re.compile(r"freelancer[-_]", re.I),                  "freelancer"),
    (re.compile(r"nemo[-_]?prism", re.I),                  "Nemotron Prism"),
    (re.compile(r"swegym", re.I),                          "SWEGym"),
    (re.compile(r"neulab[-_]", re.I),                      "neulab"),
    (re.compile(r"inferredbugs", re.I),                    "Inferred Bugs"),
    (re.compile(r"qasper", re.I),                          "Qasper"),
    (re.compile(r"wikitable_format_conversion", re.I),     "WikiTable Format Conversion"),
    (re.compile(r"bash[-_]textbook", re.I),                "bash textbook"),
    (re.compile(r"[-_]stack[-_]overflow[-_]", re.I),       "StackExchange Overflow"),
    (re.compile(r"multifile[-_]composition", re.I),        "multifile composition"),
    (re.compile(r"repo[-_]scaffold", re.I),                "repo scaffold"),
    (re.compile(r"defects4j", re.I),                       "defects 4j"),
    (re.compile(r"magicoder[-_]evol[-_]instruct", re.I),   "MagiCoder Evol Instruct"),
    (re.compile(r"glaive[-_]code[-_]assistant", re.I),     "Glaive Code Assistant"),
    (re.compile(r"selfinstruct[-_]naive", re.I),           "Self-Instruct Naive"),
    (re.compile(r"python[-_]scripts", re.I),               "python_scripts"),
    (re.compile(r"ling[-_]coder", re.I),                   "ling_coder"),
    (re.compile(r"open[-_]thoughts", re.I),                "open_thoughts"),
]


def _extract_teacher(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    if name.startswith("exp_"):
        return "GLM-4.7"
    for pat, label in _TEACHER_PATTERNS:
        if pat.search(name):
            return label
    return "GPT 5.1 Nano"


def _extract_source(repo_id: str) -> str:
    name = repo_id.split("/")[-1]
    for pat, label in _SOURCE_PATTERNS:
        if pat.search(name):
            return label
    return "unknown"


# ── supabase ──────────────────────────────────────────────────────────────────

def _get_client():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", file=sys.stderr)
        sys.exit(1)
    return create_client(url, key)


def _fetch_all_dataset_refs() -> list[str]:
    """Return deduplicated list of HF dataset repo IDs used across all models."""
    client = _get_client()
    rows: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        batch = (
            client.table("models")
            .select("dataset_names")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        if not batch.data:
            break
        rows.extend(batch.data)
        if len(batch.data) < page_size:
            break
        offset += page_size

    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        raw = (row.get("dataset_names") or "").strip()
        if not raw:
            continue
        for ref in raw.split(","):
            ref = ref.strip().strip("'\"[]")
            if not ref or ref.startswith("/") or "/" not in ref:
                continue
            if ref not in seen:
                seen.add(ref)
                result.append(ref)
    return result


# ── loading and per-shard writing ─────────────────────────────────────────────

def _load_and_write_shard(repo_id: str, shard_path: Path) -> int:
    """Load one HF dataset, add meta columns, write to Parquet, return row count.

    The Dataset object is freed as soon as the Parquet file is written so that
    only one source dataset is in memory at a time.
    """
    from datasets import load_dataset

    dsd = load_dataset(repo_id)
    if isinstance(dsd, DatasetDict):
        split = "train" if "train" in dsd else next(iter(dsd))
        ds = dsd[split]
    else:
        ds = dsd  # type: ignore[assignment]

    n = len(ds)
    ds = ds.add_column("original_source", [_extract_source(repo_id)] * n)
    ds = ds.add_column("original_teacher", [_extract_teacher(repo_id)] * n)

    # Write shard to disk and release the in-memory Dataset
    ds.to_parquet(str(shard_path))
    del ds
    return n


# ── streaming merge + upload (combined, no intermediate file) ─────────────────

_BATCH_ROWS = 2_000           # rows per read batch — keeps per-batch RAM ~20 MB
_MAX_SHARD_BYTES = 500 * 1024 * 1024  # 500 MB per upload shard


def _merge_and_upload(
    shard_paths: list[Path],
    target: str,
    private: bool,
    token: str | None,
    commit_message: str,
    tmp_root: Path,
) -> None:
    """Stream all Pass-1 shards directly to upload-sized Parquet files and push
    each to HF Hub immediately — no large intermediate file is ever created.

    Peak RAM  : O(one batch) ≈ 2 000 rows × ~12 KB ≈ 24 MB.
    Peak disk : Pass-1 shards + one upload shard at a time (~500 MB extra).
    """
    api = HfApi(token=token)
    create_repo(repo_id=target, repo_type="dataset", private=private, token=token, exist_ok=True)

    # Read only Parquet footer metadata — no row data loaded
    print(f"[concat] Unifying schema from {len(shard_paths)} shards ...")
    schemas = [pq.read_schema(str(p)) for p in shard_paths]
    unified_schema = pa.unify_schemas(schemas, promote_options="default")

    arrow_ds = pad.dataset([str(p) for p in shard_paths], format="parquet", schema=unified_schema)
    total_rows = arrow_ds.count_rows()

    # Estimate upload-shard count from total on-disk bytes of Pass-1 shards
    total_bytes = sum(p.stat().st_size for p in shard_paths)
    num_upload_shards = max(1, -(-total_bytes // _MAX_SHARD_BYTES))
    rows_per_shard = -(-total_rows // num_upload_shards)
    print(
        f"[concat] {total_rows:,} rows, ~{total_bytes / 1e9:.1f} GB "
        f"→ ~{num_upload_shards} upload shards (~{rows_per_shard:,} rows each)"
    )

    upload_dir = tmp_root / "uploads"
    upload_dir.mkdir(exist_ok=True)

    shard_idx = 0
    rows_in_shard = 0
    total_uploaded = 0
    writer: pq.ParquetWriter | None = None
    current_path: Path | None = None

    def _flush() -> None:
        nonlocal writer, current_path, shard_idx, rows_in_shard, total_uploaded
        if writer is None:
            return
        writer.close()
        writer = None
        assert current_path is not None
        size_mb = current_path.stat().st_size / 1e6
        filename = f"data/train-{shard_idx + 1:05d}-of-{num_upload_shards:05d}.parquet"
        print(f"[concat] Uploading shard {shard_idx + 1}/{num_upload_shards} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(current_path),
            path_in_repo=filename,
            repo_id=target,
            repo_type="dataset",
            commit_message=f"{commit_message} (shard {shard_idx + 1}/{num_upload_shards})",
        )
        current_path.unlink()   # free disk immediately after upload
        current_path = None
        total_uploaded += 1
        shard_idx += 1
        rows_in_shard = 0

    for batch in arrow_ds.scanner(batch_size=_BATCH_ROWS).to_batches():
        if writer is None:
            current_path = upload_dir / f"shard_{shard_idx:05d}.parquet"
            writer = pq.ParquetWriter(str(current_path), unified_schema, compression="snappy")
        writer.write_batch(batch)
        rows_in_shard += batch.num_rows
        if rows_in_shard >= rows_per_shard:
            _flush()

    _flush()  # upload final partial shard

    print(f"[concat] Done — {total_uploaded} shard(s) at https://huggingface.co/datasets/{target}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Concatenate all HF training datasets referenced in Supabase models",
    )
    p.add_argument("--target", required=True, help="Target HF dataset repo (org/name)")
    p.add_argument("--private", action="store_true", help="Create target repo as private")
    p.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env var)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print datasets and extracted metadata without loading or pushing")
    p.add_argument("--unknown-only", action="store_true",
                   help="During --dry-run, show only rows where source or teacher is 'unknown'")
    p.add_argument("--skip-on-error", action="store_true",
                   help="Skip datasets that fail to load instead of aborting")
    p.add_argument("--commit-message", default="Concatenate all Supabase training datasets",
                   help="Commit message for push_to_hub")
    p.add_argument("--limit", type=int, default=None,
                   help="Load at most N datasets (useful for smoke-testing)")
    p.add_argument("--temp-dir", default=None,
                   help="Directory for Parquet shards (default: system temp, deleted on exit)")
    p.add_argument("--keep-temp", action="store_true",
                   help="Do not delete the temp shard directory after upload")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    print("[concat] Fetching dataset refs from Supabase...")
    refs = _fetch_all_dataset_refs()
    refs = [r for r in refs if not _should_drop(r)]
    if args.limit:
        refs = refs[: args.limit]
    print(f"[concat] Found {len(refs)} unique HF dataset refs")

    if args.dry_run:
        print(f"{'repo_id':<70}  {'source':<20}  teacher")
        print("-" * 110)
        for ref in refs:
            source = _extract_source(ref)
            teacher = _extract_teacher(ref)
            if args.unknown_only and source != "unknown" and teacher != "unknown":
                continue
            print(f"{ref:<70}  {source:<20}  {teacher}")
        return

    # Set up temp directory for Parquet shards
    managed_tmp = args.temp_dir is None
    tmp_root = Path(args.temp_dir) if args.temp_dir else Path(tempfile.mkdtemp(prefix="concat_shards_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"[concat] Shard dir: {tmp_root}")

    shard_paths: list[Path] = []
    failed: list[str] = []
    total_rows = 0

    try:
        # ── Pass 1: load each dataset, write shard, free memory ───────────────
        for i, ref in enumerate(refs, 1):
            print(f"[concat] [{i}/{len(refs)}] {ref} ...", end=" ", flush=True)
            shard_path = tmp_root / f"{i:04d}.parquet"
            try:
                n = _load_and_write_shard(ref, shard_path)
                shard_paths.append(shard_path)
                total_rows += n
                print(f"{n:,} rows")
            except Exception as exc:
                print(f"FAILED: {exc}")
                failed.append(ref)
                if not args.skip_on_error:
                    print("[concat] Aborting. Use --skip-on-error to continue past failures.",
                          file=sys.stderr)
                    sys.exit(1)

        if not shard_paths:
            print("[concat] No datasets loaded — nothing to push.", file=sys.stderr)
            sys.exit(1)

        # ── Pass 2: stream-merge shards → upload shards → HF Hub ─────────────
        print(f"\n[concat] Merging {len(shard_paths)} shards ({total_rows:,} rows) and uploading ...")
        _merge_and_upload(
            shard_paths, args.target, args.private, token, args.commit_message, tmp_root
        )

    finally:
        if managed_tmp and not args.keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
        elif args.keep_temp:
            print(f"[concat] Temp shards kept at: {tmp_root}")

    if failed:
        print(f"\n[concat] {len(failed)} dataset(s) skipped due to errors:")
        for ref in failed:
            print(f"  {ref}")


if __name__ == "__main__":
    main()
