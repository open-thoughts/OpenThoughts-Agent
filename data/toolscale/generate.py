#!/usr/bin/env python3
"""Generate fixture-backed Harbor tasks from the pinned ToolScale source."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.commons import create_task_directory_unified, finalize_dataset_output, upload_tasks_to_hf
from data.toolscale.deterministic_execution import (
    DOCKERFILE,
    SOURCE_FILE,
    SOURCE_REPO,
    SOURCE_REVISION,
    TASK_TOML,
    build_domain_catalog,
    expected_fixture,
    render_check,
    render_instruction,
    render_runtime,
    render_solution,
    render_test_sh,
    selected_by_v3,
    task_domain,
)

BATCH_SIZE = 32


def add_toolscale_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dataset-prefix", default="toolscale-v4")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-parquet")
    parser.add_argument("--target-repo")
    parser.add_argument("--hf-token")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser


def source_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return Path(
        hf_hub_download(
            SOURCE_REPO,
            SOURCE_FILE,
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
    )


def source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
        rows.extend(row for row in batch.to_pylist() if selected_by_v3(row))
    return rows


def generate_tasks(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    rows = source_rows(source_path(args.source_parquet))
    catalog = build_domain_catalog(rows)
    start = max(0, args.offset)
    selected = rows[start : None if args.limit <= 0 else start + args.limit]
    output = Path(tempfile.mkdtemp(prefix="toolscale-v4-tasks-"))
    for selected_index, row in enumerate(selected, start=start):
        task_id = f"toolscale-v4-{selected_index:04d}"
        fixture = expected_fixture(row, task_id)
        metadata = {
            "id": row.get("id"),
            "repair": "fixture_backed_tool_execution_v4",
            "source": f"{SOURCE_REPO}@{SOURCE_REVISION}",
            "source_index": selected_index,
        }
        task_dir = Path(
            create_task_directory_unified(
                output_dir=output,
                task_id=selected_index,
                instruction_content=render_instruction(row, task_id),
                dataset_prefix=args.dataset_prefix,
                metadata=metadata,
                solution_content=render_solution(fixture),
                test_sh_content=render_test_sh(),
                test_py_content=render_check(),
                task_toml_content=TASK_TOML,
                dockerfile_content=DOCKERFILE,
            )
        )
        (task_dir / "tests" / "test_state.py").rename(task_dir / "tests" / "check.py")
        (task_dir / "environment" / "toolscale_runtime.py").write_text(
            render_runtime(fixture, catalog[task_domain(row)]), encoding="utf-8"
        )
        (task_dir / "tests" / "expected.json").write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not selected:
        raise RuntimeError("No ToolScale tasks selected")
    return output, {
        "produced_tasks": len(selected),
        "source": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
    }


def main() -> None:
    parser = add_toolscale_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    generated, metadata = generate_tasks(args)
    final_path = finalize_dataset_output(generated, args.output_dir)
    print(json.dumps({"output_dir": str(final_path), **metadata}, indent=2))
    if args.target_repo and not args.no_upload:
        upload_tasks_to_hf(
            dataset_path=str(final_path),
            repo_id=args.target_repo,
            private=args.hf_private,
            token=args.hf_token,
        )


if __name__ == "__main__":
    main()
