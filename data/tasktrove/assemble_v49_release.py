#!/usr/bin/env python3
"""Validate and assemble the TaskTrove v4.9 remediation release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from data.tasktrove.build_swe_verifier_isolation_v49 import (
    OPENSWE_STORAGE_MB,
    REVISIONS,
    SOURCE_REVISION,
    SWE_REBENCH_MEMORY_MB,
    SWE_REBENCH_STORAGE_MB,
    file_sha256,
    validate_output,
)

MAX_IMAGES = 20


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def assemble(repair_stage: Path, stage: Path) -> None:
    if stage.exists() and any(stage.iterdir()):
        raise ValueError(f"stage must be absent or empty: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    images: set[str] = set()
    for revision in REVISIONS:
        source = repair_stage / "datasets" / revision.output / "tasks.parquet"
        rows = pq.ParquetFile(source).metadata.num_rows
        source_parquet = Path(
            hf_hub_download(
                "open-thoughts/TaskTrove",
                f"{revision.source}/tasks.parquet",
                repo_type="dataset",
                revision=SOURCE_REVISION,
            )
        )
        source_rows = pq.ParquetFile(source_parquet).metadata.num_rows
        images.update(validate_output(source, revision, rows))
        relative = Path("datasets") / revision.output / "tasks.parquet"
        _hardlink_or_copy(source, stage / relative)
        entries.append(
            {
                "source": revision.source,
                "source_sha256": revision.source_sha256,
                "source_rows": source_rows,
                "output": revision.output,
                "output_rows": rows,
                "rejected_rows": source_rows - rows,
                "output_sha256": file_sha256(source),
                "parquet": str(relative),
                "verifier": revision.verifier,
                "storage_mb": {
                    "swe_rebench": SWE_REBENCH_STORAGE_MB,
                    "openswe": OPENSWE_STORAGE_MB,
                }.get(revision.verifier),
                "memory_mb": {
                    "swe_rebench": SWE_REBENCH_MEMORY_MB,
                }.get(revision.verifier),
            }
        )
    if len(images) > MAX_IMAGES:
        raise ValueError(f"release uses {len(images)} images, above limit {MAX_IMAGES}")
    manifest = {
        "source_repo": "open-thoughts/TaskTrove",
        "source_revision": SOURCE_REVISION,
        "source_version": "4.8",
        "target_version": "4.9",
        "datasets": entries,
        "release_unique_images": len(images),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"assembled {len(entries)} replacements using {len(images)} unique images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-stage", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.repair_stage, args.stage)


if __name__ == "__main__":
    main()
