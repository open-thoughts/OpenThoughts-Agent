"""Re-export literal-token Iris traces from durable GCS on a high-memory worker."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from datasets import load_dataset
from google.cloud import storage

from scripts.harbor.make_and_upload_trace_dataset import count_populated_literal_rows


DOWNLOAD_WORKERS = 16


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a non-empty GCS prefix into its bucket and normalized object prefix."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    bucket, separator, prefix = uri[5:].partition("/")
    if not bucket or not separator or not prefix:
        raise ValueError(f"Expected gs://bucket/prefix, got {uri!r}")
    return bucket, f"{prefix.rstrip('/')}/"


def safe_destination(root: Path, relative_key: str) -> Path:
    """Map a GCS object key to a worker-local path without traversal."""
    relative_path = PurePosixPath(relative_key)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe relative GCS object key: {relative_key!r}")
    return root.joinpath(*relative_path.parts)


def download_prefix(source_prefix: str, destination: Path) -> int:
    """Download one durable GCS prefix with bounded concurrency."""
    bucket_name, prefix = parse_gs_uri(source_prefix)
    storage.Client().bucket(bucket_name)
    blobs = [
        blob
        for blob in storage.Client().list_blobs(bucket_name, prefix=prefix)
        if blob.name != prefix
    ]
    if not blobs:
        raise RuntimeError(f"No objects found under {source_prefix}")

    def download(blob: storage.Blob) -> None:
        relative = blob.name.removeprefix(prefix)
        target = safe_destination(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)

    print(f"[literal-reexport] downloading {len(blobs)} GCS objects", flush=True)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(download, blob) for blob in blobs]
        for copied, future in enumerate(as_completed(futures), start=1):
            future.result()
            if copied % 500 == 0 or copied == len(blobs):
                print(
                    f"[literal-reexport] downloaded {copied}/{len(blobs)} objects",
                    flush=True,
                )
    return len(blobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--served-model", required=True)
    args = parser.parse_args()

    temporary_root = Path(tempfile.mkdtemp(prefix=f"{args.job_name}-literal-"))
    try:
        download_prefix(args.source_prefix, temporary_root)
        job_dir = temporary_root / args.job_name
        if not job_dir.is_dir():
            raise RuntimeError(f"Expected inner Harbor job directory at {job_dir}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.harbor.make_and_upload_trace_dataset",
                "--job_dir",
                str(job_dir),
                "--repo_id",
                args.repo_id,
                "--episodes",
                "last",
                "--served_model",
                args.served_model,
                "--include_literal_tokens",
                "--single_commit",
                "--skip_register",
            ],
            check=True,
        )
        dataset = load_dataset(args.repo_id, split="train")
        populated = count_populated_literal_rows(dataset.data.table)
        if not populated:
            raise RuntimeError(
                f"{args.repo_id} has no populated literal rows after re-export"
            )
        print(
            f"[literal-reexport] verified {args.repo_id}: {len(dataset)} rows, "
            f"{populated} with literals",
            flush=True,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
