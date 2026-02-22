#!/usr/bin/env python3
"""
Patch existing HF datasets with fixed Dockerfiles and test.sh scripts,
then run pass rate tests.
"""
import io
import os
import sys
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterator

from huggingface_hub import snapshot_download
from datasets import load_dataset
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

sys.path.append(str(Path(__file__).parent.parent))
from data.commons import upload_tasks_to_hf

# =============================================================================
# Tar extraction utilities (from tasks_parquet_converter.py)
# =============================================================================

def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts)) if parts else ""


def safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)


def _iter_archive_entries(root: Path) -> Iterator[tuple[Path, bool]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        yield dir_path, True
        for filename in filenames:
            yield dir_path / filename, False


def build_tar_bytes(task_dir: Path, compression: str = "gz") -> bytes:
    mode = "w:gz" if compression == "gz" else "w"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        for path, is_dir in _iter_archive_entries(task_dir):
            rel = path.relative_to(task_dir)
            if rel == Path("."):
                continue
            arcname = str(PurePosixPath(rel.as_posix()))
            if is_dir:
                arcname_dir = arcname.rstrip("/") + "/"
                info = tarfile.TarInfo(name=arcname_dir)
                info.type = tarfile.DIRTYPE
                stat_result = path.stat()
                info.mode = stat_result.st_mode
                info.mtime = stat_result.st_mtime
                info.uid = getattr(stat_result, "st_uid", 0)
                info.gid = getattr(stat_result, "st_gid", 0)
                info.size = 0
                tf.addfile(info)
            else:
                tf.add(path, arcname=arcname, recursive=False)
    return buf.getvalue()


def from_hf_dataset(dataset_name: str) -> Path:
    """Download a HuggingFace dataset and extract tasks to a directory."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    fingerprint = dataset["train"]._fingerprint
    safe_dataset_name = dataset_name.replace("/", "--")
    dataset_cache_dir = Path(tempfile.gettempdir()) / "hf_datasets" / safe_dataset_name / fingerprint[:12]

    marker_file = dataset_cache_dir / ".download_complete"
    if dataset_cache_dir.exists() and marker_file.exists():
        print(f"Dataset already cached at: {dataset_cache_dir}")
        return dataset_cache_dir

    print(f"Extracting dataset to: {dataset_cache_dir}")

    if dataset_cache_dir.exists():
        shutil.rmtree(dataset_cache_dir)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    # Save to parquet and extract
    temp_parquet_dir = tempfile.mkdtemp()
    temp_parquet = Path(temp_parquet_dir) / "dataset.parquet"

    try:
        dataset["train"].to_parquet(str(temp_parquet))

        # Extract from parquet
        table = pq.read_table(temp_parquet)
        cols = {name: i for i, name in enumerate(table.column_names)}
        path_col = table.column(cols["path"]).to_pylist()
        data_col = table.column(cols["task_binary"]).to_pylist()

        for rel_path, data in tqdm(zip(path_col, data_col), total=len(path_col), desc="Extracting"):
            safe_rel = PurePosixPath(rel_path)
            parts = [p for p in safe_rel.parts if p not in ("..", "")]
            rel_norm = Path(*parts)
            target_dir = (dataset_cache_dir / rel_norm).resolve()
            safe_extract_tar(bytes(data), target_dir)

        marker_file.write_text(f"fingerprint={fingerprint}\ndataset_name={dataset_name}\n")
        return dataset_cache_dir
    finally:
        shutil.rmtree(temp_parquet_dir, ignore_errors=True)


# =============================================================================
# Fixed Dockerfiles for each language
# =============================================================================

FIXED_DOCKERFILES = {
    "go": """FROM golang:1.21-alpine

WORKDIR /app

# Install bash (required by Harbor agent) and git
RUN apk add --no-cache bash git
""",
    "ruby": """FROM ruby:3.2-slim

WORKDIR /app

# Install build tools required for native gem extensions, plus bash for Harbor agent
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    bash \\
    && rm -rf /var/lib/apt/lists/*

RUN gem install rspec minitest
""",
    "rust": """FROM rust:1.75-slim

WORKDIR /app

# Install build tools, bash for Harbor agent, and common dependencies
RUN apt-get update && apt-get install -y \\
    pkg-config \\
    libssl-dev \\
    bash \\
    && rm -rf /var/lib/apt/lists/*

# Pre-create Cargo.toml with common dependencies
RUN cargo init --name solution && \\
    echo '[dependencies]' >> Cargo.toml && \\
    echo 'serde = { version = "1.0", features = ["derive"] }' >> Cargo.toml && \\
    echo 'serde_json = "1.0"' >> Cargo.toml && \\
    echo 'regex = "1"' >> Cargo.toml
""",
    "jest": """FROM node:20-slim

WORKDIR /app

# Install test frameworks globally
RUN npm install -g jest mocha chai

# Create minimal jest.config.js so Jest can find tests
RUN echo 'module.exports = { testEnvironment: "node", testMatch: ["**/tests/**/*.js", "**/*.test.js", "**/*.spec.js"] };' > /app/jest.config.js
""",
    "php": """FROM php:8.2-cli

WORKDIR /app

# Install required tools including bash for Harbor agent
RUN apt-get update && apt-get install -y unzip git bash && rm -rf /var/lib/apt/lists/*

# Install Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

# Install PHPUnit globally
RUN composer global require phpunit/phpunit
ENV PATH="${PATH}:/root/.composer/vendor/bin"

# Create autoload file for test classes
RUN echo '<?php spl_autoload_register(function($class) { $file = "/app/" . str_replace("\\\\", "/", $class) . ".php"; if (file_exists($file)) require $file; });' > /app/autoload.php
""",
}

# Fixed test.sh for pytest (handles PEP 668)
FIXED_PYTEST_TEST_SH = '''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}
trap cleanup EXIT

# Create virtual environment if needed (fixes PEP 668 on Python 3.12+)
if [ ! -d /app/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv /app/.venv
fi

# Activate virtual environment
source /app/.venv/bin/activate

# Install pytest if not available
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install --quiet pytest
fi

cd /app

echo "Running pytest..."
pytest /tests/test_solution.py -v --tb=short 2>&1 | tee /logs/verifier/pytest_output.txt

PYTEST_EXIT=${PIPESTATUS[0]}

if [ $PYTEST_EXIT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
'''

# Mapping of dataset names to their fixes
DATASET_FIXES = {
    "DCAgent/exp_rpt_stack-go": {"dockerfile": FIXED_DOCKERFILES["go"]},
    "DCAgent/exp_rpt_stack-ruby": {"dockerfile": FIXED_DOCKERFILES["ruby"]},
    "DCAgent/exp_rpt_stack-rust": {"dockerfile": FIXED_DOCKERFILES["rust"]},
    "DCAgent/exp_rpt_stack-jest": {"dockerfile": FIXED_DOCKERFILES["jest"]},
    "DCAgent/exp_rpt_stack-php": {"dockerfile": FIXED_DOCKERFILES["php"]},
    "DCAgent/exp_rpt_stack-pytest": {"test_sh": FIXED_PYTEST_TEST_SH},
    "DCAgent/exp_rpt_stack-pytest-withtests": {"test_sh": FIXED_PYTEST_TEST_SH},
    "DCAgent/exp_rpt_stack-selfdoc": {"test_sh": FIXED_PYTEST_TEST_SH},
    "DCAgent/exp_rpt_stack-selfdoc-gpt4o": {"test_sh": FIXED_PYTEST_TEST_SH},
}


def patch_dataset(dataset_id: str, fixes: dict, output_suffix: str = "-v2") -> str:
    """
    Download dataset, apply fixes, and upload to new repo.
    """
    print(f"\n{'='*60}")
    print(f"Patching: {dataset_id}")
    print(f"{'='*60}")

    # Download and extract dataset
    print("Downloading dataset...")
    dataset_path = from_hf_dataset(dataset_id)
    print(f"Downloaded to: {dataset_path}")

    # Create temp directory for patched dataset
    temp_dir = tempfile.mkdtemp(prefix="patched_")

    # Find and patch each task
    task_dirs = sorted(Path(dataset_path).glob("*/instruction.md"))
    print(f"Found {len(task_dirs)} tasks to patch")

    for task_md in tqdm(task_dirs, desc="Patching tasks"):
        task_dir = task_md.parent
        target_dir = Path(temp_dir) / task_dir.name

        # Copy entire task directory
        shutil.copytree(task_dir, target_dir)

        # Apply Dockerfile fix if specified
        if "dockerfile" in fixes:
            dockerfile_path = target_dir / "environment" / "Dockerfile"
            if dockerfile_path.exists() or (target_dir / "environment").exists():
                (target_dir / "environment").mkdir(exist_ok=True)
                dockerfile_path.write_text(fixes["dockerfile"], encoding="utf-8")

        # Apply test.sh fix if specified
        if "test_sh" in fixes:
            test_sh_path = target_dir / "tests" / "test.sh"
            if test_sh_path.exists():
                test_sh_path.write_text(fixes["test_sh"], encoding="utf-8")
                os.chmod(test_sh_path, 0o755)

    print(f"Patched {len(task_dirs)} tasks")

    # Upload to new repo
    new_repo_id = dataset_id + output_suffix
    print(f"Uploading to: {new_repo_id}")
    repo_url = upload_tasks_to_hf(temp_dir, new_repo_id)

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"Done! New dataset: {repo_url}")
    return repo_url


def main():
    """Patch all datasets with fixes."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to patch (default: all)")
    parser.add_argument("--suffix", default="-v2", help="Suffix for patched datasets")
    args = parser.parse_args()

    datasets_to_patch = args.datasets or list(DATASET_FIXES.keys())

    results = []
    for dataset_id in datasets_to_patch:
        if dataset_id not in DATASET_FIXES:
            print(f"Warning: No fixes defined for {dataset_id}, skipping")
            continue

        try:
            repo_url = patch_dataset(dataset_id, DATASET_FIXES[dataset_id], args.suffix)
            results.append((dataset_id, repo_url, "success"))
        except Exception as e:
            print(f"Error patching {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append((dataset_id, None, str(e)))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset_id, repo_url, status in results:
        if status == "success":
            print(f"OK: {dataset_id} -> {repo_url}")
        else:
            print(f"FAIL: {dataset_id} - {status}")


if __name__ == "__main__":
    main()
