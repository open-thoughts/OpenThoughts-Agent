from __future__ import annotations

"""
Bi-directional converter between a directory of Harbor tasks and a Parquet
dataset (HF-compatible) with one row per task.

Schema:
- path: str (relative path from the provided base directory to the task dir)
- task_binary: binary (tar archive bytes; defaults to gzip-compressed)

Usage examples:

  # Directory -> Parquet (recursive)
  python scripts/harbor/tasks_parquet_converter.py to-parquet \
      --base harbor/tasks \
      --out tasks.parquet \
      --recursive \
      --compression gz

  # Directory -> Parquet (fixed depth: 1 means direct children)
  python scripts/harbor/tasks_parquet_converter.py to-parquet \
      --base harbor/tasks \
      --out tasks.parquet \
      --depth 1

  # Parquet -> Directory (extract back)
  python scripts/harbor/tasks_parquet_converter.py from-parquet \
      --parquet tasks.parquet \
      --base harbor/tasks_restored \
      --on-exist error

  # Roundtrip test (compress then decompress and verify)
  python scripts/harbor/tasks_parquet_converter.py roundtrip-test \
      --base harbor/tasks \
      --out /tmp/rt \
      --recursive \
      --compression gz

Notes:
- Requires pyarrow to write/read Parquet (install via `pip install pyarrow`).
- Task detection relies on presence of an `instruction.md` inside the task directory.
"""

import argparse
import io
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Sequence
from tqdm import tqdm
from datasets import load_dataset
from data.gcs_cache import gcs_cache

# Optional heavy imports; provide clear error if missing when used.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - import-time availability varies
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


TASK_MARKER = "instruction.md"


@dataclass(frozen=True)
class TaskRecord:
    rel_path: str
    archive_bytes: bytes


def _require_pyarrow() -> tuple[object, object]:
    if pa is None or pq is None:  # type: ignore[truthy-bool]
        raise RuntimeError(
            "pyarrow is required. Please install it: pip install pyarrow"
        )
    return pa, pq  # type: ignore[return-value]

def find_logs(base: Path) -> list[Path]:
    """Return all subdirectories under the given base directory."""
    return sorted([p for p in base.iterdir() if p.is_dir()])

def find_tasks(
    base: Path,
    recursive: bool = False,
    depth: int | None = None,
    marker: str | Sequence[str] = TASK_MARKER,
) -> list[Path]:
    """Find task directories containing the marker file.

    - recursive: walk entire subtree and collect any directory containing the marker.
      When a directory is identified as a task, do not descend further under it.
    - depth: fixed search depth relative to base (1 = direct children). Ignored if recursive is True.
    """
    if isinstance(marker, str):
        markers = (marker,)
    else:
        markers = tuple(marker)

    def has_marker(path: Path) -> bool:
        return any((path / m).is_file() for m in markers)

    if recursive:
        out: list[Path] = []
        for root, dirs, files in os.walk(base):
            root_path = Path(root)
            if has_marker(root_path):
                out.append(root_path)
                # Do not search within a detected task dir
                dirs[:] = []
        return sorted(out)

    # Fixed depth search (default: depth=1)
    if depth is None:
        depth = 1
    if depth < 0:
        raise ValueError("depth must be >= 0")

    # BFS by depth
    current_level: list[Path] = [base]
    for _ in range(depth):
        next_level: list[Path] = []
        for p in current_level:
            if not p.is_dir():
                continue
            for child in p.iterdir():
                if child.is_dir():
                    next_level.append(child)
        current_level = next_level

    out = [p for p in current_level if has_marker(p)]
    return sorted(out)


def _iter_archive_entries(root: Path) -> Iterator[tuple[Path, bool]]:
    """Yield (path, is_dir) pairs for all descendants under ``root``.

    Directories are yielded before their contained files so that empty
    directories are preserved in the tar archive.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        yield dir_path, True
        for filename in filenames:
            yield dir_path / filename, False


def build_tar_bytes(task_dir: Path, compression: str = "gz") -> bytes:
    """Create an in-memory tar archive of the task directory.

    compression: one of {"none", "gz"}. Default is gzip.
    """
    mode = "w"
    if compression == "gz":
        mode = "w:gz"
    elif compression == "none":
        mode = "w"
    else:
        raise ValueError("compression must be one of: none, gz")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        # Add directories and files with relative paths, preserving empty dirs.
        for path, is_dir in _iter_archive_entries(task_dir):
            rel = path.relative_to(task_dir)
            if rel == Path("."):
                continue

            arcname = str(PurePosixPath(rel.as_posix()))

            if is_dir:
                # Ensure directory entries end with '/' and have sensible defaults.
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
                # Normalize to posix inside the tar.
                tf.add(path, arcname=arcname, recursive=False)
    return buf.getvalue()

def convert_to_parquet(tasks_dir: str) -> str:
    tasks = find_tasks(Path(tasks_dir), recursive=True)
    temp_dir = tempfile.mkdtemp()
    temp_folder = Path(temp_dir)
    parquet_path = temp_folder / "tasks.parquet"
    records = to_parquet(Path(tasks_dir), parquet_path, tasks, compression="gz")
    return str(temp_folder)

def convert_logs_to_parquet(tasks_dir: str) -> str:
    tasks = find_logs(Path(tasks_dir))
    temp_dir = tempfile.mkdtemp()
    temp_folder = Path(temp_dir)
    parquet_path = temp_folder / "logs.parquet"
    records = to_parquet(Path(tasks_dir), parquet_path, tasks, compression="gz")
    return str(temp_folder)

def to_parquet(
    base: Path,
    out_path: Path,
    tasks: Sequence[Path],
    compression: str = "gz",
) -> list[TaskRecord]:
    pa_mod, pq_mod = _require_pyarrow()
    records: list[TaskRecord] = []

    for t in tasks:
        rel = str(PurePosixPath(t.relative_to(base).as_posix()))
        archive = build_tar_bytes(t, compression=compression)
        records.append(TaskRecord(rel_path=rel, archive_bytes=archive))

    table = pa_mod.table({  # type: ignore[attr-defined]
        "path": pa_mod.array([r.rel_path for r in records], type=pa_mod.string()),
        "task_binary": pa_mod.array([r.archive_bytes for r in records], type=pa_mod.binary()),
    })
    pq_mod.write_table(table, out_path)
    return records


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    # Remove leading slashes and collapse to posix; strip .. components
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]  # keep '.' implicitly removed by PurePosixPath
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts))


def safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            # Sanitize name
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                # Directory entry; ensure exists
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                # Skip snapshot metadata that can be read-only on shared filesystems
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            # Ensure parent exists
            target.parent.mkdir(parents=True, exist_ok=True)
            # Extract regular files only; skip devices/symlinks for safety
            if member.isfile():
                with tf.extractfile(member) as src:  # type: ignore[assignment]
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                # Skip other types (symlinks, etc.) for safety
                continue

def from_hf_dataset(
    dataset_name: str,
    base: str | None = None,
    on_exist: str = "overwrite",
) -> Path:
    """Download a HuggingFace dataset and extract tasks to a directory.

    Uses the dataset's fingerprint to enable caching. If the dataset has
    already been downloaded with the same fingerprint, returns the cached path.

    Returns the base directory path where tasks were extracted.
    """
    import sys
    # Load dataset from HuggingFace (this uses HF's own caching)
    print(f"Loading dataset: {dataset_name}", file=sys.stderr)
    dataset = load_dataset(dataset_name)

    # Get the fingerprint which tracks content changes
    fingerprint = dataset["train"]._fingerprint
    print(f"Dataset fingerprint: {fingerprint}", file=sys.stderr)

    # Create a deterministic path in /tmp based on dataset name and fingerprint
    # Sanitize dataset name for use in path (replace / with --)
    safe_dataset_name = dataset_name.replace("/", "--")
    dataset_cache_dir = Path(tempfile.gettempdir()) / "hf_datasets" / safe_dataset_name / fingerprint[:12]

    # Check if dataset is already cached
    marker_file = dataset_cache_dir / ".download_complete"
    if dataset_cache_dir.exists() and marker_file.exists():
        print(f"Dataset already cached at: {dataset_cache_dir}")
        return dataset_cache_dir

    # If base is provided (legacy behavior), use that instead
    if base is not None:
        extraction_dir = Path(base)
    else:
        extraction_dir = dataset_cache_dir

    print(f"Extracting dataset to: {extraction_dir}")

    # Clean up partial downloads
    if extraction_dir.exists():
        shutil.rmtree(extraction_dir)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for the parquet file
    temp_parquet_dir = tempfile.mkdtemp()
    temp_parquet = Path(temp_parquet_dir) / "dataset.parquet"

    try:
        # Save the dataset to a parquet file
        dataset["train"].to_parquet(str(temp_parquet))

        # Extract tasks from the parquet file
        from_parquet(str(temp_parquet), base=str(extraction_dir), on_exist=on_exist)

        # Write marker file to indicate download is complete
        marker_file.write_text(f"fingerprint={fingerprint}\ndataset_name={dataset_name}\n")

        print(f"Dataset extraction complete: {extraction_dir}")
        return extraction_dir

    finally:
        # Clean up temporary parquet directory
        shutil.rmtree(temp_parquet_dir, ignore_errors=True)

@gcs_cache()
def from_hf_dataset_hdf5(
    dataset_name: str,
    output_path: str | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> str:
    """Download a HuggingFace dataset and convert directly to HDF5 format.

    This is much faster than from_hf_dataset for large datasets because:
    1. Converts directly from parquet to HDF5 without intermediate directory extraction
    2. Faster checksum computation (single file vs directory tree)
    3. Enables fast subsampling/upsampling operations

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'DCAgent/nl2bash')
        output_path: Path for output HDF5 file. If None, creates temp file.
        compression: HDF5 compression algorithm (gzip, lzf, or None)
        compression_opts: Compression level (0-9 for gzip)

    Returns:
        str: Path to created HDF5 file
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        raise ImportError("h5py and numpy are required for HDF5 conversion. Install with: pip install h5py numpy")

    pa_mod, pq_mod = _require_pyarrow()

    # Load dataset from HuggingFace
    print(f"Downloading HuggingFace dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    # Create temporary directory for parquet
    temp_dir = tempfile.mkdtemp()
    temp_parquet = Path(temp_dir) / "dataset.parquet"

    
    # Save to parquet
    print("Saving dataset to temporary parquet...")
    dataset["train"].to_parquet(str(temp_parquet))

    # Read parquet file
    print("Reading parquet file...")
    table = pq_mod.read_table(temp_parquet)
    cols = {name: i for i, name in enumerate(table.column_names)}

    if "path" not in cols or "task_binary" not in cols:
        raise RuntimeError("Parquet must have columns: 'path', 'task_binary'")

    path_col = table.column(cols["path"]).to_pylist()
    data_col = table.column(cols["task_binary"]).to_pylist()

    # Create HDF5 file
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix="tasks_hdf5_")
        os.close(fd)

    output_path = Path(output_path)
    print(f"Converting {len(path_col)} tasks directly to HDF5: {output_path}")

    with h5py.File(output_path, 'w') as h5file:
        for i, (rel_path, archive_bytes) in tqdm(enumerate(zip(path_col, data_col))):
            if i % 100 == 0:
                print(f"Processing task {i}/{len(path_col)}...")

            # Create group for this task
            task_group = h5file.create_group(f"task_{i:04d}")

            # Extract tar archive in memory and write to HDF5
            buf = io.BytesIO(bytes(archive_bytes))
            with tarfile.open(fileobj=buf, mode="r:*") as tf:
                # Store task metadata
                task_name = rel_path.split("/")[-1] if "/" in rel_path else rel_path
                task_group.attrs['task_name'] = task_name
                task_group.attrs['task_id'] = i
                task_group.attrs['dataset_prefix'] = task_name.rsplit("-", 1)[0] if "-" in task_name else "task"

                # Extract each file from tar and write to HDF5
                seen_members = set()
                for member in tf.getmembers():
                    if not member.isfile():
                        continue

                    # Sanitize member name
                    member_name = _sanitize_tar_member_name(member.name)
                    if not member_name or ".snapshot" in member_name:
                        continue

                    # Skip duplicates
                    if member_name in seen_members:
                        continue
                    seen_members.add(member_name)

                    # Read file content
                    file_obj = tf.extractfile(member)
                    if file_obj is None:
                        continue

                    content = file_obj.read()

                    # Try to decode as text, fallback to binary
                    # Files with NULL bytes or invalid UTF-8 are treated as binary
                    is_binary = False
                    try:
                        text_content = content.decode('utf-8')
                        # Check for embedded NULL bytes (not supported by HDF5 strings)
                        if '\x00' in text_content:
                            is_binary = True
                    except (UnicodeDecodeError, ValueError):
                        is_binary = True

                    if not is_binary:
                        # Scalar string datasets don't support compression in HDF5
                        task_group.create_dataset(
                            member_name,
                            data=text_content,
                            dtype=h5py.string_dtype(encoding='utf-8')
                        )
                    else:
                        # Store as binary (arrays support compression)
                        task_group.create_dataset(
                            member_name,
                            data=np.frombuffer(content, dtype=np.uint8),
                            compression=compression,
                            compression_opts=compression_opts if compression == 'gzip' else None
                        )
                        task_group[member_name].attrs['binary'] = True

    print(f"Successfully converted HuggingFace dataset to HDF5: {output_path}")
    return str(output_path)


def _process_parquet_row(args: tuple[int, str, bytes, Path, str]) -> Path | None:
    """Helper function to process a single parquet row in parallel.

    Returns the target_dir Path if successful, or None if skipped.
    Raises exceptions on error to propagate back to main process.
    """
    i, rel_path, data, base, on_exist = args

    if not isinstance(rel_path, str):
        raise RuntimeError(f"Row {i}: 'path' must be a string")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise RuntimeError(f"Row {i}: 'task_binary' must be bytes")

    safe_rel = PurePosixPath(rel_path)
    # Normalize and strip unsafe components
    parts = [p for p in safe_rel.parts if p not in ("..", "")]  # '.' dropped
    rel_norm = Path(*parts)
    target_dir = (base / rel_norm).resolve()
    if not _is_within(base, target_dir):
        raise RuntimeError(f"Unsafe target path: {rel_path}")

    if target_dir.exists():
        if on_exist == "skip":
            return None
        if on_exist == "error":
            raise FileExistsError(f"Target exists: {target_dir}")
        if on_exist == "overwrite":
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
            else:
                target_dir.unlink()
        else:
            raise ValueError("on_exist must be one of: skip, overwrite, error")

    safe_extract_tar(bytes(data), target_dir)
    return target_dir


def from_parquet(
    parquet_path: str,
    base: str,
    on_exist: str = "error",
    max_workers: int = 10,
) -> list[Path]:
    """Extract tasks from parquet file to directory in parallel.

    Args:
        parquet_path: Path to parquet file
        base: Base directory to extract to
        on_exist: What to do if target exists ('skip', 'error', 'overwrite')
        max_workers: Number of parallel workers (10 by default)

    Returns:
        List of extracted task directories
    """
    pa_mod, pq_mod = _require_pyarrow()
    table = pq_mod.read_table(parquet_path)
    cols = {name: i for i, name in enumerate(table.column_names)}
    if "path" not in cols or "task_binary" not in cols:
        raise RuntimeError("Parquet must have columns: 'path', 'task_binary'")

    base = Path(base).resolve()
    path_col = table.column(cols["path"]).to_pylist()
    data_col = table.column(cols["task_binary"]).to_pylist()

    # Prepare arguments for parallel processing
    tasks_args = [
        (i, rel_path, data, base, on_exist)
        for i, (rel_path, data) in enumerate(zip(path_col, data_col))
    ]

    written: list[Path] = []

    # Process tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_parquet_row, args) for args in tasks_args]

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting tasks"):
            result = future.result()
            if result is not None:
                written.append(result)

    return written
