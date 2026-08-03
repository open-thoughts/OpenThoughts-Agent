"""Orchestrate the full RL conversion pipeline for a source.

Pipeline per source:
1. Download HF task set
2. Rewire each task with a sound verifier (language-specific test.sh + Dockerfile)
3. Build Docker image from the shared Dockerfile
4. Run oracle gate (gold→1 AND empty→0), drop failures
5. Package survivors as Harbor task binaries (gzip-tar → parquet)
6. Upload to laion/<source>-v2 on HF
7. Replace in TaskTrove (version bump)
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def make_task_binary(
    files: dict[str, str | bytes],
) -> bytes:
    """Create a gzip-tar task binary from a dict of {path: content}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            ti = tarfile.TarInfo(name=name)
            ti.size = len(content)
            tar.addfile(ti, io.BytesIO(content))
    return gzip.compress(buf.getvalue())


def tasks_to_parquet(
    tasks: list[tuple[str, bytes]],
    output_path: str | Path,
) -> None:
    """Write a list of (path, task_binary) to a parquet file."""
    table = pa.table({
        "path": [p for p, _ in tasks],
        "task_binary": [b for _, b in tasks],
    })
    pq.write_table(table, str(output_path))


def upload_to_hf(
    parquet_path: str | Path,
    repo_id: str,
    readme: str | None = None,
) -> None:
    """Upload parquet (+ optional README) to HF as a dataset."""
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="tasks.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    if readme:
        readme_path = Path(parquet_path).parent / "README.md"
        readme_path.write_text(readme)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")


def replace_in_tasktrove(
    new_subdir_name: str,
    parquet_path: str | Path,
    old_subdir_name: str,
    version_note: str,
    readme_md: str,
) -> None:
    """Replace a TaskTrove subdir and bump the version note in README."""
    from huggingface_hub import hf_hub_download, HfApi
    import tempfile

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    REPO = "open-thoughts/TaskTrove"

    with tempfile.TemporaryDirectory() as stage:
        stage_p = Path(stage)

        # Stage new subdir
        new_dir = stage_p / new_subdir_name
        new_dir.mkdir(parents=True)
        import shutil
        shutil.copy2(parquet_path, new_dir / "tasks.parquet")

        # Download + update README
        readme_path = hf_hub_download(
            REPO, "README.md", repo_type="dataset",
            local_dir=stage, token=token,
        )
        readme = Path(readme_path).read_text()

        # Bump version
        import re
        version_match = re.search(r"v3\.(\d+)\s*\(current\)", readme)
        if version_match:
            current_ver = int(version_match.group(1))
            new_ver = current_ver + 1
            old_header = f"v3.{current_ver} (current)"
            new_header = f"v3.{new_ver} (current)"
            old_line = f"> **{old_header}**"
            new_block = (
                f"> **{new_header}** — {version_note}\n>\n"
                f"> **v3.{current_ver}**"
            )
            readme = readme.replace(old_line, new_block, 1)

        readme_path = stage_p / "README.md"
        readme_path.write_text(readme)

        # Upload with delete_patterns for old subdir
        api.upload_folder(
            folder_path=str(stage),
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"{new_header}: replace {old_subdir_name} with {new_subdir_name}",
            delete_patterns=[
                f"{old_subdir_name}/*",
                f"{old_subdir_name}/**/*",
            ],
            token=token,
        )

    print(f"TaskTrove updated: {old_subdir_name} → {new_subdir_name}")
