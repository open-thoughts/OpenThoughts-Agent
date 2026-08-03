"""Publish tasktrove_v2 to open-thoughts/TaskTrove main.

 1. Copy v2 README into tasktrove_v2/README.md (the staging dir).
 2. Determine v1-only top-level subdirs to delete (anything on current main that
    is NOT one of our 96 v2 names).
 3. upload_folder with delete_patterns covering all obsolete v1 subdirs.

Pre-req: v1 tag already created.
"""
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasktrove_v2_datasets import ALL

from huggingface_hub import HfApi

REPO = "open-thoughts/TaskTrove"
STAGING = Path("/Users/benjaminfeuer/Documents/scaling_laws_papers/tasktrove_v2")
README_SRC = Path("/Users/benjaminfeuer/Documents/scaling_laws_papers/tasktrove_v2_README.md")


def v2_dir_names() -> set[str]:
    return {repo.replace("/", "__") for repo, _ in ALL}


def main():
    api = HfApi()
    keep = v2_dir_names()
    assert len(keep) == 96, f"expected 96 dataset dirs, got {len(keep)}"

    # 1. README
    shutil.copyfile(README_SRC, STAGING / "README.md")
    print(f"Copied README -> {STAGING}/README.md ({(STAGING / 'README.md').stat().st_size} bytes)")

    # 2. Compute v1-only dirs to delete
    files = api.list_repo_files(repo_id=REPO, repo_type="dataset", revision="main")
    top_dirs = sorted({f.split("/")[0] for f in files if "/" in f})
    to_delete = sorted([d for d in top_dirs if d not in keep])
    print(f"Current main: {len(top_dirs)} top-level dirs")
    print(f"Will delete: {len(to_delete)} v1-only dirs not in v2")

    # 3. Sanity check staged dirs match expected set
    staged = {p.name for p in STAGING.iterdir() if p.is_dir()}
    missing = keep - staged
    if missing:
        print(f"MISSING v2 dirs ({len(missing)}): {sorted(missing)}")
        sys.exit(1)
    print("All 96 v2 dirs present in staging.")

    # 4. delete_patterns for every depth
    delete_patterns = []
    for d in to_delete:
        delete_patterns.append(f"{d}/*")
        delete_patterns.append(f"{d}/**/*")

    print(f"\nDelete patterns: {len(delete_patterns)}")
    print(f"  e.g. {delete_patterns[:4]}")
    print("\nUploading via api.upload_folder (single commit with deletes) ...")
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=REPO,
        repo_type="dataset",
        commit_message="v2: replace with 96 vetted, smoke-tested datasets; tag v1 preserves prior state",
        commit_description="See README for the full index, difficulty buckets, and v1 resolution instructions.",
        ignore_patterns=[".cache/**", "staging.log", "staging.stdout.log"],
        delete_patterns=delete_patterns if delete_patterns else None,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
