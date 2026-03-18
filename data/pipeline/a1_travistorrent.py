#!/usr/bin/env python3
"""
Generate tasks from TravisTorrent dataset (100K+ CI builds).

TravisTorrent provides:
- 100K+ analyzed Travis CI builds
- Test counts, pass/fail status, test frameworks used
- Links to GitHub repos with passing tests
"""

import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import (
    TEST_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 1000
MODEL = "gpt-4o-mini"


def load_travistorrent(limit: int = LIMIT) -> List[Dict]:
    """
    Load TravisTorrent dataset from local CSV file.

    TravisTorrent data is downloaded from Figshare:
    https://doi.org/10.6084/m9.figshare.19314170
    """
    print("Loading TravisTorrent dataset...")

    # Path to local CSV file
    csv_path = Path(__file__).parent / "datasets" / "travistorrent.csv"

    if not csv_path.exists():
        raise ValueError(
            f"TravisTorrent CSV not found at {csv_path}. "
            "Please download from https://doi.org/10.6084/m9.figshare.19314170"
        )

    print(f"Loading from local CSV: {csv_path}")

    # Load CSV with pandas for efficiency - only load needed columns
    needed_cols = [
        "tr_build_id", "gh_project_name", "gh_lang", "tr_status",
        "tr_log_num_tests_ok", "tr_log_frameworks", "git_trigger_commit"
    ]

    # Read CSV in chunks to handle large file
    samples = []
    chunk_size = 100000

    # Target languages - filter during loading for efficiency
    target_languages = {"python", "java", "javascript"}

    print(f"Collecting up to {limit} samples with passing tests (Python/Java/JavaScript only)...")

    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size, usecols=needed_cols),
                      desc="Processing chunks"):
        # Filter for passed builds with at least 5 passing tests AND target languages
        chunk["gh_lang_lower"] = chunk["gh_lang"].str.lower()
        mask = (
            (chunk["tr_status"] == "passed") &
            (chunk["tr_log_num_tests_ok"] >= 5) &
            (chunk["gh_lang_lower"].isin(target_languages))
        )
        filtered = chunk[mask]

        for _, row in filtered.iterrows():
            repo_name = row["gh_project_name"]
            if pd.isna(repo_name) or not repo_name:
                continue

            # Construct GitHub URL from project name
            repo_url = f"https://github.com/{repo_name}"

            language = str(row.get("gh_lang", "")).lower() if pd.notna(row.get("gh_lang")) else ""
            test_count = int(row["tr_log_num_tests_ok"]) if pd.notna(row["tr_log_num_tests_ok"]) else 0

            samples.append({
                "repo_url": repo_url,
                "repo_name": repo_name,
                "language": language,
                "test_framework": str(row.get("tr_log_frameworks", "")) if pd.notna(row.get("tr_log_frameworks")) else "",
                "test_count": test_count,
                "build_id": str(row.get("tr_build_id", "")) if pd.notna(row.get("tr_build_id")) else "",
                "commit_sha": str(row.get("git_trigger_commit", "")) if pd.notna(row.get("git_trigger_commit")) else "",
            })

            if len(samples) >= limit:
                break

        if len(samples) >= limit:
            break

    print(f"Loaded {len(samples)} repos with passing tests")
    return samples


def clone_and_extract_tests(repo_url: str, commit_sha: str = None, max_files: int = 5) -> Dict:
    """
    Clone a repository and extract test files.

    Args:
        repo_url: GitHub repository URL
        commit_sha: Optional commit to checkout
        max_files: Maximum number of test files to extract

    Returns:
        Dict with test files and metadata
    """
    result = {
        "test_files": {},
        "setup_files": {},
        "error": None,
    }

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone repository (shallow)
            clone_cmd = ["git", "clone", "--depth", "1"]
            if commit_sha:
                clone_cmd.extend(["--branch", commit_sha])
            clone_cmd.extend([repo_url, tmpdir])

            subprocess.run(clone_cmd, capture_output=True, timeout=60)

            repo_path = Path(tmpdir)

            # Find test files
            test_patterns = ["**/test_*.py", "**/tests/*.py", "**/*_test.py",
                            "**/test/*.py", "**/*Test.java", "**/tests/*.js"]

            test_files = []
            for pattern in test_patterns:
                test_files.extend(repo_path.glob(pattern))

            # Extract test file contents
            for tf in test_files[:max_files]:
                try:
                    content = tf.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 100:  # Skip tiny files
                        result["test_files"][tf.name] = content
                except:
                    pass

            # Look for setup files
            setup_files = ["requirements.txt", "setup.py", "package.json", "pom.xml"]
            for sf in setup_files:
                sf_path = repo_path / sf
                if sf_path.exists():
                    try:
                        result["setup_files"][sf] = sf_path.read_text(encoding="utf-8", errors="ignore")
                    except:
                        pass

    except Exception as e:
        result["error"] = str(e)

    return result


def process_repos_for_tasks(samples: List[Dict], max_per_repo: int = 3, target_tasks: int = 10000) -> List[Dict]:
    """
    Process repositories to extract test files and create task samples.

    Args:
        samples: List of repo samples from TravisTorrent
        max_per_repo: Maximum test files to extract per repo
        target_tasks: Target number of tasks to generate
    """
    print(f"Extracting test files from repositories (target: {target_tasks} tasks)...")
    task_samples = []
    repos_processed = 0
    repos_failed = 0

    for sample in tqdm(samples, desc="Processing repos"):
        if len(task_samples) >= target_tasks:
            break

        if sample.get("language") not in ["python", "java", "javascript"]:
            continue

        extracted = clone_and_extract_tests(
            sample["repo_url"],
            sample.get("commit_sha"),
            max_files=max_per_repo
        )

        repos_processed += 1

        if extracted.get("error") or not extracted.get("test_files"):
            repos_failed += 1
            continue

        for filename, content in extracted["test_files"].items():
            task_samples.append({
                "text": content,
                "filename": filename,
                "language": sample["language"],
                "repo": sample["repo_name"],
                "repo_url": sample["repo_url"],
                "test_framework": sample.get("test_framework", ""),
                "setup_files": extracted.get("setup_files", {}),
            })

            if len(task_samples) >= target_tasks:
                break

    print(f"Processed {repos_processed} repos ({repos_failed} failed)")
    print(f"Extracted {len(task_samples)} test files")
    return task_samples


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory from extracted test file."""
    language = sample.get("language", "python")

    if language == "python":
        dockerfile = get_dockerfile("python")
        test_sh = create_pytest_test_sh("/tests/test_solution.py")
        test_filename = "test_solution.py"
    elif language == "java":
        dockerfile = get_dockerfile("java")
        test_sh = create_generic_test_sh("mvn test -q")
        test_filename = "TestSolution.java"
    else:
        dockerfile = get_dockerfile("node")
        test_sh = create_generic_test_sh("npx jest /tests/test_solution.js")
        test_filename = "test_solution.js"

    # Process test content
    test_content = sample.get("text", "")
    if language == "python" and "sys.path" not in test_content:
        test_content = "import sys\nsys.path.insert(0, '/app')\n\n" + test_content

    test_files = {test_filename: test_content}

    metadata = {
        "source": "travistorrent",
        "repo": sample.get("repo", ""),
        "repo_url": sample.get("repo_url", ""),
        "original_filename": sample.get("filename", ""),
        "test_framework": sample.get("test_framework", ""),
    }

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=test_sh,
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "travistorrent",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating tasks"
    )):
        create_harbor_task(temp_dir, i, instruction, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT, skip_clone: bool = False) -> None:
    """Main pipeline for generating TravisTorrent tasks."""

    print("Step 1: Loading TravisTorrent dataset...")
    # Load more samples than needed since many repos may fail to clone or have no tests
    load_limit = limit * 10  # 10x buffer
    samples = load_travistorrent(limit=load_limit)
    print(f"  -> {len(samples)} repos with passing tests")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    if not skip_clone:
        print("\nStep 2: Extracting test files from repositories...")
        task_samples = process_repos_for_tasks(samples, target_tasks=limit)
    else:
        # Use repo metadata directly without cloning
        task_samples = [
            {
                "text": f"# Test file from {s['repo_name']}\n# Framework: {s.get('test_framework', 'unknown')}",
                "language": s["language"],
                "repo": s["repo_name"],
                "repo_url": s["repo_url"],
            }
            for s in samples if s.get("language") in ["python", "java", "javascript"]
        ]

    if not task_samples:
        print("\nNo task samples extracted. Exiting.")
        return

    print(f"  -> {len(task_samples)} test files extracted")

    print("\nStep 3: Generating task descriptions...")
    dataset = Dataset.from_list(task_samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": TEST_TO_INSTRUCTION_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    print("\nStep 4: Generating harbor task directories...")
    task_dir = generate_tasks(task_samples, instructions, "travistorrent")
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 5: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_travistorrent")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(task_samples)} TravisTorrent tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from TravisTorrent dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum repos to process")
    parser.add_argument("--skip-clone", action="store_true", help="Skip repository cloning")

    args = parser.parse_args()
    main(limit=args.limit, skip_clone=args.skip_clone)
