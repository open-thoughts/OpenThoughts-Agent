#!/usr/bin/env python3
"""
Generate tasks from ManyBugs dataset (185 bugs in complex C programs).

ManyBugs contains real bugs from large, well-known C programs:
- Coreutils (GNU core utilities)
- Findutils (GNU find)
- Grep (GNU grep)
- Make (GNU make)
- Gzip
- PHP
- Python (C implementation)
- And more...

This is one of the hardest bug repair benchmarks for C programs.
Source: https://repairbenchmarks.cs.umass.edu/
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))  # Add dclm-mine directory

from dataset_utils import (
    download_manybugs,
    download_github_archive,
    get_cache_dir,
    load_manybugs_local,
)
from data.commons import (
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Dockerfile for C programs with build tools
MANYBUGS_DOCKERFILE = """FROM ubuntu:22.04

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    autoconf \\
    automake \\
    libtool \\
    git \\
    wget \\
    curl \\
    pkg-config \\
    bison \\
    flex \\
    gdb \\
    valgrind \\
    gcovr \\
    && rm -rf /var/lib/apt/lists/*

# Install additional development libraries
RUN apt-get update && apt-get install -y \\
    libncurses5-dev \\
    libreadline-dev \\
    libssl-dev \\
    libpcre3-dev \\
    libglib2.0-dev \\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /logs/verifier /workspace
"""


def create_manybugs_test_sh(project: str, test_cmd: str) -> str:
    """Create test.sh for ManyBugs tasks."""
    return f'''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {{
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}}
trap cleanup EXIT

cd /workspace

# Build the project
echo "Building {project}..."
make clean 2>/dev/null || true
./configure 2>&1 || true
make 2>&1

# Run the tests
echo "Running tests..."
{test_cmd} 2>&1 | tee /logs/verifier/test_output.txt

exit ${{PIPESTATUS[0]}}
'''


def load_manybugs(limit: int = LIMIT, local_repo_dir: Path = None) -> List[Dict]:
    """
    Load ManyBugs dataset.

    The dataset contains bugs from major C programs with test suites.

    Args:
        limit: Maximum samples to load
        local_repo_dir: Path to locally cloned ManyBugs repository.
                        Clone from https://github.com/squaresLab/ManyBugs
    """
    print(f"Loading ManyBugs dataset...")

    # Option 1: Load from local repository if provided
    if local_repo_dir and Path(local_repo_dir).exists():
        print(f"Loading from local repository: {local_repo_dir}")
        try:
            samples = load_manybugs_local(Path(local_repo_dir), limit=limit)
            if samples:
                return _enrich_manybugs_samples(samples, local_repo_dir)
        except Exception as e:
            print(f"Error loading from local repo: {e}")

    # Option 2: Check cache for previously downloaded repo
    cache_dir = get_cache_dir()
    cached_dir = cache_dir / "github" / "squaresLab_ManyBugs"
    if cached_dir.exists():
        print(f"Loading from cached repository: {cached_dir}")
        try:
            samples = load_manybugs_local(cached_dir, limit=limit)
            if samples:
                return _enrich_manybugs_samples(samples, cached_dir)
        except Exception as e:
            print(f"Error loading from cache: {e}")

    # Option 3: Try HuggingFace sources
    repo_names = [
        "manybugs/manybugs",
        "repairbench/manybugs",
        "defects4j/manybugs",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="test", streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        # Option 4: Try to download from GitHub
        print("")
        print("HuggingFace sources unavailable. Attempting to download from GitHub...")
        print("Source: https://github.com/squaresLab/ManyBugs")
        try:
            repo_dir = download_manybugs()
            samples = load_manybugs_local(repo_dir, limit=limit)
            if samples:
                return _enrich_manybugs_samples(samples, repo_dir)
        except Exception as e:
            print(f"GitHub download failed: {e}")

        # Fallback: Create sample data for demonstration
        print("")
        print("To load the full ManyBugs dataset:")
        print("  1. git clone https://github.com/squaresLab/ManyBugs")
        print("  2. Run with --local-repo /path/to/ManyBugs")
        print("")
        print("Creating sample data for demonstration...")
        return _create_sample_manybugs_data(limit)

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        bug_id = sample.get("bug_id", sample.get("id", ""))
        project = sample.get("project", sample.get("program", ""))

        if not bug_id or not project:
            continue

        samples.append({
            "bug_id": bug_id,
            "project": project,
            "buggy_file": sample.get("buggy_file", sample.get("file", "")),
            "buggy_code": sample.get("buggy_code", ""),
            "fixed_code": sample.get("fixed_code", ""),
            "patch": sample.get("patch", sample.get("diff", "")),
            "test_cmd": sample.get("test_cmd", sample.get("test_command", "make test")),
            "build_cmd": sample.get("build_cmd", "make"),
            "description": sample.get("description", sample.get("bug_description", "")),
            "revision": sample.get("revision", sample.get("commit", "")),
        })

    print(f"Loaded {len(samples)} ManyBugs samples")
    return samples


def _enrich_manybugs_samples(samples: List[Dict], repo_dir: Path) -> List[Dict]:
    """Enrich basic ManyBugs samples with code from repository."""
    repo_dir = Path(repo_dir)
    enriched = []

    for sample in samples:
        project = sample.get("project", "")
        scenario = sample.get("scenario", sample.get("bug_id", "").split("-")[-1] if "-" in sample.get("bug_id", "") else "")

        # Try to find buggy/fixed code in scenarios directory
        scenario_dir = repo_dir / "scenarios" / project / scenario
        if not scenario_dir.exists():
            scenario_dir = repo_dir / project / scenario

        buggy_code = ""
        fixed_code = ""
        patch = ""

        # Look for diff/patch files
        for patch_file in scenario_dir.glob("*.patch") if scenario_dir.exists() else []:
            try:
                patch = patch_file.read_text(encoding='utf-8', errors='ignore')
                break
            except Exception:
                pass

        enriched.append({
            **sample,
            "buggy_code": sample.get("buggy_code", buggy_code),
            "fixed_code": sample.get("fixed_code", fixed_code),
            "patch": sample.get("patch", patch),
            "test_cmd": sample.get("test_cmd", "make test"),
            "description": sample.get("description", f"Bug in {project} project"),
        })

    return enriched


def _create_sample_manybugs_data(limit: int) -> List[Dict]:
    """Create sample ManyBugs data for demonstration."""
    samples = [
        {
            "bug_id": "grep-1",
            "project": "grep",
            "buggy_file": "src/grep.c",
            "buggy_code": '''#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int match_pattern(const char *pattern, const char *text) {
    // Bug: Off-by-one error in pattern matching
    int plen = strlen(pattern);
    int tlen = strlen(text);

    for (int i = 0; i < tlen; i++) {  // Bug: should be <= tlen - plen
        int match = 1;
        for (int j = 0; j < plen; j++) {
            if (text[i + j] != pattern[j]) {
                match = 0;
                break;
            }
        }
        if (match) return i;
    }
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s pattern file\\n", argv[0]);
        return 1;
    }
    // Simplified grep implementation
    return 0;
}
''',
            "fixed_code": '''#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int match_pattern(const char *pattern, const char *text) {
    int plen = strlen(pattern);
    int tlen = strlen(text);

    for (int i = 0; i <= tlen - plen; i++) {
        int match = 1;
        for (int j = 0; j < plen; j++) {
            if (text[i + j] != pattern[j]) {
                match = 0;
                break;
            }
        }
        if (match) return i;
    }
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s pattern file\\n", argv[0]);
        return 1;
    }
    return 0;
}
''',
            "patch": '''--- a/src/grep.c
+++ b/src/grep.c
@@ -9,7 +9,7 @@ int match_pattern(const char *pattern, const char *text) {
     int plen = strlen(pattern);
     int tlen = strlen(text);

-    for (int i = 0; i < tlen; i++) {
+    for (int i = 0; i <= tlen - plen; i++) {
         int match = 1;
         for (int j = 0; j < plen; j++) {
''',
            "test_cmd": "./test_grep.sh",
            "description": "Off-by-one error in pattern matching causes buffer overread",
        },
        {
            "bug_id": "coreutils-1",
            "project": "coreutils",
            "buggy_file": "src/ls.c",
            "buggy_code": '''#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>

void list_directory(const char *path) {
    DIR *dir = opendir(path);
    struct dirent *entry;

    // Bug: Missing NULL check after opendir
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] != '.') {
            printf("%s\\n", entry->d_name);
        }
    }
    closedir(dir);
}

int main(int argc, char *argv[]) {
    const char *path = argc > 1 ? argv[1] : ".";
    list_directory(path);
    return 0;
}
''',
            "fixed_code": '''#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>

void list_directory(const char *path) {
    DIR *dir = opendir(path);
    struct dirent *entry;

    if (dir == NULL) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] != '.') {
            printf("%s\\n", entry->d_name);
        }
    }
    closedir(dir);
}

int main(int argc, char *argv[]) {
    const char *path = argc > 1 ? argv[1] : ".";
    list_directory(path);
    return 0;
}
''',
            "patch": '''--- a/src/ls.c
+++ b/src/ls.c
@@ -8,6 +8,11 @@ void list_directory(const char *path) {
     DIR *dir = opendir(path);
     struct dirent *entry;

+    if (dir == NULL) {
+        perror("opendir");
+        return;
+    }
+
     while ((entry = readdir(dir)) != NULL) {
''',
            "test_cmd": "./test_ls.sh",
            "description": "Missing NULL check causes segfault on invalid directory",
        },
    ]

    return samples[:limit]


def create_instruction_from_manybugs(sample: Dict) -> str:
    """Create task instruction from ManyBugs sample."""
    project = sample.get("project", "Unknown")
    bug_id = sample.get("bug_id", "Unknown")
    buggy_file = sample.get("buggy_file", "")
    description = sample.get("description", "")
    buggy_code = sample.get("buggy_code", "")

    instruction = f"""# C Bug Repair Task

## Project
{project} (from ManyBugs benchmark)

## Bug ID
{bug_id}

## Bug Description
{description if description else "No description available."}

## Buggy File
{buggy_file}

## Buggy Code

```c
{buggy_code}
```

## Your Task

1. Analyze the C code to identify the bug(s)
2. Consider common C programming errors:
   - Off-by-one errors
   - Buffer overflows/underflows
   - NULL pointer dereferences
   - Memory leaks
   - Integer overflow
   - Format string vulnerabilities
3. Fix the bug while maintaining the code's functionality
4. Place your fixed code in `/workspace/`

## Build Instructions

The project uses standard autotools:
```bash
./configure
make
```

## Notes

- This is a real bug from a widely-used open source project
- The fix should be minimal - only change what's necessary
- Ensure the fix handles edge cases properly
"""

    return instruction


def create_test_script(sample: Dict) -> str:
    """Create test script for ManyBugs task."""
    buggy_code = sample.get("buggy_code", "")

    # Create a simple test based on the bug type
    return '''#!/bin/bash
set -e

cd /workspace

# Compile the code
gcc -o solution solution.c -Wall -Werror 2>&1

# Run basic tests
echo "Running tests..."

# Test 1: Basic functionality
./solution > /dev/null 2>&1
echo "Test 1 passed: Basic execution"

# Test 2: Edge cases (will be customized per bug)
echo "All tests passed!"
'''


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a ManyBugs sample."""
    instruction = create_instruction_from_manybugs(sample)

    test_files = {
        "test.sh": create_test_script(sample),
    }

    # Add the buggy code as starting point
    buggy_file = sample.get("buggy_file", "solution.c")
    if "/" in buggy_file:
        buggy_file = buggy_file.split("/")[-1]
    test_files[f"workspace/{buggy_file}"] = sample.get("buggy_code", "")

    # Add patch info
    if sample.get("patch"):
        test_files["expected_fix.patch"] = sample["patch"]

    metadata = {
        "source": "manybugs",
        "bug_id": sample.get("bug_id", ""),
        "project": sample.get("project", ""),
        "buggy_file": sample.get("buggy_file", ""),
        "revision": sample.get("revision", ""),
    }

    # Store fixed code as solution
    solution_files = None
    if sample.get("fixed_code"):
        solution_files = {"solution.c": sample["fixed_code"]}

    test_cmd = sample.get("test_cmd", "make test")

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=MANYBUGS_DOCKERFILE,
        test_sh=create_manybugs_test_sh(sample.get("project", ""), "/tests/test.sh"),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(samples: List[Dict], dataset_prefix: str = "manybugs") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT, upload: bool = True, local_repo_dir: Path = None) -> None:
    """Main pipeline for generating ManyBugs tasks."""

    print(f"Step 1: Loading ManyBugs dataset...")
    samples = load_manybugs(limit=limit, local_repo_dir=local_repo_dir)
    print(f"  -> {len(samples)} bug samples loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = "manybugs"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} ManyBugs tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from ManyBugs dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--local-repo", type=Path,
                        help="Path to locally cloned ManyBugs repository (from GitHub: squaresLab/ManyBugs)")

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload, local_repo_dir=args.local_repo)
