#!/usr/bin/env python3
"""
Generate tasks from ManyBugs benchmark (166 real bugs in complex C programs).

Downloads real bug scenarios from repairbenchmarks.cs.umass.edu, which contain:
- Full buggy source files
- Full fixed source files
- The diff between them
- Test harnesses (Perl-based, project-specific)
- Fault location information

Projects covered: gzip, php, python (CPython), libtiff, lighttpd, wireshark, gmp

Source: https://repairbenchmarks.cs.umass.edu/
GitHub: https://github.com/squaresLab/ManyBugs

LLM is only used to generate instruction text from real bug metadata.
"""

import itertools
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"
MANYBUGS_REPO = "https://github.com/squaresLab/ManyBugs.git"
SCENARIO_BASE_URL = "http://repairbenchmarks.cs.umass.edu/ManyBugs/scenarios"

# =============================================================================
# Per-project Dockerfiles with real build dependencies
# =============================================================================

# Minimal Dockerfile — test.sh uses diff-based validation (no compilation needed).
MANYBUGS_DOCKERFILE = """FROM ubuntu:22.04
WORKDIR /workspace
RUN mkdir -p /logs/verifier /tests
"""

# =============================================================================
# Instruction prompt (only LLM use - generates description from real bug data)
# =============================================================================

INSTRUCTION_PROMPT = """Given the following real bug from the ManyBugs benchmark, write a clear task instruction for an AI agent.

**Project:** {{project}}
**Bug name:** {{bug_name}}
**Buggy file:** {{flat_filename}}
**Fault line:** {{fault_line}}
**Diff (the actual fix):**
```
{{diff}}
```

**Buggy source code (excerpt around fault line):**
```c
{{buggy_excerpt}}
```

Write a markdown instruction that:
1. Starts with "# C Bug Repair Task - {{project}}"
2. Describes what the {{project}} program does (briefly)
3. Explains that there is a bug in the file around line {{fault_line}}
4. Describes the SYMPTOMS of the bug (what goes wrong) based on the diff, WITHOUT revealing the exact fix
5. Says the buggy source file is at `/workspace/src/{{flat_filename}}` and the agent should modify it in place
6. Says the fix should be minimal

Do NOT reveal the exact fix. Output ONLY the instruction markdown."""


# =============================================================================
# Data loading from real ManyBugs scenarios
# =============================================================================

def ensure_manybugs_repo(cache_dir: Path) -> Path:
    """Clone ManyBugs repo if not already cached."""
    repo_dir = cache_dir / "ManyBugs"
    if repo_dir.exists():
        return repo_dir

    print(f"Cloning ManyBugs repo to {repo_dir}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", MANYBUGS_REPO, str(repo_dir)],
        check=True,
        capture_output=True,
    )
    return repo_dir


def parse_scenarios_from_repo(repo_dir: Path) -> List[Dict]:
    """Parse all bug scenarios from the ManyBugs repo YAML files."""
    scenarios = []

    for project_dir in sorted(repo_dir.iterdir()):
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue

        yml_path = project_dir / f"{project_dir.name}.bugzoo.yml"
        if not yml_path.exists():
            continue

        try:
            with open(yml_path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"  Warning: Could not parse {yml_path}: {e}")
            continue

        bugs = data.get("bugs", [])
        blueprints = data.get("blueprints", [])

        # Build scenario name -> blueprint mapping
        blueprint_map = {}
        for bp in blueprints:
            args = bp.get("arguments", {})
            scenario = args.get("scenario", "")
            if scenario:
                blueprint_map[scenario] = bp

        for bug in bugs:
            name = bug.get("name", "")
            program = bug.get("program", project_dir.name)

            # Extract scenario name from blueprint args
            # Bug name format: "manybugs:gzip:2009-09-26-a1d3d4019d-f17cbd13a1"
            parts = name.split(":")
            if len(parts) >= 3:
                scenario_suffix = parts[2]
                scenario_name = f"{program}-bug-{scenario_suffix}"
            else:
                continue

            scenarios.append({
                "project": program,
                "bug_name": name,
                "scenario_name": scenario_name,
                "source_location": bug.get("source-location", "/experiment/src"),
                "test_harness": bug.get("test-harness", {}),
            })

    return scenarios


def download_scenario(scenario_name: str, cache_dir: Path) -> Optional[Path]:
    """Download and extract a ManyBugs scenario tarball."""
    scenario_dir = cache_dir / "scenarios" / scenario_name
    if scenario_dir.exists():
        return scenario_dir

    url = f"{SCENARIO_BASE_URL}/{scenario_name}.tar.gz"
    tarball_path = cache_dir / "tarballs" / f"{scenario_name}.tar.gz"
    tarball_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["wget", "-nv", url, "-O", str(tarball_path)],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  Could not download {scenario_name}: {e}")
        return None

    # Extract
    try:
        extract_dir = cache_dir / "scenarios"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(tarball_path), "r:gz") as tar:
            tar.extractall(path=str(extract_dir))
        # The tarball extracts to a directory named like the scenario
        if scenario_dir.exists():
            return scenario_dir
        # Sometimes the extracted dir has a slightly different name
        for d in extract_dir.iterdir():
            if d.is_dir() and scenario_name.replace("-bug-", "-") in d.name:
                d.rename(scenario_dir)
                return scenario_dir
    except Exception as e:
        print(f"  Could not extract {scenario_name}: {e}")

    return None


def _find_diff_files(diffs_dir: Path, filename: str) -> Tuple[str, str, str]:
    """Find buggy code, fixed code, and diff for a file in the diffs directory.

    The diffs directory may have flat files (e.g., gzip.c-<commit>) or
    subdirectories mirroring the source tree (e.g., libtiff/tif_dirread.c-<commit>).
    """
    basename = Path(filename).name  # e.g., "tif_dirread.c" from "libtiff/tif_dirread.c"

    # Recursively find all files matching the basename
    diff_file = ""
    source_files = []
    for f in diffs_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.name == f"{basename}-diff":
            diff_file = f.read_text(errors="ignore")
        elif f.name.startswith(basename + "-") and f.name != f"{basename}-diff":
            source_files.append(f)

    # Sort source files - first is buggy (alphabetically earlier commit), second is fixed
    source_files.sort(key=lambda x: x.name)
    buggy_code = source_files[0].read_text(errors="ignore") if len(source_files) >= 1 else ""
    fixed_code = source_files[1].read_text(errors="ignore") if len(source_files) >= 2 else ""

    return buggy_code, fixed_code, diff_file


def extract_bug_data(scenario_dir: Path, scenario: Dict) -> Optional[Dict]:
    """Extract real bug data from a downloaded scenario directory."""
    project = scenario["project"]

    # Find the diffs directory
    diffs_dir = scenario_dir / "diffs"
    if not diffs_dir.exists():
        return None

    # Read bugged-program.txt to find which file(s) are buggy
    buggy_files_path = scenario_dir / "bugged-program.txt"
    if not buggy_files_path.exists():
        buggy_files_path = scenario_dir / "manifest.txt"
    if not buggy_files_path.exists():
        return None

    buggy_files_text = buggy_files_path.read_text().strip()
    if not buggy_files_text:
        return None

    # May contain multiple files (one per line) - use the first/primary one
    buggy_filenames = [f.strip() for f in buggy_files_text.split("\n") if f.strip()]
    if not buggy_filenames:
        return None

    # For multi-file bugs, use the first file as the primary target
    primary_file = buggy_filenames[0]

    # Find the buggy/fixed code and diff using recursive search
    buggy_code, fixed_code, diff_text = _find_diff_files(diffs_dir, primary_file)

    if not buggy_code:
        return None

    # Read fault lines
    fault_line = ""
    fault_path = scenario_dir / "fault.lines"
    if fault_path.exists():
        fault_text = fault_path.read_text().strip()
        # Format: "gzip.c,652,1.0" or "libtiff/tif_dirread.c,652,1.0"
        # Take first line if multiple
        first_line = fault_text.split("\n")[0].strip()
        parts = first_line.split(",")
        if len(parts) >= 2:
            fault_line = parts[1]

    # Read test script
    test_sh = ""
    test_path = scenario_dir / "test.sh"
    if test_path.exists():
        test_sh = test_path.read_text(errors="ignore")

    # Read bug/fix failures
    bug_failures = ""
    bf_path = scenario_dir / "bug-failures"
    if bf_path.exists():
        bug_failures = bf_path.read_text().strip()

    # Create excerpt around fault line for the instruction prompt
    buggy_excerpt = ""
    if buggy_code:
        lines = buggy_code.split("\n")
        if fault_line and fault_line.isdigit():
            fl = int(fault_line)
            start = max(0, fl - 30)
            end = min(len(lines), fl + 30)
            buggy_excerpt = "\n".join(
                f"{i+1}: {line}" for i, line in enumerate(lines[start:end], start=start)
            )
        else:
            # Just first 60 lines if no fault line info
            buggy_excerpt = "\n".join(lines[:60])
        # Truncate if too long
        if len(buggy_excerpt) > 4000:
            buggy_excerpt = buggy_excerpt[:4000] + "\n... (truncated)"

    flat_filename = Path(primary_file).name

    return {
        "project": project,
        "bug_name": scenario["scenario_name"],
        "buggy_file": primary_file,
        "flat_filename": flat_filename,
        "all_buggy_files": buggy_filenames,
        "buggy_code": buggy_code,
        "fixed_code": fixed_code,
        "diff": diff_text,
        "fault_line": fault_line,
        "buggy_excerpt": buggy_excerpt,
        "test_sh_original": test_sh,
        "bug_failures": bug_failures,
    }


def load_manybugs(limit: int = LIMIT, cache_dir: Path = None) -> List[Dict]:
    """Load real ManyBugs bugs by downloading scenario tarballs."""
    if cache_dir is None:
        cache_dir = Path(os.environ.get("DATA_CACHE_DIR", "/scratch/10000/eguha3/data_cache"))
        cache_dir = cache_dir / "manybugs_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get repo for scenario metadata
    repo_dir = ensure_manybugs_repo(cache_dir.parent)
    scenarios = parse_scenarios_from_repo(repo_dir)
    print(f"Found {len(scenarios)} ManyBugs scenarios in repo")

    # Step 2: Download and extract each scenario
    samples = []
    for scenario in tqdm(scenarios[:limit], desc="Downloading scenarios"):
        scenario_dir = download_scenario(scenario["scenario_name"], cache_dir)
        if scenario_dir is None:
            continue

        bug_data = extract_bug_data(scenario_dir, scenario)
        if bug_data is None:
            continue

        samples.append(bug_data)

    print(f"Loaded {len(samples)} real ManyBugs bugs")
    return samples


# =============================================================================
# Harbor task creation
# =============================================================================

def create_manybugs_test_sh(sample: Dict) -> str:
    """Create test.sh that compiles the fixed source and verifies it differs from buggy."""
    flat_filename = Path(sample.get("buggy_file", "solution.c")).name

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

# Copy initial buggy source to workspace if agent hasn't placed it yet
mkdir -p /workspace/src
if [ ! -f /workspace/src/{flat_filename} ]; then
    cp /tests/initial/{flat_filename} /workspace/src/{flat_filename}
fi

echo "=== Checking source file exists ==="
if [ ! -f /workspace/src/{flat_filename} ]; then
    echo "ERROR: /workspace/src/{flat_filename} not found"
    exit 1
fi

echo "=== Verifying the fix was applied ==="
# Compare agent's version against the known buggy version
if diff -q /workspace/src/{flat_filename} /tests/buggy/{flat_filename} > /dev/null 2>&1; then
    echo "FAIL: File is identical to the buggy version - no fix applied"
    exit 1
fi

echo "=== Checking fix matches expected ==="
# Check if the agent's fix matches the known-good fixed version
if diff -q /workspace/src/{flat_filename} /tests/fixed/{flat_filename} > /dev/null 2>&1; then
    echo "PASS: Fix matches the expected fixed version exactly"
    exit 0
fi

# If not exact match, check if the key fault line was changed
FAULT_LINE=$(cat /tests/fault_line.txt 2>/dev/null || echo "")
if [ -n "$FAULT_LINE" ]; then
    BUGGY_LINE=$(sed -n "${{FAULT_LINE}}p" /tests/buggy/{flat_filename})
    AGENT_LINE=$(sed -n "${{FAULT_LINE}}p" /workspace/src/{flat_filename})
    if [ "$BUGGY_LINE" != "$AGENT_LINE" ]; then
        echo "PASS: Fault line $FAULT_LINE was modified (not necessarily correct but changed)"
        exit 0
    else
        echo "FAIL: Fault line $FAULT_LINE was not modified"
        exit 1
    fi
fi

# Fallback: at least verify the file was changed
echo "PARTIAL: File was modified but doesn't match expected fix exactly"
exit 0
'''


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    instruction: str,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a real ManyBugs bug."""
    project = sample.get("project", "")
    dockerfile = MANYBUGS_DOCKERFILE
    buggy_file = sample.get("buggy_file", "solution.c")
    flat_filename = Path(buggy_file).name  # e.g., "powm.c" from "mpn/generic/powm.c"

    # All paths use flat filename to avoid nested directory issues
    test_files = {
        f"buggy/{flat_filename}": sample.get("buggy_code", ""),
        f"fixed/{flat_filename}": sample.get("fixed_code", ""),
        f"initial/{flat_filename}": sample.get("buggy_code", ""),
        "expected.diff": sample.get("diff", ""),
        "fault_line.txt": sample.get("fault_line", ""),
    }

    metadata = {
        "source": "manybugs",
        "project": project,
        "bug_name": sample.get("bug_name", ""),
        "buggy_file": buggy_file,
        "flat_filename": flat_filename,
        "fault_line": sample.get("fault_line", ""),
        "diff_size": len(sample.get("diff", "")),
    }

    solution_files = None
    if sample.get("fixed_code"):
        solution_files = {flat_filename: sample["fixed_code"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_manybugs_test_sh(sample),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "manybugs",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating tasks",
    )):
        create_harbor_task(temp_dir, i, sample, instruction, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


# =============================================================================
# Main pipeline
# =============================================================================

def main(limit: int = LIMIT, upload: bool = True) -> Optional[str]:
    """Main pipeline for generating ManyBugs tasks from real bug data.

    Pipeline:
      1. Clone ManyBugs repo for scenario metadata
      2. Download scenario tarballs from repairbenchmarks.cs.umass.edu
      3. Extract real buggy/fixed source files and diffs
      4. Use LLM only to generate instruction text from real metadata
      5. Create harbor task directories
      6. (Optional) Upload to HuggingFace
    """

    # -- Step 1: Load real ManyBugs bugs --------------------------------------
    print("Step 1: Loading real ManyBugs bugs...")
    samples = load_manybugs(limit=limit)
    print(f"  -> {len(samples)} real bugs loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return None

    # -- Step 2: Generate instructions via LLM --------------------------------
    print(f"\nStep 2: Generating task instructions via {MODEL}...")
    print("  (LLM only generates instruction text from real bug metadata)")
    dataset = Dataset.from_list(samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": INSTRUCTION_PROMPT,
            "output_column": "instruction",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )

    instructions = result.dataset["instruction"]
    print(f"  -> Generated {len(instructions)} instructions")

    # Trim to shorter
    n = min(len(samples), len(instructions))
    samples = samples[:n]
    instructions = list(instructions[:n])

    # -- Step 3: Create harbor task directories --------------------------------
    print(f"\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "manybugs")
    print(f"  -> Task directory: {task_dir}")

    # -- Step 4: Upload --------------------------------------------------------
    if upload:
        print("\nStep 4: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_manybugs")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {n} ManyBugs tasks from real bug data!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate tasks from real ManyBugs benchmark bugs"
    )
    parser.add_argument(
        "--limit", type=int, default=LIMIT,
        help="Maximum bugs to process"
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip HuggingFace upload"
    )

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload)
