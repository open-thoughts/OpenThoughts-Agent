#!/usr/bin/env python3
"""
Patch R2E-Gym tasks to use a generic python:3.11-bookworm base image instead of
per-task pre-built docker images.

R2E-Gym tasks currently use one unique docker image per task (~4,578 total),
each containing the repo code, a virtualenv, and custom test infrastructure
(r2e_tests/, run_tests.sh). This overwhelms Daytona's snapshot system.

After patching, all tasks share a single Dockerfile → 1 snapshot.
Test infrastructure moves to setup_files/, which Harbor uploads to /setup_files/
in the container before the agent runs (requires penfever/temp-override branch).

Two-phase workflow
------------------
Phase 1 — extract test infrastructure from docker images (run once, requires docker):

    python patch_r2egym_tasks.py /path/to/tasks --extract-tests [--parallel 8]

    Pulls each unique docker image, extracts /r2e_tests/ and /root/run_tests.sh,
    and writes them into setup_files/ for each task.

Phase 2 — apply the patch (idempotent, can be re-run):

    python patch_r2egym_tasks.py /path/to/tasks --output-dir /path/to/patched

    Rewrites Dockerfile, instruction.md, test.sh, and moves metadata to setup_files/.
    If --extract-tests has not been run, tasks missing setup_files/r2e_tests.tar.gz
    are skipped with a warning.

Changes per task
----------------
  1. environment/Dockerfile      — Generic python:3.11-bookworm; no COPY, no repo-specific layers
  2. setup_files/metadata.json   — Moved from environment/workspace/metadata.json
  3. setup_files/r2e_tests.tar.gz — r2e_tests/ extracted from original docker image
  4. setup_files/run_tests.sh    — run_tests.sh extracted from original docker image
  5. instruction.md              — Setup preamble prepended (git clone, pip install)
  6. tests/test.sh               — Updated to use setup_files/ infrastructure
  7. tests/calculate_reward.py   — Updated path: /workspace → /setup_files

Usage
-----
    # Dry run (no writes):
    python patch_r2egym_tasks.py /path/to/tasks --dry-run

    # Write patched tasks to a new directory (leaves originals untouched):
    python patch_r2egym_tasks.py /path/to/tasks --output-dir /path/to/patched

    # Patch in-place:
    python patch_r2egym_tasks.py /path/to/tasks
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

DOCKERFILE_TEMPLATE = """\
FROM python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \\
    git curl wget jq \\
    build-essential \\
    libffi-dev libssl-dev \\
    locales locales-all tzdata \\
    tmux \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv
RUN mkdir -p /logs /testbed /r2e_tests && chmod 777 /testbed
WORKDIR /testbed
"""

# test.sh is kept structurally identical to the original r2egym test.sh;
# the only changes are:
#   - extract r2e_tests + run_tests.sh from /setup_files/ (uploaded by Harbor)
#     instead of relying on the pre-built image
#   - read metadata from /setup_files/metadata.json instead of /workspace/metadata.json
TEST_SH_TEMPLATE = """\
#!/bin/bash
set -e

# Setup PATH (matches original r2egym convention)
source ~/.bashrc 2>/dev/null || true
export PATH=/testbed/.venv/bin:$PATH

# ---------------------------------------------------------------------------
# Restore test infrastructure from setup_files/ (uploaded by Harbor before
# the agent runs — requires penfever/temp-override Harbor branch).
# ---------------------------------------------------------------------------
if [ -f /setup_files/r2e_tests.tar.gz ]; then
    tar -xzf /setup_files/r2e_tests.tar.gz -C /
else
    echo "Warning: /setup_files/r2e_tests.tar.gz not found; tests may fail."
fi

if [ -f /setup_files/run_tests.sh ]; then
    cp /setup_files/run_tests.sh /root/run_tests.sh
    chmod +x /root/run_tests.sh
else
    echo "Warning: /setup_files/run_tests.sh not found."
fi

# Clean up compiled bytecode to avoid stale cache issues
find /testbed -name '*.pyc' -delete 2>/dev/null || true
find /testbed -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true
find /r2e_tests -name '*.pyc'  -delete 2>/dev/null || true
find /r2e_tests -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true

# Move test files out of the agent's working directory
REPO_PATH="/testbed"
ALT_PATH="/root"
for skip_file in "run_tests.sh" "r2e_tests"; do
    if [ -e "$REPO_PATH/$skip_file" ]; then
        mv "$REPO_PATH/$skip_file" "$ALT_PATH/$skip_file"
    fi
done
if [ -d /r2e_tests ]; then
    mv /r2e_tests "$ALT_PATH/r2e_tests"
fi
if [ -d "$ALT_PATH/r2e_tests" ]; then
    ln -sf "$ALT_PATH/r2e_tests" "$REPO_PATH/r2e_tests"
fi

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
mkdir -p /logs/verifier

TEST_OUTPUT_FILE="/tmp/test_output.txt"
if [ -f /root/run_tests.sh ]; then
    bash /root/run_tests.sh 2>&1 | tee "$TEST_OUTPUT_FILE"
    TEST_EXIT_CODE=${{PIPESTATUS[0]}}
else
    echo "Error: /root/run_tests.sh not found after extraction."
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

# ---------------------------------------------------------------------------
# Calculate reward
# ---------------------------------------------------------------------------
mkdir -p /tests
cat > /tests/calculate_reward.py << 'REWARD_SCRIPT_EOF'
{calculate_reward_py}
REWARD_SCRIPT_EOF

chmod +x /tests/calculate_reward.py
python3 /tests/calculate_reward.py "$TEST_OUTPUT_FILE"

if [ ! -f /logs/verifier/reward.txt ]; then
    echo "Error: reward.txt was not created"
    echo "0" > /logs/verifier/reward.txt
fi
"""

# calculate_reward.py updated to read metadata from /setup_files/metadata.json
CALCULATE_REWARD_PY = """\
#!/usr/bin/env python3
import re
import json
import sys
from pathlib import Path


def parse_log_pytest(log):
    if log is None:
        return {}
    test_status_map = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[1].strip()
    for line in log.split("\\n"):
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_status_map[test_name.split(" - ")[0]] = "ERROR"
    return test_status_map


def decolor_dict_keys(d):
    decolor = lambda k: re.sub(r"\\u001b\\[\\d+m", "", k)
    return {decolor(k): v for k, v in d.items()}


def get_reward(parse, expected_output_json):
    parse = decolor_dict_keys(parse)
    expected = json.loads(expected_output_json)
    expected = decolor_dict_keys(expected)
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}
    if len(parse) != len(expected):
        return 0.0
    return 1.0 if all(
        k in expected and parse[k] == expected[k]
        for k in parse.keys() if k
    ) else 0.0


if __name__ == "__main__":
    test_output = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()

    # Metadata lives in /setup_files/ after patching (was /workspace/)
    for candidate in ["/setup_files/metadata.json", "/workspace/metadata.json"]:
        meta_path = Path(candidate)
        if meta_path.exists():
            break
    else:
        print("Error: metadata.json not found", file=sys.stderr)
        Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
        Path("/logs/verifier/reward.txt").write_text("0")
        sys.exit(1)

    metadata = json.loads(meta_path.read_text())
    expected_output_json = metadata.get("expected_output_json", "{}")
    parsed = parse_log_pytest(test_output)
    reward = get_reward(parsed, expected_output_json)

    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.txt").write_text(str(reward))
    print(f"Reward: {reward}")
"""

SETUP_PREAMBLE_TEMPLATE = """\
## Environment Setup (complete these steps first)

```bash
cd /testbed
git clone https://github.com/{repo}.git . && git checkout {base_commit}
pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true
```

---

"""

SOLVE_SH_SETUP_TEMPLATE = """\
# --- Environment setup (for generic base images) ---
cd /testbed
git clone https://github.com/{repo}.git . && git checkout {base_commit}
pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || true
# --- End environment setup ---

"""


# ---------------------------------------------------------------------------
# Docker extraction helpers
# ---------------------------------------------------------------------------

def _extract_from_image(
    docker_image: str,
    task_dir: Path,
    dry_run: bool = False,
) -> dict[str, bool]:
    """
    Pull docker image and extract r2e_tests/ + run_tests.sh into setup_files/.

    Returns a dict of what was extracted.
    """
    results: dict[str, bool] = {}
    setup_files_dir = task_dir / "setup_files"

    r2e_tar = setup_files_dir / "r2e_tests.tar.gz"
    run_tests_dst = setup_files_dir / "run_tests.sh"

    if r2e_tar.exists() and run_tests_dst.exists():
        # Already extracted
        results["r2e_tests.tar.gz"] = False
        results["run_tests.sh"] = False
        return results

    if dry_run:
        results["r2e_tests.tar.gz"] = True
        results["run_tests.sh"] = True
        return results

    setup_files_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Pull image
        subprocess.run(
            ["docker", "pull", docker_image],
            check=True, capture_output=True, text=True, timeout=300,
        )

        # Create a (non-running) container to copy from
        create_result = subprocess.run(
            ["docker", "create", docker_image],
            check=True, capture_output=True, text=True, timeout=30,
        )
        container_id = create_result.stdout.strip()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)

                # Extract r2e_tests/ — try both common locations
                r2e_extracted = False
                for src_path in ["/r2e_tests", "/testbed/r2e_tests"]:
                    cp_result = subprocess.run(
                        ["docker", "cp", f"{container_id}:{src_path}", str(tmp / "r2e_tests")],
                        capture_output=True, text=True, timeout=60,
                    )
                    if cp_result.returncode == 0:
                        # Tar up the extracted directory
                        with tarfile.open(r2e_tar, "w:gz") as tf:
                            tf.add(tmp / "r2e_tests", arcname="r2e_tests")
                        results["r2e_tests.tar.gz"] = True
                        r2e_extracted = True
                        break

                if not r2e_extracted:
                    results["r2e_tests.tar.gz"] = False

                # Extract run_tests.sh — try both common locations
                run_extracted = False
                for src_path in ["/root/run_tests.sh", "/testbed/run_tests.sh"]:
                    cp_result = subprocess.run(
                        ["docker", "cp", f"{container_id}:{src_path}", str(tmp / "run_tests.sh")],
                        capture_output=True, text=True, timeout=30,
                    )
                    if cp_result.returncode == 0:
                        shutil.copy(tmp / "run_tests.sh", run_tests_dst)
                        results["run_tests.sh"] = True
                        run_extracted = True
                        break

                if not run_extracted:
                    results["run_tests.sh"] = False

        finally:
            subprocess.run(
                ["docker", "rm", container_id],
                capture_output=True, text=True, timeout=30,
            )

    except subprocess.CalledProcessError as e:
        results["error"] = True
        results["error_msg"] = str(e)

    return results


# ---------------------------------------------------------------------------
# Patching logic
# ---------------------------------------------------------------------------

def _read_metadata(task_dir: Path) -> dict | None:
    """Read task metadata from workspace/metadata.json."""
    for candidate in [
        task_dir / "environment" / "workspace" / "metadata.json",
        task_dir / "setup_files" / "metadata.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def patch_task(
    task_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    require_tests: bool = True,
) -> dict[str, bool | str]:
    """
    Patch a single task directory.

    Returns a dict mapping filenames to True (changed), False (skipped/unchanged),
    or {"error": True, "reason": str} on failure.
    """
    changes: dict[str, bool | str] = {}

    metadata = _read_metadata(task_dir)
    if metadata is None:
        return {"error": True, "reason": "no metadata.json"}

    repo_name = metadata.get("repo_name", "")
    base_commit = metadata.get("base_commit", "HEAD")

    # Determine target directory
    if output_dir:
        target = output_dir / task_dir.name
        if not dry_run:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(task_dir, target)
    else:
        target = task_dir

    # Check that test infrastructure has been extracted (unless skipping check)
    r2e_tar = target / "setup_files" / "r2e_tests.tar.gz"
    run_tests_sh = target / "setup_files" / "run_tests.sh"
    tests_ready = r2e_tar.exists() and run_tests_sh.exists()

    if require_tests and not tests_ready:
        return {
            "error": True,
            "reason": (
                "setup_files/r2e_tests.tar.gz or run_tests.sh missing — "
                "run with --extract-tests first"
            ),
        }

    # --- 1. Dockerfile (fully generic, no COPY) ---
    dockerfile_path = target / "environment" / "Dockerfile"
    if dry_run:
        changes["Dockerfile"] = True
    else:
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(DOCKERFILE_TEMPLATE)
        changes["Dockerfile"] = True

    # --- 2. setup_files/metadata.json (move from environment/workspace/) ---
    original_workspace = target / "environment" / "workspace" / "metadata.json"
    setup_meta = target / "setup_files" / "metadata.json"
    if original_workspace.exists() and not setup_meta.exists():
        if dry_run:
            changes["setup_files/metadata.json"] = True
        else:
            setup_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(original_workspace, setup_meta)
            changes["setup_files/metadata.json"] = True
    else:
        changes["setup_files/metadata.json"] = False

    # --- 3. instruction.md (prepend setup preamble) ---
    instruction_path = target / "instruction.md"
    already_patched_marker = "## Environment Setup (complete these steps first)"
    if instruction_path.exists():
        original_text = instruction_path.read_text()
        if already_patched_marker in original_text:
            changes["instruction.md"] = False
        elif dry_run:
            changes["instruction.md"] = True
        else:
            preamble = SETUP_PREAMBLE_TEMPLATE.format(
                repo=repo_name,
                base_commit=base_commit,
            )
            instruction_path.write_text(preamble + original_text)
            changes["instruction.md"] = True
    else:
        changes["instruction.md"] = False

    # --- 4. tests/test.sh ---
    test_sh_path = target / "tests" / "test.sh"
    new_test_sh = TEST_SH_TEMPLATE.format(calculate_reward_py=CALCULATE_REWARD_PY)
    if dry_run:
        changes["test.sh"] = True
    else:
        test_sh_path.parent.mkdir(parents=True, exist_ok=True)
        test_sh_path.write_text(new_test_sh)
        changes["test.sh"] = True

    # --- 5. solution/solve.sh (prepend environment setup for oracle agent) ---
    solve_sh_path = target / "solution" / "solve.sh"
    already_patched_solve_marker = "# --- Environment setup (for generic base images) ---"
    if solve_sh_path.exists():
        original_solve = solve_sh_path.read_text()
        if already_patched_solve_marker in original_solve:
            changes["solve.sh"] = False
        elif dry_run:
            changes["solve.sh"] = True
        else:
            setup_block = SOLVE_SH_SETUP_TEMPLATE.format(
                repo=repo_name,
                base_commit=base_commit,
            )
            lines = original_solve.split("\n")
            # Insert after the shebang + any leading comments
            insert_idx = next(
                (i for i, ln in enumerate(lines)
                 if ln.strip() and not ln.strip().startswith("#")),
                0,
            )
            patched = (
                "\n".join(lines[:insert_idx])
                + "\n"
                + setup_block
                + "\n".join(lines[insert_idx:])
            )
            solve_sh_path.write_text(patched)
            changes["solve.sh"] = True
    else:
        changes["solve.sh"] = False

    return changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_extract(tasks_root: Path, parallel: int, dry_run: bool) -> None:
    """Phase 1: extract r2e_tests + run_tests.sh from docker images."""
    task_dirs = sorted(
        d for d in tasks_root.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    print(f"Found {len(task_dirs)} tasks — extracting test infrastructure from docker images")

    totals: dict[str, int] = {}
    errors = 0

    def _do_extract(td: Path) -> tuple[str, dict]:
        meta = _read_metadata(td)
        if meta is None:
            return td.name, {"error": True, "reason": "no metadata.json"}
        docker_image = meta.get("docker_image", "")
        if not docker_image:
            return td.name, {"error": True, "reason": "no docker_image in metadata"}
        return td.name, _extract_from_image(docker_image, td, dry_run=dry_run)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_do_extract, td): td for td in task_dirs}
        for fut in as_completed(futures):
            name, result = fut.result()
            if result.get("error"):
                errors += 1
                print(f"  ERROR {name}: {result.get('reason', result.get('error_msg', '?'))}")
            else:
                for k, v in result.items():
                    if v:
                        totals[k] = totals.get(k, 0) + 1

    action = "Would extract" if dry_run else "Extracted"
    print(f"\n{action}:")
    for filename, count in sorted(totals.items()):
        print(f"  {filename}: {count}/{len(task_dirs)}")
    if errors:
        print(f"  Errors: {errors}")


def run_patch(
    tasks_root: Path,
    output_dir: Path | None,
    dry_run: bool,
    skip_missing_tests: bool,
) -> None:
    """Phase 2: apply the Dockerfile + instruction + test.sh patch."""
    task_dirs = sorted(
        d for d in tasks_root.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    print(f"Found {len(task_dirs)} tasks in {tasks_root}")

    if output_dir and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    totals: dict[str, int] = {}
    errors = 0
    skipped = 0

    for td in task_dirs:
        result = patch_task(
            td,
            output_dir=output_dir,
            dry_run=dry_run,
            require_tests=not skip_missing_tests,
        )
        if result.get("error"):
            reason = result.get("reason", "?")
            if "extract-tests" in str(reason):
                skipped += 1
            else:
                errors += 1
                print(f"  ERROR {td.name}: {reason}")
            continue
        for k, v in result.items():
            if v is True:
                totals[k] = totals.get(k, 0) + 1

    action = "Would patch" if dry_run else "Patched"
    print(f"\n{action}:")
    for filename, count in sorted(totals.items()):
        print(f"  {filename}: {count}/{len(task_dirs)}")
    if skipped:
        print(f"  Skipped (run --extract-tests first): {skipped}")
    if errors:
        print(f"  Errors: {errors}")

    # Report unique Dockerfiles after patching
    if not dry_run:
        out_root = output_dir or tasks_root
        dockerfiles: set[str] = set()
        for td in sorted(out_root.iterdir()):
            df = td / "environment" / "Dockerfile"
            if df.exists():
                dockerfiles.add(df.read_text())
        print(f"\nUnique Dockerfiles: {len(dockerfiles)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch R2E-Gym tasks for generic base images + Harbor file upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tasks_dir", help="Root directory containing task folders")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write patched tasks here (default: patch in-place)",
    )
    parser.add_argument(
        "--extract-tests",
        action="store_true",
        help="Phase 1: pull docker images and extract test infrastructure to setup_files/",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel docker pulls for --extract-tests (default: 4)",
    )
    parser.add_argument(
        "--skip-missing-tests",
        action="store_true",
        help="Patch tasks even if setup_files/r2e_tests.tar.gz is missing (for dry-run/testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any files",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir)
    if not tasks_root.is_dir():
        raise SystemExit(f"Not a directory: {tasks_root}")

    if args.extract_tests:
        run_extract(tasks_root, parallel=args.parallel, dry_run=args.dry_run)
    else:
        run_patch(
            tasks_root,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            skip_missing_tests=args.skip_missing_tests,
        )


if __name__ == "__main__":
    main()
