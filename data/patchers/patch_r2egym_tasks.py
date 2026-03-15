#!/usr/bin/env python3
"""
Patch R2E-Gym tasks from 4,578 unique docker images → 10 images (one per repo).

R2E-Gym/R2E-Gym-Subset contains 4,578 tasks across exactly 10 GitHub repos:
  pandas (1444), numpy (781), pillow (620), orange3 (482), aiohttp (299),
  tornado (261), scrapy (215), pyramid (189), datalad (179), coveragepy (108)

Each task currently uses a unique pre-built image (repo @ specific buggy commit).
After patching, each task uses one of 10 shared images where the repo is
pre-cloned at HEAD with all deps installed. The agent's only setup step is:

    git checkout {commit_hash}   # fast: seconds, not minutes

Per-task test infrastructure (r2e_tests/, run_tests.sh) is extracted once from
the original docker images and placed in setup_files/, which Harbor uploads to
/setup_files/ in the container before the agent runs.
Requires: penfever/temp-override Harbor branch.

Two-phase workflow
------------------
Phase 1 — extract test infrastructure from original docker images (run once):

    python patch_r2egym_tasks.py /path/to/tasks --extract-tests [--parallel 8]

Phase 2 — apply patch (idempotent):

    python patch_r2egym_tasks.py /path/to/tasks --output-dir /path/to/patched

Changes per task
----------------
  1. environment/Dockerfile       — One of 10 repo-specific images (pre-cloned + installed)
  2. setup_files/metadata.json    — Moved from environment/workspace/metadata.json
  3. setup_files/r2e_tests.tar.gz — Test infrastructure extracted from original image
  4. setup_files/run_tests.sh     — Test runner extracted from original image
  5. instruction.md               — Setup preamble reduced to: git checkout {commit}
  6. tests/test.sh                — Updated to use setup_files/ infrastructure

Usage
-----
    python patch_r2egym_tasks.py /path/to/tasks --dry-run
    python patch_r2egym_tasks.py /path/to/tasks --output-dir /path/to/patched
    python patch_r2egym_tasks.py /path/to/tasks --extract-tests --parallel 8
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
# Per-repo Dockerfile configs
# Each repo gets one shared image: pre-cloned at HEAD, deps pre-installed.
# Agent only runs: git checkout {commit_hash}
# ---------------------------------------------------------------------------

# Native apt packages needed beyond the common baseline, keyed by repo name.
# (Common baseline: git curl wget jq build-essential libffi-dev libssl-dev)
_REPO_EXTRA_APT: dict[str, str] = {
    "pandas":     "gfortran libopenblas-dev liblapack-dev pkg-config",
    "numpy":      "gfortran libopenblas-dev liblapack-dev pkg-config",
    "pillow":     "libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libwebp-dev",
    "orange3":    "libxml2-dev libxslt1-dev",
    "aiohttp":    "",
    "tornado":    "",
    "scrapy":     "libxml2-dev libxslt1-dev",
    "pyramid":    "",
    "datalad":    "git-annex",
    "coveragepy": "",
}

# pip install command run inside the cloned repo.
# Most repos: pip install -e .[dev] or -e .[test]; fall back to plain -e .
_REPO_INSTALL_CMD: dict[str, str] = {
    "pandas":     "pip install -e '.[dev,test]' --no-build-isolation 2>/dev/null || pip install -e . --no-build-isolation",
    "numpy":      "pip install -e . --no-build-isolation",
    "pillow":     "pip install -e '.[tests]' 2>/dev/null || pip install -e .",
    "orange3":    "pip install -e '.[test]' 2>/dev/null || pip install -e .",
    "aiohttp":    "pip install -e '.[dev]' 2>/dev/null || pip install -e .",
    "tornado":    "pip install -e . && pip install pytest",
    "scrapy":     "pip install -e '.[tests]' 2>/dev/null || pip install -e .",
    "pyramid":    "pip install -e '.[testing]' 2>/dev/null || pip install -e .",
    "datalad":    "pip install -e '.[devel]' 2>/dev/null || pip install -e .",
    "coveragepy": "pip install -e '.[dev]' 2>/dev/null || pip install -e .",
}

# GitHub clone URLs
_REPO_GITHUB_URL: dict[str, str] = {
    "pandas":     "https://github.com/pandas-dev/pandas.git",
    "numpy":      "https://github.com/numpy/numpy.git",
    "pillow":     "https://github.com/python-pillow/Pillow.git",
    "orange3":    "https://github.com/biolab/orange3.git",
    "aiohttp":    "https://github.com/aio-libs/aiohttp.git",
    "tornado":    "https://github.com/tornadoweb/tornado.git",
    "scrapy":     "https://github.com/scrapy/scrapy.git",
    "pyramid":    "https://github.com/Pylons/pyramid.git",
    "datalad":    "https://github.com/datalad/datalad.git",
    "coveragepy": "https://github.com/nedbat/coveragepy.git",
}

# Python version per repo (3.11 is safe for all; adjust if specific repos need 3.10)
_REPO_PYTHON_VERSION: dict[str, str] = {
    "pandas":     "3.11",
    "numpy":      "3.11",
    "pillow":     "3.11",
    "orange3":    "3.10",   # orange3 sometimes lags on Python version support
    "aiohttp":    "3.11",
    "tornado":    "3.11",
    "scrapy":     "3.11",
    "pyramid":    "3.11",
    "datalad":    "3.11",
    "coveragepy": "3.11",
}


def _build_dockerfile(repo_name: str) -> str:
    """Build the Dockerfile for a specific repo."""
    python_version = _REPO_PYTHON_VERSION.get(repo_name, "3.11")
    github_url = _REPO_GITHUB_URL[repo_name]
    install_cmd = _REPO_INSTALL_CMD[repo_name]
    extra_apt = _REPO_EXTRA_APT.get(repo_name, "").strip()

    apt_extra_line = (
        f"    {extra_apt} \\\n" if extra_apt else ""
    )

    return f"""\
FROM python:{python_version}-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \\
    git curl wget jq \\
    build-essential \\
    libffi-dev libssl-dev \\
    locales locales-all tzdata \\
    tmux \\
{apt_extra_line}\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

# Pre-clone the repo and install all dependencies at HEAD.
# Agent only needs: git checkout {{commit_hash}}
RUN git clone {github_url} /testbed
WORKDIR /testbed
RUN {install_cmd}

RUN mkdir -p /logs /r2e_tests
"""


# ---------------------------------------------------------------------------
# test.sh template
# ---------------------------------------------------------------------------

TEST_SH_TEMPLATE = """\
#!/bin/bash
set -e

export PATH=/testbed/.venv/bin:/root/.local/bin:$PATH

# ---------------------------------------------------------------------------
# Restore per-task test infrastructure from setup_files/
# (uploaded by Harbor before the agent runs — penfever/temp-override branch)
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

# Clean up bytecode to avoid stale cache
find /testbed -name '*.pyc' -delete 2>/dev/null || true
find /testbed -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find /r2e_tests  -name '*.pyc' -delete 2>/dev/null || true
find /r2e_tests  -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# Move test infrastructure out of agent's reach
for skip_file in "run_tests.sh" "r2e_tests"; do
    if [ -e "/testbed/$skip_file" ]; then
        mv "/testbed/$skip_file" "/root/$skip_file"
    fi
done
if [ -d /r2e_tests ]; then
    mv /r2e_tests /root/r2e_tests
fi
if [ -d /root/r2e_tests ]; then
    ln -sf /root/r2e_tests /testbed/r2e_tests
fi

mkdir -p /logs/verifier

# Run tests
TEST_OUTPUT_FILE="/tmp/test_output.txt"
if [ -f /root/run_tests.sh ]; then
    bash /root/run_tests.sh 2>&1 | tee "$TEST_OUTPUT_FILE"
else
    echo "Error: /root/run_tests.sh not found after extraction."
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

# Calculate reward (reads expected_output_json from /setup_files/metadata.json)
mkdir -p /tests
cat > /tests/calculate_reward.py << 'REWARD_SCRIPT_EOF'
#!/usr/bin/env python3
import re, json, sys
from pathlib import Path

def parse_log_pytest(log):
    if not log or "short test summary info" not in log:
        return {}
    status_map = {}
    for line in log.split("short test summary info")[1].strip().split("\\n"):
        if "PASSED" in line:
            status_map[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            status_map[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            try:
                name = ".".join(line.split("::")[1:])
            except IndexError:
                name = line
            status_map[name.split(" - ")[0]] = "ERROR"
    return status_map

def decolor(d):
    strip = lambda k: re.sub(r"\\u001b\\[\\d+m", "", k)
    return {strip(k): v for k, v in d.items()}

def get_reward(parsed, expected_json):
    p = {k.split(" - ")[0]: v for k, v in decolor(parsed).items()}
    e = {k.split(" - ")[0]: v for k, v in decolor(json.loads(expected_json)).items()}
    p = dict(sorted(p.items()))
    e = dict(sorted(e.items()))
    if len(p) != len(e):
        return 0.0
    return 1.0 if all(k in e and p[k] == e[k] for k in p if k) else 0.0

test_output = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()
for candidate in ["/setup_files/metadata.json", "/workspace/metadata.json"]:
    if Path(candidate).exists():
        meta = json.loads(Path(candidate).read_text())
        break
else:
    Path("/logs/verifier/reward.txt").write_text("0")
    sys.exit(0)

reward = get_reward(parse_log_pytest(test_output), meta.get("expected_output_json", "{}"))
Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text(str(reward))
print(f"Reward: {reward}")
REWARD_SCRIPT_EOF

python3 /tests/calculate_reward.py "$TEST_OUTPUT_FILE"
if [ ! -f /logs/verifier/reward.txt ]; then
    echo "0" > /logs/verifier/reward.txt
fi
"""

# instruction preamble: only git checkout needed (repo + deps already in image)
SETUP_PREAMBLE_TEMPLATE = """\
## Environment Setup (complete this step first)

```bash
cd /testbed && git checkout {base_commit}
```

---

"""

SOLVE_SH_SETUP_TEMPLATE = """\
# --- Environment setup: checkout specific bug-fix commit ---
cd /testbed && git checkout {base_commit}
# --- End environment setup ---

"""


# ---------------------------------------------------------------------------
# Docker extraction helpers (Phase 1)
# ---------------------------------------------------------------------------

def _extract_from_image(
    docker_image: str,
    task_dir: Path,
    dry_run: bool = False,
) -> dict[str, bool | str]:
    """Pull docker image, extract r2e_tests/ + run_tests.sh into setup_files/."""
    setup_files_dir = task_dir / "setup_files"
    r2e_tar = setup_files_dir / "r2e_tests.tar.gz"
    run_tests_dst = setup_files_dir / "run_tests.sh"

    if r2e_tar.exists() and run_tests_dst.exists():
        return {"r2e_tests.tar.gz": False, "run_tests.sh": False}

    if dry_run:
        return {"r2e_tests.tar.gz": True, "run_tests.sh": True}

    setup_files_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool | str] = {}

    try:
        subprocess.run(
            ["docker", "pull", docker_image],
            check=True, capture_output=True, text=True, timeout=300,
        )
        create_out = subprocess.run(
            ["docker", "create", docker_image],
            check=True, capture_output=True, text=True, timeout=30,
        )
        container_id = create_out.stdout.strip()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)

                # r2e_tests/ — try common locations
                results["r2e_tests.tar.gz"] = False
                for src in ["/r2e_tests", "/testbed/r2e_tests"]:
                    cp = subprocess.run(
                        ["docker", "cp", f"{container_id}:{src}", str(tmp / "r2e_tests")],
                        capture_output=True, text=True, timeout=60,
                    )
                    if cp.returncode == 0:
                        with tarfile.open(r2e_tar, "w:gz") as tf:
                            tf.add(tmp / "r2e_tests", arcname="r2e_tests")
                        results["r2e_tests.tar.gz"] = True
                        break

                # run_tests.sh — try common locations
                results["run_tests.sh"] = False
                for src in ["/root/run_tests.sh", "/testbed/run_tests.sh"]:
                    cp = subprocess.run(
                        ["docker", "cp", f"{container_id}:{src}", str(tmp / "run_tests.sh")],
                        capture_output=True, text=True, timeout=30,
                    )
                    if cp.returncode == 0:
                        shutil.copy(tmp / "run_tests.sh", run_tests_dst)
                        results["run_tests.sh"] = True
                        break
        finally:
            subprocess.run(["docker", "rm", container_id],
                           capture_output=True, text=True, timeout=30)

    except subprocess.CalledProcessError as exc:
        results["error"] = True
        results["error_msg"] = str(exc)

    return results


# ---------------------------------------------------------------------------
# Patching logic (Phase 2)
# ---------------------------------------------------------------------------

def _read_metadata(task_dir: Path) -> dict | None:
    for candidate in [
        task_dir / "environment" / "workspace" / "metadata.json",
        task_dir / "setup_files" / "metadata.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def _repo_name_from_metadata(metadata: dict) -> str | None:
    """Extract short repo name (e.g. 'pandas') from metadata."""
    repo_name = metadata.get("repo_name", "")
    # repo_name may be "pandas-dev/pandas" or just "pandas"
    short = repo_name.split("/")[-1].lower() if repo_name else ""
    # Map common variations
    aliases = {"pillow": "pillow", "pil": "pillow", "coveragepy": "coveragepy"}
    return aliases.get(short, short) if short in _REPO_GITHUB_URL else (
        next((k for k in _REPO_GITHUB_URL if short.startswith(k) or k.startswith(short)), None)
    )


def patch_task(
    task_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    require_tests: bool = True,
) -> dict[str, bool | str]:
    """Patch a single task. Returns dict of changes."""
    metadata = _read_metadata(task_dir)
    if metadata is None:
        return {"error": True, "reason": "no metadata.json"}

    repo_name = _repo_name_from_metadata(metadata)
    if repo_name is None:
        return {"error": True, "reason": f"unknown repo: {metadata.get('repo_name')}"}

    base_commit = metadata.get("base_commit", "HEAD")

    if output_dir:
        target = output_dir / task_dir.name
        if not dry_run:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(task_dir, target)
    else:
        target = task_dir

    # Check test infrastructure is present
    r2e_tar = target / "setup_files" / "r2e_tests.tar.gz"
    run_tests_sh = target / "setup_files" / "run_tests.sh"
    if require_tests and not (r2e_tar.exists() and run_tests_sh.exists()):
        return {
            "error": True,
            "reason": "setup_files missing — run --extract-tests first",
        }

    changes: dict[str, bool | str] = {}

    # 1. Dockerfile (one of 10, keyed by repo)
    dockerfile_path = target / "environment" / "Dockerfile"
    new_dockerfile = _build_dockerfile(repo_name)
    if dry_run:
        changes["Dockerfile"] = True
    else:
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(new_dockerfile)
        changes["Dockerfile"] = True

    # 2. setup_files/metadata.json (move from environment/workspace/)
    original_workspace_meta = target / "environment" / "workspace" / "metadata.json"
    setup_meta = target / "setup_files" / "metadata.json"
    if original_workspace_meta.exists() and not setup_meta.exists():
        if dry_run:
            changes["setup_files/metadata.json"] = True
        else:
            setup_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(original_workspace_meta, setup_meta)
            changes["setup_files/metadata.json"] = True
    else:
        changes["setup_files/metadata.json"] = False

    # 3. instruction.md (prepend lightweight setup preamble)
    instruction_path = target / "instruction.md"
    already_patched = "## Environment Setup (complete this step first)"
    if instruction_path.exists():
        original = instruction_path.read_text()
        if already_patched in original:
            changes["instruction.md"] = False
        elif dry_run:
            changes["instruction.md"] = True
        else:
            preamble = SETUP_PREAMBLE_TEMPLATE.format(base_commit=base_commit)
            instruction_path.write_text(preamble + original)
            changes["instruction.md"] = True
    else:
        changes["instruction.md"] = False

    # 4. tests/test.sh
    test_sh_path = target / "tests" / "test.sh"
    if dry_run:
        changes["test.sh"] = True
    else:
        test_sh_path.parent.mkdir(parents=True, exist_ok=True)
        test_sh_path.write_text(TEST_SH_TEMPLATE)
        changes["test.sh"] = True

    # 5. solution/solve.sh (prepend checkout step for oracle agent)
    solve_sh_path = target / "solution" / "solve.sh"
    already_patched_solve = "# --- Environment setup: checkout specific bug-fix commit ---"
    if solve_sh_path.exists():
        original_solve = solve_sh_path.read_text()
        if already_patched_solve in original_solve:
            changes["solve.sh"] = False
        elif dry_run:
            changes["solve.sh"] = True
        else:
            setup_block = SOLVE_SH_SETUP_TEMPLATE.format(base_commit=base_commit)
            lines = original_solve.split("\n")
            insert_idx = next(
                (i for i, ln in enumerate(lines)
                 if ln.strip() and not ln.strip().startswith("#")),
                0,
            )
            patched = "\n".join(lines[:insert_idx]) + "\n" + setup_block + "\n".join(lines[insert_idx:])
            solve_sh_path.write_text(patched)
            changes["solve.sh"] = True
    else:
        changes["solve.sh"] = False

    return changes


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def run_extract(tasks_root: Path, parallel: int, dry_run: bool) -> None:
    task_dirs = sorted(d for d in tasks_root.iterdir()
                       if d.is_dir() and (d / "instruction.md").exists())
    print(f"Found {len(task_dirs)} tasks — extracting test infrastructure from docker images")

    totals: dict[str, int] = {}
    errors = 0

    def _do(td: Path) -> tuple[str, dict]:
        meta = _read_metadata(td)
        if not meta:
            return td.name, {"error": True, "reason": "no metadata.json"}
        docker_image = meta.get("docker_image", "")
        if not docker_image:
            return td.name, {"error": True, "reason": "no docker_image in metadata"}
        return td.name, _extract_from_image(docker_image, td, dry_run=dry_run)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_do, td): td for td in task_dirs}
        for fut in as_completed(futures):
            name, result = fut.result()
            if result.get("error"):
                errors += 1
                print(f"  ERROR {name}: {result.get('reason', result.get('error_msg', '?'))}")
            else:
                for k, v in result.items():
                    if v is True:
                        totals[k] = totals.get(k, 0) + 1

    action = "Would extract" if dry_run else "Extracted"
    print(f"\n{action}:")
    for k, c in sorted(totals.items()):
        print(f"  {k}: {c}/{len(task_dirs)}")
    if errors:
        print(f"  Errors: {errors}")


def run_patch(
    tasks_root: Path,
    output_dir: Path | None,
    dry_run: bool,
    skip_missing_tests: bool,
) -> None:
    task_dirs = sorted(d for d in tasks_root.iterdir()
                       if d.is_dir() and (d / "instruction.md").exists())
    print(f"Found {len(task_dirs)} tasks in {tasks_root}")

    if output_dir and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    errors = skipped = 0

    for td in task_dirs:
        result = patch_task(td, output_dir=output_dir, dry_run=dry_run,
                            require_tests=not skip_missing_tests)
        if result.get("error"):
            reason = str(result.get("reason", ""))
            if "extract-tests" in reason:
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
    for k, c in sorted(totals.items()):
        print(f"  {k}: {c}/{len(task_dirs)}")
    if skipped:
        print(f"  Skipped (run --extract-tests first): {skipped}")
    if errors:
        print(f"  Errors: {errors}")

    # Report unique Dockerfiles
    if not dry_run:
        out_root = output_dir or tasks_root
        unique_dfs: set[str] = set()
        for td in out_root.iterdir():
            df = td / "environment" / "Dockerfile"
            if df.exists():
                unique_dfs.add(df.read_text())
        print(f"\nUnique Dockerfiles after patching: {len(unique_dfs)}")
        print(f"(Expected: {len(_REPO_GITHUB_URL)} — one per repo)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch R2E-Gym tasks: 4,578 unique images → 10 (one per repo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("tasks_dir", help="Root directory containing task folders")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write patched tasks here (default: in-place)")
    parser.add_argument("--extract-tests", action="store_true",
                        help="Phase 1: pull docker images and extract test infra to setup_files/")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Parallel docker pulls for --extract-tests (default: 4)")
    parser.add_argument("--skip-missing-tests", action="store_true",
                        help="Patch even if setup_files/r2e_tests.tar.gz is missing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing files")
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir)
    if not tasks_root.is_dir():
        raise SystemExit(f"Not a directory: {tasks_root}")

    if args.extract_tests:
        run_extract(tasks_root, parallel=args.parallel, dry_run=args.dry_run)
    else:
        run_patch(tasks_root, output_dir=args.output_dir,
                  dry_run=args.dry_run, skip_missing_tests=args.skip_missing_tests)


if __name__ == "__main__":
    main()
