#!/usr/bin/env python3
"""
Patch openswe-harbor-tasks to reduce from ~45K unique snapshots to 4
(one per normalized Python version: 3.9 / 3.10 / 3.11 / 3.12).

Strategy
--------
Each task's unique Dockerfile is replaced with a generic miniconda3-based
image whose only variable is the Python version.  The per-task install steps
(git clone, checkout, pip installs, env vars, apt installs) are extracted and
placed in setup_files/setup.sh, which the agent runs at the start of each
trial and which solution/solve.sh calls automatically for oracle runs.

Snapshot count: 45 320 → 4

Python version normalisation
----------------------------
  2.7 / 3.5 / 3.6 / 3.7 / 3.8  →  3.9
  3.9                            →  3.9
  3.10                           →  3.10
  3.11                           →  3.11
  3.12 / 3.13 / 3.14            →  3.12

Usage
-----
    python data/patchers/patch_openswe_tasks.py \\
        /Users/benjaminfeuer/Documents/openswe-tasks \\
        --output-dir /Users/benjaminfeuer/Documents/openswe-tasks-patched

    # Quick smoke-test on 20 tasks
    python data/patchers/patch_openswe_tasks.py \\
        /Users/benjaminfeuer/Documents/openswe-tasks \\
        --output-dir /tmp/openswe_patched_test \\
        --limit 20

    # Verify snapshot count afterwards
    python -m scripts.harbor.count_snapshots_from_tasks \\
        --local-dataset /Users/benjaminfeuer/Documents/openswe-tasks-patched
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import textwrap
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Python version normalisation
# ---------------------------------------------------------------------------

SUPPORTED_VERSIONS = ("3.9", "3.10", "3.11", "3.12")
VERSION_FLOOR = "3.9"
VERSION_CEIL = "3.12"


def normalize_python_version(raw: str) -> str:
    """Normalise an arbitrary Python version string to one of the 4 supported."""
    try:
        major, minor = int(raw.split(".")[0]), int(raw.split(".")[1])
    except Exception:
        return VERSION_FLOOR
    if major < 3 or (major == 3 and minor < 9):
        return VERSION_FLOOR
    if major == 3 and minor >= 12:
        return VERSION_CEIL
    normalized = f"{major}.{minor}"
    return normalized if normalized in SUPPORTED_VERSIONS else VERSION_CEIL


def extract_python_version(df_text: str) -> str:
    """Extract Python version from conda create command in Dockerfile."""
    m = re.search(r"conda create -n testbed python=(\S+)", df_text)
    return normalize_python_version(m.group(1)) if m else VERSION_FLOOR


# ---------------------------------------------------------------------------
# Base Dockerfile template (one per Python version — the shared snapshot)
# ---------------------------------------------------------------------------

BASE_DOCKERFILE_TEMPLATE = """\
FROM continuumio/miniconda3:latest

# Create conda testbed env with Python {python_version}
RUN conda create -n testbed python={python_version} -y && conda clean -afy

# Initialize conda for bash login shells and activate testbed by default
RUN conda init bash && \\
    echo 'conda activate testbed' >> /root/.bashrc && \\
    printf '[[ -f ~/.bashrc ]] && . ~/.bashrc\\n' > /root/.bash_profile

# Install git, build tools, and common system libs
RUN apt-get update && apt-get install -y \\
    git build-essential curl wget \\
    libssl-dev libffi-dev libpq-dev \\
    libxml2-dev libxslt1-dev \\
    libmysqlclient-dev default-libmysqlclient-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/* || \\
    (apt-get update && apt-get install -y git build-essential curl wget \\
     libssl-dev libffi-dev pkg-config && rm -rf /var/lib/apt/lists/*)

# Pre-install common test infrastructure in the testbed env
RUN /opt/conda/envs/testbed/bin/pip install --upgrade pip setuptools wheel && \\
    /opt/conda/envs/testbed/bin/pip install pytest coverage

RUN mkdir -p /logs/verifier /testbed && chmod 777 /logs/verifier
WORKDIR /testbed
"""


def make_base_dockerfile(python_version: str) -> str:
    return BASE_DOCKERFILE_TEMPLATE.format(python_version=python_version)


# ---------------------------------------------------------------------------
# Dockerfile → shell-script translation
# ---------------------------------------------------------------------------

def _join_continuation_lines(text: str) -> list[str]:
    """Join multi-line Dockerfile instructions (lines ending with \\) into single strings."""
    lines = text.split("\n")
    joined: list[str] = []
    buf: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf.append(stripped[:-1])
        else:
            buf.append(stripped)
            joined.append(" ".join(buf))
            buf = []
    if buf:
        joined.append(" ".join(buf))
    return joined


def _split_cmd_and(cmd: str) -> list[str]:
    """Split a shell command string on && that are not inside single or double quotes."""
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        if c == '&' and not in_single and not in_double and i + 1 < len(cmd) and cmd[i + 1] == '&':
            parts.append(''.join(current).strip())
            current = []
            i += 2
            continue
        current.append(c)
        i += 1
    if current:
        parts.append(''.join(current).strip())
    return [p for p in parts if p]


def _transform_cmd(cmd: str) -> Optional[str]:
    """
    Apply post-extraction transformations to a single shell command.

    Returns the transformed command, or None if the command should be dropped.
    """
    cmd = cmd.strip()
    if not cmd:
        return None
    # Drop pipx ensurepath — conda env bin is already on PATH
    if re.match(r'^pipx\s+ensurepath\b', cmd):
        return None
    # pipx install X → pip install X (pipx puts binaries in ~/.local/bin which
    # is not on PATH when test.sh runs later; pip installs into conda env bin)
    if re.match(r'^pipx\s+install\b', cmd):
        cmd = re.sub(r'^pipx\s+install\b', 'pip install', cmd)
    # apt-get install may fail for x86-only packages on ARM64; add || true so
    # setup continues and pip installs Python deps from pre-built wheels regardless
    if re.match(r'^apt(?:-get)?\s+install\b', cmd) and '|| true' not in cmd:
        cmd = cmd + ' || true'
    # Shell variable assignments using command substitution (VAR=$(cmd)) can exit
    # fatally with set -e when cmd fails on ARM64 (e.g. basename on an empty find
    # result for x86-only paths like /usr/lib/x86_64-linux-gnu).  Make them
    # non-fatal so the rest of setup.sh continues even if the capture fails.
    if re.match(r'^[A-Za-z_][A-Za-z_0-9]*=\$\(', cmd) and '|| true' not in cmd:
        cmd = cmd + ' || true'
    return cmd


def _extract_run_cmds(cmd: str) -> list[str]:
    """
    Extract executable shell commands from a Dockerfile RUN instruction value.

    Handles:
      - bash -lc 'inner' [|| fallback]   (single-quoted, any suffix ignored)
      - bash -lc "inner" [|| fallback]   (double-quoted)
      - A && B chains                     (quote-aware split)
      - Plain commands                    (kept as-is)
    """
    result: list[str] = []
    for part in _split_cmd_and(cmd):
        part = part.strip()
        if not part:
            continue
        # bash -lc 'inner'  (single-quoted; stops at closing ')
        m = re.match(r"bash\s+-(?:l?c|lc|c)\s+'([^']*)'", part, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            t = _transform_cmd(inner)
            if t:
                result.append(t)
            continue
        # bash -lc "inner"  (double-quoted; handles \" escapes)
        m2 = re.match(r'bash\s+-(?:l?c|lc|c)\s+"((?:[^"\\]|\\.)*)"', part, re.DOTALL)
        if m2:
            inner = m2.group(1).strip()
            t = _transform_cmd(inner)
            if t:
                result.append(t)
            continue
        # /bin/bash -c 'inner'
        m3 = re.match(r"(?:/bin/)?bash\s+(?:-[a-z]+\s+)*'([^']*)'", part)
        if m3:
            inner = m3.group(1).strip()
            t = _transform_cmd(inner)
            if t:
                result.append(t)
            continue
        # Skip pure echo/true/: fallback parts (common after ||)
        # But keep echo commands that redirect to files (echo "..." > file) —
        # those actually write config files and must be preserved.
        if re.match(r'^(?:true|:)(?:\s|$)', part):
            continue
        if re.match(r'^echo(?:\s|$)', part) and '>' not in part:
            continue
        # Strip trailing || fallback from plain commands
        plain = re.sub(r'\s*\|\|\s*(?:echo|true|:)(?:\s.*)?$', '', part).strip()
        if not plain:
            continue
        t = _transform_cmd(plain)
        if t:
            result.append(t)
    return result


def translate_dockerfile_to_shell(post_workdir_text: str, repo_url: str, base_commit: str) -> str:
    """
    Convert post-WORKDIR Dockerfile content into a bash setup script.

    The script:
      1. Activates the conda testbed env
      2. Clones the repo (idempotent)
      3. Runs translated RUN/ENV instructions in order
    """
    lines = _join_continuation_lines(post_workdir_text)

    # Per-RUN-instruction command blocks; we emit `cd <workdir>` between blocks to
    # mimic Docker's behaviour where each RUN instruction starts fresh at WORKDIR.
    shell_blocks: list[list[str]] = []
    env_exports: list[str] = []
    cur_workdir = "/testbed"

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Identify instruction keyword
        parts = stripped.split(None, 1)
        keyword = parts[0].upper() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if keyword == "SHELL":
            continue

        if keyword == "ENV":
            # ENV VAR=value  or  ENV VAR value  or  ENV VAR=v1 \\\n    VAR2=v2
            for assignment in re.split(r"\s+(?=[A-Z_][A-Z_0-9]*=)", rest):
                assignment = assignment.strip()
                if not assignment:
                    continue
                if "=" in assignment:
                    env_exports.append(f"export {assignment}")
                else:
                    # old-style ENV KEY VALUE
                    kv = assignment.split(None, 1)
                    if len(kv) == 2:
                        env_exports.append(f'export {kv[0]}="{kv[1]}"')
            continue

        if keyword == "WORKDIR":
            # Track WORKDIR so inter-block resets land in the right directory
            wd = rest.strip().rstrip("/")
            if wd:
                cur_workdir = wd
            continue

        if keyword in ("LABEL", "EXPOSE", "VOLUME", "USER", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "ARG"):
            continue

        if keyword in ("COPY", "ADD"):
            # Skip — these require a build context we don't have at runtime
            continue

        if keyword == "RUN":
            cmds = _extract_run_cmds(rest.strip())
            if cmds:
                shell_blocks.append(cmds)
            continue

        # Skip Dockerfile-only instructions with no shell equivalent
        if keyword in ("FROM", "CMD", "ENTRYPOINT", "MAINTAINER"):
            continue

        # Fallback: bare shell continuation lines — these appear when an inline
        # comment (# ...) breaks a &&-chain continuation without a trailing \,
        # causing _join_continuation_lines to emit the remaining parts as their
        # own "lines" without a Dockerfile keyword prefix.
        # Append to the last block since these are continuations of the previous RUN.
        cmds = _extract_run_cmds(stripped)
        if cmds:
            if shell_blocks:
                shell_blocks[-1].extend(cmds)
            else:
                shell_blocks.append(cmds)

    # Flatten blocks, inserting `cd <workdir>` between them to prevent directory
    # drift when a RUN instruction ends in a different directory (e.g. cd /tmp && ...).
    shell_commands: list[str] = []
    for i, block in enumerate(shell_blocks):
        if i > 0:
            shell_commands.append(f"cd {cur_workdir}")
        shell_commands.extend(block)

    # Build the script
    clone_block = textwrap.dedent(f"""\
        # Clone repository at the target commit (idempotent)
        if [ ! -d /testbed/.git ]; then
            git clone {repo_url}.git /testbed
            cd /testbed
            git checkout {base_commit}
        fi
        cd /testbed
    """)

    env_block = "\n".join(env_exports) + "\n" if env_exports else ""
    cmd_block = "\n".join(shell_commands) + "\n" if shell_commands else ""

    script = textwrap.dedent("""\
        #!/bin/bash
        # Auto-generated setup script for openswe task.
        # Clones the repository, checks out the target commit, and installs dependencies.
        set -e
        source /opt/conda/etc/profile.d/conda.sh
        conda activate testbed
        # Pin setuptools<72 globally so that pip's isolated build environments also
        # get a setuptools version that exposes pkg_resources (removed in >=72).
        # This covers both direct imports and packages whose setup.py use pkg_resources.
        echo 'setuptools<72' > /tmp/pip-constraints.txt
        export PIP_CONSTRAINT=/tmp/pip-constraints.txt
        python -c 'import pkg_resources' 2>/dev/null || pip install -q 'setuptools<72' || true
        pip install -q wheel || true

    """) + clone_block + "\n" + env_block + cmd_block

    return script


# ---------------------------------------------------------------------------
# Instruction.md preamble
# ---------------------------------------------------------------------------

INSTRUCTION_PREAMBLE = """\
## Environment Setup

The repository has **not** been cloned yet. Run the following command first
to clone the repo, check out the target commit, and install dependencies:

```bash
bash /setup_files/setup.sh
```

The code will then be available at `/testbed`.

---

"""


# ---------------------------------------------------------------------------
# solution/solve.sh wrapper
# ---------------------------------------------------------------------------

def wrap_solve_sh(original_solve_sh: str) -> str:
    """Prepend a setup.sh call to solve.sh so the oracle works out-of-the-box."""
    header = textwrap.dedent("""\
        #!/bin/bash
        # Ensure repository is cloned and dependencies are installed before applying fix
        if [ ! -d /testbed/.git ]; then
            bash /setup_files/setup.sh
        fi

    """)
    # Strip the original shebang if present to avoid duplicate
    body = re.sub(r"^#!/bin/bash\s*\n", "", original_solve_sh, count=1)
    return header + body


# ---------------------------------------------------------------------------
# Per-task patching
# ---------------------------------------------------------------------------

def patch_task(
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Patch one task directory. Returns a status dict."""
    result = {"task": src_dir.name, "status": "ok", "python_version": None, "notes": []}

    # --- Read source files ---
    df_path = src_dir / "environment" / "Dockerfile"
    if not df_path.exists():
        result["status"] = "skip_no_dockerfile"
        return result

    df_text = df_path.read_text()
    config_path = src_dir / "tests" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    repo = config.get("repo", "")
    base_commit = config.get("base_commit", "")

    # Extract repo URL from Dockerfile (more reliable than config.json for URL format)
    m = re.search(r"git clone (https://github\.com/[^\s]+?)(?:\.git)?\s+/testbed", df_text)
    if m:
        # Capture group already excludes .git suffix; use as-is
        repo_url = m.group(1).rstrip("/")
    elif repo:
        repo_url = f"https://github.com/{repo}"
    else:
        result["status"] = "skip_no_repo"
        return result

    if not base_commit:
        # Try to extract from Dockerfile
        m2 = re.search(r"git checkout ([a-f0-9]{7,40})", df_text)
        base_commit = m2.group(1) if m2 else ""
    if not base_commit:
        result["status"] = "skip_no_commit"
        return result

    python_version = extract_python_version(df_text)
    result["python_version"] = python_version

    # Check for COPY/ADD in post-workdir (complex edge case)
    def get_post_workdir(text: str) -> str:
        mt = re.search(r"^WORKDIR\s+/testbed[/\s]*$", text, re.MULTILINE)
        return text[mt.end():].strip() if mt else ""

    post = get_post_workdir(df_text)
    if re.search(r"^(COPY|ADD)\s", post, re.MULTILINE):
        result["notes"].append("has_copy_or_add")

    # --- Generate new file contents ---
    new_dockerfile = make_base_dockerfile(python_version)
    setup_sh = translate_dockerfile_to_shell(post, repo_url, base_commit)

    instruction_path = src_dir / "instruction.md"
    original_instruction = instruction_path.read_text() if instruction_path.exists() else ""
    new_instruction = INSTRUCTION_PREAMBLE + original_instruction

    solve_path = src_dir / "solution" / "solve.sh"
    original_solve = solve_path.read_text() if solve_path.exists() else ""
    new_solve = wrap_solve_sh(original_solve) if original_solve else ""

    if dry_run:
        return result

    # --- Write output ---
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy entire task tree first, then overwrite changed files
    if dst_dir != src_dir:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    # Overwrite Dockerfile
    (dst_dir / "environment" / "Dockerfile").write_text(new_dockerfile)

    # Write setup_files/setup.sh
    setup_files_dir = dst_dir / "setup_files"
    setup_files_dir.mkdir(exist_ok=True)
    setup_sh_path = setup_files_dir / "setup.sh"
    setup_sh_path.write_text(setup_sh)
    setup_sh_path.chmod(setup_sh_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # Overwrite instruction.md
    (dst_dir / "instruction.md").write_text(new_instruction)

    # Overwrite solution/solve.sh
    if new_solve and solve_path.exists():
        new_solve_path = dst_dir / "solution" / "solve.sh"
        new_solve_path.write_text(new_solve)
        new_solve_path.chmod(new_solve_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch openswe-harbor-tasks: reduce 45K snapshots → 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory containing openswe task subdirs",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (defaults to input_dir + '_patched')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N tasks (for smoke tests)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir or Path(str(input_dir) + "_patched")
    output_dir = output_dir.expanduser().resolve()

    task_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"[patch_openswe] {len(task_dirs)} tasks  {input_dir} → {output_dir}")
    if args.dry_run:
        print("[patch_openswe] DRY RUN — no files will be written")

    stats: dict[str, int] = {"ok": 0, "skip_no_dockerfile": 0, "skip_no_repo": 0,
                              "skip_no_commit": 0, "has_copy_or_add": 0}
    py_version_counts: dict[str, int] = {}

    from tqdm import tqdm
    for td in tqdm(task_dirs, unit="task"):
        dst = output_dir / td.name if not args.dry_run else td
        result = patch_task(td, dst, dry_run=args.dry_run)

        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        if "has_copy_or_add" in result.get("notes", []):
            stats["has_copy_or_add"] += 1
        if result["python_version"]:
            pv = result["python_version"]
            py_version_counts[pv] = py_version_counts.get(pv, 0) + 1

        if args.verbose or status != "ok":
            print(f"  {td.name}: {status}  py={result['python_version']}  {result.get('notes','')}")

    print("\n[patch_openswe] Done.")
    print(f"  Status counts: {stats}")
    print(f"  Python version distribution: {dict(sorted(py_version_counts.items()))}")
    print(f"\nNext: verify snapshot count with:")
    print(f"  python -m scripts.harbor.count_snapshots_from_tasks --local-dataset {output_dir}")


if __name__ == "__main__":
    main()
