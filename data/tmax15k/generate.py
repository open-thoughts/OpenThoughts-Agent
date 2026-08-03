#!/usr/bin/env python3
"""
Convert allenai/TMax-15K into Harbor-compatible tasks (snapshot-safe).

TMax-15K is a ~15k synthetic terminal/RL-environment dataset. Each record ships:
  - description        : the agent-facing problem  -> instruction.md
  - container_def      : a Singularity/Apptainer def (Bootstrap: docker / From: ubuntu:22.04)
                         whose %post builds the task environment (apt + pip + heredoc
                         files + useradd + git repos) -> shared Dockerfile + setup_files/setup.sh
  - test_final_state   : a pytest module that verifies the solution -> tests/test_state.py
  - test_initial_state : pytest preconditions (kept in metadata for reference)
  - truth              : a setup-script + EXPECTED-RESULTS description (NOT an agent
                         solution) -> kept in metadata as an oracle-teacher hint
  - language/domain/.. : metadata

SNAPSHOT SAFETY (critical):
  Harbor/Daytona hashes the ENTIRE environment/ directory (Dockerfile + fixtures) to key
  snapshots (harbor.utils.container_cache.get_task_environment_hash). The original
  container_def bakes per-task files into the build => one snapshot per task (unlaunchable).
  Instead we render ONE shared Dockerfile (apt union + pytest) for every task, and move the
  per-task %post body into a setup.sh that runs at TRIAL time. => 1 unique environment/ => 1 snapshot.

HARBOR MOUNT CONTRACT (the real one — verified against the harbor source):
  Harbor mounts ONLY these per-task dirs into the container:
    - environment/ -> baked into the Docker image at build (Dockerfile tasks); NOT re-uploaded.
    - solution/    -> uploaded to /solution by the OracleAgent (oracle phase only).
    - tests/       -> uploaded to /tests by the verifier (verify phase only).
  There is NO /setup_files mount (the TaskPaths docstring advertises one, but nothing in the
  codebase wires it up). So the per-task env-prep script (setup.sh) must reach each phase via a
  real mount:
    - AGENT phase:    setup.sh is embedded INLINE in instruction.md; the agent runs it first.
    - ORACLE phase:   setup.sh is shipped as solution/setup.sh (uploaded with /solution);
                      solution/solve.sh runs `bash /solution/setup.sh` before solving.
    - VERIFIER phase: setup.sh is shipped as tests/setup.sh (uploaded with /tests);
                      tests/test.sh runs `bash /tests/setup.sh` before pytest (recreates the
                      ground-truth fixtures the test compares against).
  setup.sh is idempotent (sentinel) so re-running across phases is safe.

ORACLE:
  TMax has no gold agent solution (`truth` is env-setup, not a solution). solution/solve.sh
  is left as a placeholder here; Stage 4 generates it with a teacher
  (data/patchers/synthesize_oracle_solvers.py) and the oracle gate filters non-verifying ones.

Filtering:
  Records that reference external %files fixtures (/gpfs/scrubbed/... = ~35%) are UNBUILDABLE
  for us and are skipped. The clean buildable subset is ~9465 records.

Usage:
    PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
    # smoke (5 tasks, local only):
    $PY -m data.tmax15k.generate --limit 5 --output-dir /tmp/tmax15k_smoke
    # full (no cap) + upload:
    $PY -m data.tmax15k.generate --limit 0 \
        --output-dir /tmp/tmax15k_raw \
        --target-repo laion/tmax15k-tasks-raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

from datasets import load_dataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.commons import (  # type: ignore  # pylint: disable=wrong-import-position
    create_task_directory_unified,
    upload_tasks_to_hf,
)

if __package__ in (None, ""):
    DATASET_NAME = "allenai/TMax-15K"
else:
    from . import DATASET_NAME


# --------------------------------------------------------------------------- #
# Shared environment (ONE Dockerfile for all tasks -> 1 snapshot)
#
# Union of the most common apt/pip deps observed across the clean subset, so the
# vast majority of tasks need no extra runtime installs. setup.sh still runs the
# original task-specific `apt-get install` line as a safety net for the long tail.
# --------------------------------------------------------------------------- #

SHARED_DOCKERFILE = dedent(
    """\
    FROM ubuntu:22.04

    ENV DEBIAN_FRONTEND=noninteractive
    ENV LANG=C.UTF-8
    WORKDIR /app

    # Union of the common system deps across TMax tasks. Per-task setup.sh installs
    # anything rare that is missing (apt update cache is preserved below).
    RUN apt-get update && apt-get install -y --no-install-recommends \\
            bash coreutils findutils grep sed gawk \\
            python3 python3-pip python3-venv python3-dev \\
            gcc g++ make cmake build-essential binutils libc6-dev gdb valgrind strace \\
            golang-go cargo rustc \\
            git curl wget ca-certificates openssh-client \\
            sqlite3 libsqlite3-dev \\
            jq bc xxd file patch sudo \\
            tar gzip zip unzip \\
            procps psmisc cron logrotate \\
            netcat-openbsd socat expect tcpdump iproute2 \\
            openssl libssl-dev \\
            tzdata locales \\
        && rm -rf /var/lib/apt/lists/*

    RUN pip3 install --no-cache-dir \\
            pytest pytest-timeout hypothesis \\
            numpy pandas scipy scikit-learn networkx \\
            requests jsonschema packaging setuptools

    WORKDIR /app
    """
)


# --------------------------------------------------------------------------- #
# Instruction preamble: tell the agent the env is materialized by setup.sh.
# --------------------------------------------------------------------------- #

def build_instruction(description: str, setup_sh: str) -> str:
    """Agent-facing instruction with the environment-setup script INLINED.

    The agent has no mounted setup file (Harbor only mounts environment/ -> image,
    solution/ -> /solution at oracle time, tests/ -> /tests at verify time). So the
    setup script that materializes this task's input files is embedded here for the
    agent to run first. The verifier runs the identical (idempotent) script before
    grading, so the state it checks is exactly the one this script produces.
    """
    return (
        "**Environment setup (run this FIRST).** This task's input files are created\n"
        "by the setup script below. Save it verbatim and run it before you start — it\n"
        "is idempotent (safe to run once). The verifier runs the identical script\n"
        "before grading, so the state it checks is exactly the one this produces.\n"
        "\n"
        "```bash\n"
        "cat > /tmp/task_setup.sh <<'TMAX_SETUP_EOF'\n"
        + setup_sh.rstrip("\n")
        + "\n"
        "TMAX_SETUP_EOF\n"
        "bash /tmp/task_setup.sh\n"
        "```\n"
        "\n"
        "---\n\n"
        + description.strip()
        + "\n"
    )


# --------------------------------------------------------------------------- #
# container_def (%post) -> setup_files/setup.sh
# --------------------------------------------------------------------------- #

# apt packages already present in the shared image; drop them from the runtime
# `apt-get install` to avoid needless re-installs (the line still runs for the
# long tail of rare packages).
BASE_APT = {
    "bash", "coreutils", "findutils", "grep", "sed", "gawk",
    "python3", "python3-pip", "python3-venv", "python3-dev",
    "gcc", "g++", "make", "cmake", "build-essential", "binutils", "libc6-dev",
    "gdb", "valgrind", "strace", "golang-go", "golang", "cargo", "rustc",
    "git", "curl", "wget", "ca-certificates", "openssh-client",
    "sqlite3", "libsqlite3-dev", "jq", "bc", "xxd", "file", "patch", "sudo",
    "tar", "gzip", "zip", "unzip", "procps", "psmisc", "cron", "logrotate",
    "netcat-openbsd", "socat", "expect", "tcpdump", "iproute2",
    "openssl", "libssl-dev", "tzdata", "locales",
}


def _extract_post_body(container_def: str) -> str:
    """Return the %post body of a Singularity def, sans the leading boilerplate.

    We keep apt-get/pip3 lines (so rare deps still install at runtime) but strip
    `export DEBIAN_FRONTEND` (set in the image) and re-de-indent.
    """
    lines = container_def.splitlines()
    # find %post .. next %section
    body: list[str] = []
    in_post = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("%post"):
            in_post = True
            continue
        if in_post and stripped.startswith("%") and not stripped.startswith("%post"):
            in_post = False
            continue
        if in_post:
            body.append(ln)
    # de-indent: drop a common leading indentation (Singularity %post is indented)
    text = "\n".join(body)
    # compute min indent of non-empty lines
    nonempty = [l for l in body if l.strip()]  # noqa: E741
    if nonempty:
        indents = [len(l) - len(l.lstrip(" ")) for l in nonempty]  # noqa: E741
        common = min(indents)
        if common:
            text = "\n".join(l[common:] if len(l) >= common else l for l in body)  # noqa: E741
    return text


def _extract_environment_exports(container_def: str) -> list[str]:
    """Pull `export VAR=...` lines from a %environment section (rare, ~134 tasks)."""
    out: list[str] = []
    in_env = False
    for ln in container_def.splitlines():
        s = ln.strip()
        if s.startswith("%environment"):
            in_env = True
            continue
        if in_env and s.startswith("%"):
            in_env = False
            continue
        if in_env and s.startswith("export "):
            out.append(s)
    return out


def build_setup_script(container_def: str) -> str:
    """Render an idempotent setup.sh from a task's container_def %post body."""
    post = _extract_post_body(container_def)
    env_exports = _extract_environment_exports(container_def)

    # Trim a redundant `export DEBIAN_FRONTEND` (we set it in the env via the call).
    post_lines = []
    for ln in post.splitlines():
        if re.match(r"\s*export\s+DEBIAN_FRONTEND", ln):
            continue
        post_lines.append(ln)
    post = "\n".join(post_lines)

    env_block = "\n".join(env_exports)

    header = dedent(
        """\
        #!/usr/bin/env bash
        # Idempotent task-environment setup, factored out of the TMax container_def %post
        # so the Harbor environment/ dir (and thus the Daytona snapshot) stays shared.
        # Re-running is safe: a sentinel short-circuits the second invocation.
        # Runs at trial time in EVERY phase (agent via instruction.md, oracle via
        # /solution/setup.sh, verifier via /tests/setup.sh) -> all phases see the same
        # materialized fixtures + ground truth.
        set -e
        export DEBIAN_FRONTEND=noninteractive

        SENTINEL="/tmp/.tmax_setup_done"
        if [ -f "$SENTINEL" ]; then
            exit 0
        fi

        # Trust any git repo this setup creates (containers run as root but the repo
        # may be chowned to `user`; avoids "detected dubious ownership" failures that
        # break otherwise-correct solutions).
        git config --global --add safe.directory '*' 2>/dev/null || true
        """
    )
    if env_block:
        header += "\n# --- %environment exports ---\n" + env_block + "\n"

    footer = '\ntouch "$SENTINEL" 2>/dev/null || true\n'

    return header + "\n# --- begin original %post body ---\n" + post + "\n# --- end original %post body ---\n" + footer


# --------------------------------------------------------------------------- #
# tests/test.sh : run setup.sh, then pytest the final-state suite -> reward.txt
# --------------------------------------------------------------------------- #

def build_test_sh() -> str:
    return dedent(
        """\
        #!/usr/bin/env bash
        set -uo pipefail

        LOG_DIR="/logs"
        VERIFIER_DIR="/logs/verifier"
        REWARD_FILE="$VERIFIER_DIR/reward.txt"
        mkdir -p "$LOG_DIR" "$VERIFIER_DIR"
        : > "$REWARD_FILE"

        # Materialize the task environment + ground truth (idempotent). setup.sh
        # ships inside tests/ so it's mounted at /tests alongside this script. The
        # agent ran an identical copy (embedded in instruction.md); the sentinel makes
        # a re-run a no-op, so the ground-truth fixtures the tests compare against are
        # the ones this script produces.
        SETUP_SH="$(dirname "$0")/setup.sh"
        if [ -f "$SETUP_SH" ]; then
            bash "$SETUP_SH" || {
                echo "[tmax] setup.sh failed" >&2
                echo 0 > "$REWARD_FILE"
                exit 1
            }
        fi

        # The pytest verification module lives next to this script.
        TEST_PY="$(dirname "$0")/test_state.py"
        if [ ! -f "$TEST_PY" ]; then
            echo "[tmax] test_state.py missing" >&2
            echo 0 > "$REWARD_FILE"
            exit 1
        fi

        if python3 -m pytest -q "$TEST_PY"; then
            echo 1 > "$REWARD_FILE"
            exit 0
        else
            echo 0 > "$REWARD_FILE"
            exit 1
        fi
        """
    )


# Placeholder solver — replaced in Stage 4 by the teacher (synthesize_oracle_solvers.py).
PLACEHOLDER_SOLVE = dedent(
    """\
    #!/usr/bin/env bash
    # PLACEHOLDER oracle. Generated in Stage 4 by a teacher and filtered by the oracle gate.
    set -e
    if [ -f /solution/setup.sh ]; then bash /solution/setup.sh; fi
    """
)


TASK_TOML = dedent(
    """\
    version = "1.0"

    [metadata]
    author_name = "tmax15k-converter"
    author_email = "research@ot-agent.invalid"
    difficulty = "{difficulty}"
    category = "{category}"
    tags = {tags}

    [verifier]
    restart_environment = false
    timeout_sec = 1200.0

    [agent]
    timeout_sec = 1200.0
    """
)


# Map TMax task_complexity -> a coarse difficulty label.
def _difficulty(task_complexity: str) -> str:
    tc = (task_complexity or "").lower()
    if tc.startswith("short"):
        return "easy"
    if tc.startswith("moderate"):
        return "medium"
    if tc.startswith("complex"):
        return "hard"
    if tc.startswith("intricate"):
        return "very-hard"
    return "medium"


def _toml_tags(record: dict) -> str:
    tags = ["tmax15k", "synthetic", "terminal"]
    for k in ("domain", "skill_type", "language"):
        v = (record.get(k) or "").strip()
        if v and len(v) < 40:
            tags.append(re.sub(r"[^a-zA-Z0-9_.-]", "_", v.lower()))
    return json.dumps(tags)


def _is_buildable(record: dict) -> bool:
    """Skip records that need external %files fixtures we do not have."""
    cd = record.get("container_def") or ""
    if "%files" in cd:
        return False
    if "/gpfs/" in cd or "scrubbed" in cd:
        return False
    if not record.get("description", "").strip():
        return False
    if not record.get("test_final_state", "").strip():
        return False
    if "def test_" not in record.get("test_final_state", ""):
        return False
    return True


# --------------------------------------------------------------------------- #
# FULL-SET regeneration from a Harbor-Hub export (data/tmax15k.generate --from-hub-export)
#
# The `_is_buildable()` gate above drops the ~5,136 records whose container_def
# references external `%files` fixtures (e.g. /gpfs/scrubbed/.../image.png) — those
# files are NOT present in the raw allenai/TMax-15K dataset, so generate.py's source
# path tops out at ~9,465 buildable tasks. The Harbor-Hub export `tmax/TMax-15K-Harbor`
# (12,926 tasks) DID ship those fixtures (under environment/_fixtures/), so it is the
# only source for the FULL unfiltered set. That export is a STALE per-task-patcher
# artifact (12,839 snapshots: each task bakes its post_install.sh + _fixtures into
# environment/). This mode re-renders each hub task through the CURRENT
# 1-shared-Dockerfile contract — base_install (shared, identical across all tasks) +
# post_install + base64-restored fixtures move into the runtime setup.sh (inlined in
# instruction.md, shipped in tests/ + solution/), leaving environment/ = one shared
# Dockerfile => 1 snapshot, full 12,926 tasks, no solution/quality filter.
# --------------------------------------------------------------------------- #

import base64  # noqa: E402
import shlex  # noqa: E402


def _hub_fixtures_block(env_dir: Path) -> str:
    """Restore environment/_fixtures/<copy-dest> files into the container at setup time.

    The hub Dockerfile COPYs each fixture from _fixtures/<relpath> to a dest path. We
    re-derive (relpath -> dest) from the Dockerfile's `COPY _fixtures/...` lines and
    base64-embed each (small: max ~20KB, ~30MB total) so a single self-contained
    setup.sh recreates them — no environment/ fixture => no per-task snapshot hash.
    """
    dockerfile = env_dir / "Dockerfile"
    fixtures_root = env_dir / "_fixtures"
    if not dockerfile.exists() or not fixtures_root.is_dir():
        return ""
    blocks: list[str] = []
    for ln in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s.startswith("COPY _fixtures/"):
            continue
        parts = shlex.split(s)
        if len(parts) < 3:
            continue
        src_rel = parts[1][len("_fixtures/"):]  # path under _fixtures/
        dest = parts[2]
        src_path = fixtures_root / src_rel
        if not src_path.is_file():
            continue
        b64 = base64.b64encode(src_path.read_bytes()).decode("ascii")
        dest_q = shlex.quote(dest)
        blocks.append(
            f"mkdir -p \"$(dirname {dest_q})\"\n"
            f"base64 -d > {dest_q} <<'TMAX_FIXTURE_EOF'\n{b64}\nTMAX_FIXTURE_EOF"
        )
    if not blocks:
        return ""
    return "\n# --- restore external fixtures (was COPY _fixtures/...) ---\n" + "\n".join(blocks) + "\n"


def _build_hub_setup_script(task_dir: Path, base_install: str) -> str:
    """Render an idempotent setup.sh from a hub task's base_install + post_install + fixtures."""
    env_dir = task_dir / "environment"
    post = (env_dir / "post_install.sh").read_text(encoding="utf-8", errors="replace") if (env_dir / "post_install.sh").exists() else ""
    # Drop a redundant `export DEBIAN_FRONTEND` (set in the shared image env).
    post = "\n".join(l for l in post.splitlines() if not re.match(r"\s*export\s+DEBIAN_FRONTEND", l))  # noqa: E741
    fixtures = _hub_fixtures_block(env_dir)

    header = dedent(
        """\
        #!/usr/bin/env bash
        # Idempotent task-environment setup, factored out of the TMax Harbor-Hub export
        # (base_install + post_install + _fixtures) so the Harbor environment/ dir (and
        # thus the Daytona snapshot) stays shared across all tasks. Re-running is safe:
        # a sentinel short-circuits the second invocation. Runs at trial time in EVERY
        # phase (agent via instruction.md, oracle via /solution/setup.sh, verifier via
        # /tests/setup.sh) -> all phases see the same materialized fixtures + ground truth.
        set -e
        export DEBIAN_FRONTEND=noninteractive

        SENTINEL="/tmp/.tmax_setup_done"
        if [ -f "$SENTINEL" ]; then
            exit 0
        fi

        git config --global --add safe.directory '*' 2>/dev/null || true
        """
    )
    footer = '\ntouch "$SENTINEL" 2>/dev/null || true\n'
    # base_install is the SHARED base; it is already baked into SHARED_DOCKERFILE_FULL,
    # so we do NOT re-run it at trial time (it is a long apt/pip/torch install). Only
    # the per-task post_install + fixtures run here.
    return (
        header
        + fixtures
        + "\n# --- begin original post_install body ---\n"
        + post
        + "\n# --- end original post_install body ---\n"
        + footer
    )


# Shared Dockerfile for the FULL set: bakes the hub's shared base_install (the union
# apt/pip/torch base every hub task was built against) so the runtime setup.sh only has
# to run each task's post_install + fixtures. One Dockerfile string for all tasks.
def _shared_dockerfile_full(base_install: str) -> str:
    base_install.rstrip("\n")
    return (
        "FROM ubuntu:22.04\n\n"
        "ENV DEBIAN_FRONTEND=noninteractive\n"
        "ENV LC_ALL=C.UTF-8\n"
        "ENV LANG=C.UTF-8\n"
        "ENV GOPATH=/home/user/go\n"
        "ENV PATH=/home/user/go/bin:/home/user/.cargo/bin:$PATH\n"
        "WORKDIR /app\n\n"
        "# Shared base install (identical across all TMax-15K-Harbor tasks).\n"
        "COPY base_install.sh /tmp/base_install.sh\n"
        "RUN bash /tmp/base_install.sh && rm /tmp/base_install.sh\n\n"
        "WORKDIR /app\n"
    )


def convert_from_hub_export(
    hub_dir: Path,
    output_dir: Path,
    limit: int = 0,
) -> tuple[int, int]:
    """Re-render a Harbor-Hub TMax-15K export into the shared-Dockerfile (1-snapshot) layout.

    Includes EVERY hub task (the full ~12,926 set) — NO solution/quality filter, NO
    `%files`/buildability drop (the fixtures ship with the export and are restored at
    trial time). solution/solve.sh is the placeholder (the full set intentionally
    includes tasks that lack a real oracle solution).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_subdirs = sorted(
        d for d in hub_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )

    # The shared base_install.sh is identical across all hub tasks; grab one copy.
    base_install = ""
    for d in task_subdirs:
        bi = d / "environment" / "base_install.sh"
        if bi.exists():
            base_install = bi.read_text(encoding="utf-8", errors="replace")
            break
    shared_dockerfile = _shared_dockerfile_full(base_install)

    made = 0
    skipped = 0
    idx = 0
    for d in task_subdirs:
        instruction_path = d / "instruction.md"
        test_py = d / "tests" / "test_final_state.py"
        if not test_py.exists():
            skipped += 1
            continue
        description = instruction_path.read_text(encoding="utf-8", errors="replace")
        setup_sh = _build_hub_setup_script(d, base_install)
        instruction = build_instruction(description, setup_sh)
        test_state = test_py.read_text(encoding="utf-8", errors="replace")

        # Preserve the hub's metadata from its task.toml (best-effort) + the source id.
        metadata = {
            "source": "tmax/TMax-15K-Harbor",
            "hub_task_dir": d.name,
        }
        toml = TASK_TOML.format(difficulty="medium", category="terminal",
                                tags=json.dumps(["tmax15k", "synthetic", "terminal", "full"]))

        task_dir = create_task_directory_unified(
            output_dir=output_dir,
            task_id=idx,
            instruction_content=instruction,
            dataset_prefix="tmax15k",
            metadata=metadata,
            solution_content=PLACEHOLDER_SOLVE,
            test_sh_content=build_test_sh(),
            test_py_content=None,
            task_toml_content=toml,
            dockerfile_content=shared_dockerfile,
        )

        # Ship the shared base_install.sh inside environment/ (referenced by the shared
        # Dockerfile COPY). It is identical for every task, so it does NOT change the
        # environment/ content hash across tasks => still 1 snapshot.
        (task_dir / "environment" / "base_install.sh").write_text(base_install, encoding="utf-8")

        (task_dir / "tests" / "test_state.py").write_text(test_state, encoding="utf-8")
        for sub in ("tests", "solution"):
            sp = task_dir / sub / "setup.sh"
            sp.parent.mkdir(exist_ok=True)
            sp.write_text(setup_sh, encoding="utf-8")
            sp.chmod(0o755)

        made += 1
        idx += 1
        if limit and made >= limit:
            break

    return made, skipped


def convert(
    output_dir: Path,
    limit: int = 0,
    split: str = "train",
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_NAME)[split]

    made = 0
    skipped = 0
    idx = 0
    for record in ds:
        if not _is_buildable(record):
            skipped += 1
            continue

        setup_sh = build_setup_script(record["container_def"])
        instruction = build_instruction(record["description"], setup_sh)
        test_sh = build_test_sh()
        test_state = record["test_final_state"]

        metadata = {
            "source": DATASET_NAME,
            "task_id": record.get("task_id"),
            "domain": record.get("domain"),
            "skill_type": record.get("skill_type"),
            "primitive_skills": record.get("primitive_skills"),
            "task_complexity": record.get("task_complexity"),
            "command_complexity": record.get("command_complexity"),
            "scenario": record.get("scenario"),
            "language": record.get("language"),
            # `truth` is env-setup + EXPECTED RESULTS — fed to the oracle teacher as a hint.
            "truth_hint": record.get("truth"),
            "test_initial_state": record.get("test_initial_state"),
        }

        toml = TASK_TOML.format(
            difficulty=_difficulty(record.get("task_complexity", "")),
            category=re.sub(r"[^a-zA-Z0-9_-]", "-", (record.get("domain") or "terminal").lower()),
            tags=_toml_tags(record),
        )

        task_dir = create_task_directory_unified(
            output_dir=output_dir,
            task_id=idx,
            instruction_content=instruction,
            dataset_prefix="tmax15k",
            metadata=metadata,
            solution_content=PLACEHOLDER_SOLVE,
            test_sh_content=test_sh,
            test_py_content=None,  # we write test_state.py ourselves (verbatim TMax pytest)
            task_toml_content=toml,
            dockerfile_content=SHARED_DOCKERFILE,
        )

        # tests/test_state.py = the verbatim TMax final-state pytest module.
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test_state.py").write_text(test_state, encoding="utf-8")

        # setup.sh ships in BOTH tests/ (mounted -> /tests for the verifier) and
        # solution/ (mounted -> /solution for the oracle), since Harbor has no
        # /setup_files mount. The agent runs an inline copy from instruction.md.
        for sub in ("tests", "solution"):
            d = task_dir / sub
            d.mkdir(exist_ok=True)
            sp = d / "setup.sh"
            sp.write_text(setup_sh, encoding="utf-8")
            sp.chmod(0o755)

        made += 1
        idx += 1
        if limit and made >= limit:
            break

    return made, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Convert TMax-15K into Harbor tasks")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0, help="<=0 means no cap (max size)")
    p.add_argument("--split", default="train")
    p.add_argument("--target-repo", default=None, help="HF repo to upload to (laion/...)")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--no-upload", action="store_true")
    p.add_argument(
        "--from-hub-export",
        type=Path,
        default=None,
        help=(
            "Regenerate the FULL unfiltered set (~12,926 tasks) from a local "
            "Harbor-Hub TMax-15K-Harbor export dir, re-rendered through the current "
            "1-shared-Dockerfile contract. Bypasses the source-path _is_buildable "
            "%files gate (fixtures ship with the export); NO solution/quality filter."
        ),
    )
    args = p.parse_args()

    if args.from_hub_export is not None:
        made, skipped = convert_from_hub_export(
            args.from_hub_export, args.output_dir, limit=args.limit
        )
        print(f"Regenerated {made} FULL-set tasks (skipped {skipped} no-test) "
              f"from hub export {args.from_hub_export} -> {args.output_dir}")
    else:
        made, skipped = convert(args.output_dir, limit=args.limit, split=args.split)
        print(f"Converted {made} tasks (skipped {skipped} unbuildable) -> {args.output_dir}")

    if args.target_repo and not args.no_upload:
        url = upload_tasks_to_hf(
            dataset_path=str(args.output_dir),
            repo_id=args.target_repo,
            private=False,
            token=args.hf_token,
            commit_message=f"TMax-15K -> Harbor tasks ({made} tasks)",
        )
        print(f"Uploaded to {url}")


if __name__ == "__main__":
    main()
