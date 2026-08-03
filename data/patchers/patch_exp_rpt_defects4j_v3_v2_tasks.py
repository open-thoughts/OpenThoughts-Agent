#!/usr/bin/env python3
"""
exp_rpt_defects4j v3-v2 -> v3-v3 patcher (uploaded as
`laion/exp_rpt_defects4j-v3-v3`).

Bug (found in QC 2026-05-15):
  v3-v2's headline solve rate is 2/200 (1.0%) on a `gpt-5-nano +
  terminus-2 + max_episodes=1` validation run. The v2 patch correctly
  killed the rubber-stamp (no auto-fallback `cp` of the buggy initial)
  and both reported solves are real (4 distinct JUnit tests pass).
  But QC trace inspection reveals a *new* defect that systematically
  blocks single-turn fix attempts:

  Defect 3 (instruction lies about file location):
    The Dockerfile creates `/app/` containing only an empty `classes/`
    subdir. There is NO `Solution.java` at `/app/Solution.java` at
    container start — v1 used to silently `cp` it in from
    `/tests/initial/Solution.java` (which was the rubber-stamp); v2
    correctly removed that `cp` but did NOT also fix the instruction
    text, which still tells the agent:

        "The buggy file is located at `/app/Solution.java`."

    So the agent's first command (`sed -n '1,200p' /app/Solution.java`,
    or `grep ... /app/Solution.java`) fails because the file doesn't
    exist. The agent has to (a) know to look under `/tests/initial/`,
    OR (b) realize the file is missing and recover. Either way, the
    instruction is *wrong* about the starting state. In the 200-trial
    QC:
      - 190/198 reward=0 trials triggered Gate A ("agent did not write
        /app/Solution.java"), i.e. the agent never produced any file
        at the path the instruction promised.
      - 180/198 reward=0 trials never even emitted valid JSON commands;
        the agent burned its single turn analyzing-and-planning without
        executing.
      - The 10 trials that did emit commands typically started with
        `cat`/`sed` of `/app/Solution.java`, which fails immediately.

    A multi-turn agent can recover (the v1 trace generators evidently
    did), but a 1-turn validator catches the mismatch and scores 0.
    The instruction needs to be consistent with the sandbox state.

  Two cleanest fixes considered:
    (A) Rewrite instruction.md per-task to point at `/tests/initial/
        Solution.java` and explicitly tell the agent to write the
        fixed copy to `/app/Solution.java`. Touches 304 task files,
        each with a different bug description.
    (B) Pre-stage `/app/Solution.java` with the buggy initial in the
        Dockerfile. Matches the instruction text exactly. Single-line
        Dockerfile edit per task; no per-task semantic rewrites.

  We pick (B). To preserve the v2 anti-rubber-stamp protection, Gate A
  in `test.sh` is upgraded from "file exists" to "file differs from
  the buggy initial" — i.e. the agent must have actually edited
  `/app/Solution.java`, not just left it untouched. SHA-256 hash
  comparison.

  Side note: 6 of the 304 tasks ship with `solution/Solution.java`
  byte-identical to `tests/initial/Solution.java` (`defects4j-0009,
  -0032, -0044, -0311, -0361, -0419`). Under the new Gate A these
  would be unsolvable (the "fix" IS the buggy file). They're
  degenerate even semantically — the prompt asks the agent to fix a
  bug that doesn't exist in the source. v3 DROPS these 6 tasks.

Fix (uploaded as v3-v3):
  1. REWRITE `environment/Dockerfile` for surviving tasks to add a
     `COPY tests/initial/Solution.java /app/Solution.java` line so the
     buggy file actually lives where the instruction says it does.
     This restores the v1 starting state for the agent *without* the
     v1 rubber-stamp (v2's test.sh removal of the auto-cp is kept).

  2. REWRITE `tests/test.sh` so Gate A is "agent edited
     /app/Solution.java" (sha256 differs from
     /tests/initial/Solution.java), not just "file exists". Gates B
     (compile), C (>=1 test, 0 failures), D (clean junit exit) are
     unchanged from v2.

  3. DROP the 6 tasks where solution == initial (degenerate; would be
     unsolvable under the new Gate A and confusing semantically).

  4. Idempotency: drop a `.laion_v3_patched` marker file at each
     surviving task root; write `_LAION_V3_PATCH_REPORT.txt` at
     --root with the dropped-task list. Re-runs that find the marker
     skip the task.

Constraints (per upload spec):
  - `task.toml`, `metadata.json`, `solution/*`, `tests/initial/*`, and
    `tests/TestSolution.java`, `instruction.md` are NOT touched. Only
    `environment/Dockerfile` and `tests/test.sh` are rewritten, and a
    marker file is added. Dropped tasks have their entire task dir
    removed.
  - Idempotent and deterministic.
  - The v2 marker file is preserved; the v3 marker is added on top.

Usage:
  python data/patchers/patch_exp_rpt_defects4j_v3_v2_tasks.py \
      --root /path/to/extracted-tasks [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Markers (idempotency)
# --------------------------------------------------------------------------- #

_V3_MARKER_FILE = ".laion_v3_patched"
_V3_DOCKERFILE_MARKER = "# --- laion v3 patch: pre-stage buggy initial ---"
_V3_TESTSH_MARKER = "# --- laion v3 patch: edit-gate (sha differs) ---"
_V3_REPORT_FILENAME = "_LAION_V3_PATCH_REPORT.txt"

# Dockerfile snippet to append: copies the buggy initial into /app so
# the instruction's "/app/Solution.java" claim is true at container
# start.
#
# IMPORTANT — Daytona build-context detail: Daytona's
# `Image.from_dockerfile(path)` (used by Harbor's daytona env backend at
# daytona.py:1289) resolves relative COPY sources against the
# Dockerfile's PARENT directory, not the task root. The Dockerfile lives
# at `<task>/environment/Dockerfile`, so a `COPY tests/...` source would
# resolve to `<task>/environment/tests/...` — which doesn't exist and
# would fail the build silently (the SDK falls through to a glob pattern
# of zero matches and ships an empty context entry).
#
# To make the COPY work, we ALSO drop a copy of the buggy initial at
# `<task>/environment/initial_solution_buggy.java` and reference it via
# the bare filename (resolves to the Dockerfile's parent dir = build
# context). The on-disk verifier still reads `/tests/initial/
# Solution.java` from inside the container (mounted by Harbor at trial
# time), which is unaffected.
_V3_INITIAL_COPY_FILENAME = "initial_solution_buggy.java"

_V3_DOCKERFILE_APPEND = f"""
{_V3_DOCKERFILE_MARKER}
# v3 fix: pre-stage the buggy initial Solution.java at /app/Solution.java
# so the instruction's "The buggy file is located at /app/Solution.java"
# is true at container start. The v2 test.sh Gate A is upgraded to
# require the file's sha256 to differ from /tests/initial/Solution.java
# (i.e. the agent must have edited it), which preserves the anti-
# rubber-stamp protection without lying to the agent about where the
# file lives. The source file (`{_V3_INITIAL_COPY_FILENAME}`) is a
# byte-identical copy of `tests/initial/Solution.java`, dropped into the
# same dir as this Dockerfile so it lands in Daytona's build context.
COPY {_V3_INITIAL_COPY_FILENAME} /app/Solution.java
"""

# --------------------------------------------------------------------------- #
# New tests/test.sh.
#
# Semantics vs. v2:
#   - Gate A becomes "the agent edited the file" (sha256 differs from
#     /tests/initial/Solution.java) instead of "the file exists". The
#     v3 Dockerfile pre-stages the buggy initial at the path the
#     instruction promises, so "file exists" is now trivially true
#     even when the agent does nothing — we need a stronger gate to
#     keep the rubber-stamp dead.
#   - Gates B (compile), C (junit-summary), D (junit exit) unchanged.
#   - Reward floor 0, only upgraded after all gates pass.
#   - Still no `trap cleanup EXIT` based scoring.
# --------------------------------------------------------------------------- #
NEW_TEST_SH = f"""#!/bin/bash
{_V3_TESTSH_MARKER}
# v3 verifier: same gates as v2 (compile, junit summary, clean junit
# exit) but Gate A now requires the agent to have EDITED
# /app/Solution.java (sha256 differs from the buggy initial), not just
# "file exists". v3 Dockerfile pre-stages the buggy initial at
# /app/Solution.java so the instruction is consistent with the sandbox
# state; this gate stops "agent did nothing" from passing.

set +e  # we manage exit/reward explicitly; don't abort on first failure

mkdir -p /logs/verifier
# Reward floor: any abnormal exit below leaves this 0 in place.
echo "0" > /logs/verifier/reward.txt

cd /app

# Gate A: the agent must have EDITED /app/Solution.java. The v3
# Dockerfile pre-stages the buggy initial there so the file always
# exists at /app/Solution.java; "agent did nothing" leaves it
# byte-identical to /tests/initial/Solution.java, which we reject.
if [ ! -f /app/Solution.java ]; then
    echo "FAIL: /app/Solution.java missing (Dockerfile pre-stage failed?)" \\
        | tee /logs/verifier/test_output.txt
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi
if [ ! -f /tests/initial/Solution.java ]; then
    echo "FAIL: /tests/initial/Solution.java missing (task data corrupt?)" \\
        | tee /logs/verifier/test_output.txt
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi
agent_sha=$(sha256sum /app/Solution.java | awk '{{print $1}}')
init_sha=$(sha256sum /tests/initial/Solution.java | awk '{{print $1}}')
if [ "$agent_sha" = "$init_sha" ]; then
    echo "FAIL: /app/Solution.java identical to /tests/initial/Solution.java" \\
         "(agent did not edit the file)" \\
        | tee /logs/verifier/test_output.txt
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

echo "=== Compiling solution and tests ==="
mkdir -p /app/classes
javac -cp /junit/junit-platform-console-standalone.jar:/app:/tests \\
    /app/Solution.java /tests/TestSolution.java -d /app/classes 2>&1 \\
    | tee /logs/verifier/test_output.txt
javac_rc=${{PIPESTATUS[0]}}

if [ "$javac_rc" -ne 0 ]; then
    echo "COMPILATION FAILED" | tee -a /logs/verifier/test_output.txt
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

echo "=== Running JUnit tests ==="
java -jar /junit/junit-platform-console-standalone.jar \\
    --class-path /app/classes \\
    --scan-class-path 2>&1 | tee -a /logs/verifier/test_output.txt
junit_rc=${{PIPESTATUS[0]}}

tests_started=$(grep -oE '\\[ *[0-9]+ tests started *\\]' /logs/verifier/test_output.txt \\
    | tail -1 | grep -oE '[0-9]+' | head -1)
tests_failed=$(grep -oE '\\[ *[0-9]+ tests failed *\\]' /logs/verifier/test_output.txt \\
    | tail -1 | grep -oE '[0-9]+' | head -1)
containers_failed=$(grep -oE '\\[ *[0-9]+ containers failed *\\]' /logs/verifier/test_output.txt \\
    | tail -1 | grep -oE '[0-9]+' | head -1)

tests_started=${{tests_started:-0}}
tests_failed=${{tests_failed:-0}}
containers_failed=${{containers_failed:-0}}

echo "v3 verifier: junit_rc=$junit_rc tests_started=$tests_started" \\
     "tests_failed=$tests_failed containers_failed=$containers_failed" \\
    | tee -a /logs/verifier/test_output.txt

if [ "$junit_rc" -eq 0 ] \\
        && [ "$tests_started" -ge 1 ] \\
        && [ "$tests_failed" -eq 0 ] \\
        && [ "$containers_failed" -eq 0 ]; then
    echo "ALL TESTS PASSED" | tee -a /logs/verifier/test_output.txt
    echo "1" > /logs/verifier/reward.txt
    exit 0
else
    echo "TESTS FAILED (junit_rc=$junit_rc tests_started=$tests_started" \\
         "tests_failed=$tests_failed containers_failed=$containers_failed)" \\
        | tee -a /logs/verifier/test_output.txt
    echo "0" > /logs/verifier/reward.txt
    if [ "$junit_rc" -ne 0 ]; then
        exit "$junit_rc"
    fi
    exit 1
fi
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_bytes(p: Path) -> bytes | None:
    try:
        return p.read_bytes()
    except OSError:
        return None


def _patch_one_task(task_dir: Path, dry_run: bool) -> str:
    """Patch a single task dir. Returns one of:
        'dropped_sol_eq_initial'
        'patched'
        'already_patched'
        'no_dockerfile'
        'no_test_sh'
        'no_initial_solution'
        'no_gold_solution'
    `task_dir` may be deleted entirely on the drop path.
    """
    marker = task_dir / _V3_MARKER_FILE
    if marker.exists():
        return "already_patched"

    dockerfile = task_dir / "environment" / "Dockerfile"
    test_sh = task_dir / "tests" / "test.sh"
    initial_solution = task_dir / "tests" / "initial" / "Solution.java"
    gold_solution = task_dir / "solution" / "Solution.java"

    if not dockerfile.is_file():
        return "no_dockerfile"
    if not test_sh.is_file():
        return "no_test_sh"
    if not initial_solution.is_file():
        return "no_initial_solution"
    if not gold_solution.is_file():
        return "no_gold_solution"

    # Degenerate task drop: if gold == initial, the agent has nothing
    # to edit and would fail Gate A.
    init_bytes = _read_bytes(initial_solution)
    gold_bytes = _read_bytes(gold_solution)
    if init_bytes is not None and gold_bytes is not None and init_bytes == gold_bytes:
        if not dry_run:
            shutil.rmtree(task_dir, ignore_errors=False)
        return "dropped_sol_eq_initial"

    # Drop a copy of the buggy initial into the same dir as the
    # Dockerfile so it's inside Daytona's build context (the SDK
    # resolves COPY sources relative to the Dockerfile's parent dir).
    initial_copy = dockerfile.parent / _V3_INITIAL_COPY_FILENAME
    if not initial_copy.is_file():
        if not dry_run:
            initial_copy.write_bytes(init_bytes or b"")

    # Dockerfile: append the COPY line if it isn't already there.
    df_text = dockerfile.read_text(encoding="utf-8", errors="replace")
    if _V3_DOCKERFILE_MARKER not in df_text:
        new_df = df_text.rstrip() + "\n" + _V3_DOCKERFILE_APPEND
        if not dry_run:
            dockerfile.write_text(new_df, encoding="utf-8")

    # test.sh: rewrite if not already v3-marked.
    sh_text = test_sh.read_text(encoding="utf-8", errors="replace")
    if _V3_TESTSH_MARKER not in sh_text:
        if not dry_run:
            test_sh.write_text(NEW_TEST_SH, encoding="utf-8")
            test_sh.chmod(0o755)

    if not dry_run:
        marker.write_text(
            "laion v3 patch: pre-stage /app/Solution.java; gate via sha diff\n",
            encoding="utf-8",
        )
    return "patched"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to extracted task corpus")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    counts: dict[str, int] = {}
    dropped_names: list[str] = []

    for i, td in enumerate(task_dirs, 1):
        result = _patch_one_task(td, dry_run=args.dry_run)
        counts[result] = counts.get(result, 0) + 1
        if result == "dropped_sol_eq_initial":
            dropped_names.append(td.name)

        if i % 100 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] "
                f"patched={counts.get('patched', 0)} "
                f"dropped={counts.get('dropped_sol_eq_initial', 0)} "
                f"already={counts.get('already_patched', 0)} "
                f"no_dockerfile={counts.get('no_dockerfile', 0)} "
                f"no_test_sh={counts.get('no_test_sh', 0)} "
                f"no_initial={counts.get('no_initial_solution', 0)} "
                f"no_gold={counts.get('no_gold_solution', 0)}",
                flush=True,
            )

    if not args.dry_run:
        report = root / _V3_REPORT_FILENAME
        with report.open("w", encoding="utf-8") as f:
            f.write("laion exp_rpt_defects4j-v3-v2 -> v3 patch report\n")
            f.write(f"total_task_dirs_seen = {n_total}\n")
            for k in sorted(counts):
                f.write(f"{k} = {counts[k]}\n")
            f.write("\n# Dropped (solution==initial) task names:\n")
            for name in dropped_names:
                f.write(name + "\n")

    print(
        f"\nDone. total={n_total} "
        f"patched={counts.get('patched', 0)} "
        f"dropped={counts.get('dropped_sol_eq_initial', 0)} "
        f"already_patched={counts.get('already_patched', 0)} "
        f"no_dockerfile={counts.get('no_dockerfile', 0)} "
        f"no_test_sh={counts.get('no_test_sh', 0)} "
        f"no_initial={counts.get('no_initial_solution', 0)} "
        f"no_gold={counts.get('no_gold_solution', 0)} "
        f"(dry_run={args.dry_run})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
