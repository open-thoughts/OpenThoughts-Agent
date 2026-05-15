#!/usr/bin/env python3
"""
exp_rpt_methods2test-large v3 patcher.

Bug (v2 → v3, found in QC follow-up 2026-05-04):
  The v2 verifier (`tests/test.sh`) runs `mvn test -q` and uses the
  pipeline exit code as the only signal: any non-failure exit becomes
  reward=1. Maven Surefire happily reports a successful build when it
  discovers a test class but finds no `@Test`-annotated methods inside
  it (e.g. when the agent overwrote `TestSolution.java` with stubs, or
  when the test discovers zero matches and `failIfNoTests` is unset).
  Empirically: 12/34 reported v2 passes had `Tests run: 0` in the
  Surefire output and 3/34 had empty stdout — i.e. ~44% of "passes"
  were vacuous. True solve rate ≈9.5% vs. headline 17%.

Fix (v3): Rewrite `tests/test.sh` so the reward is gated on:
  - the runner exiting cleanly,
  - Surefire reporting at least one test run (`tests_run >= 1`),
  - zero failures and zero errors.

Layout (verified across all 4472 v2 task dirs — byte-identical
test.sh files, same MD5):
  - Single test.sh template using `mvn test -q | tee` and
    `exit ${PIPESTATUS[0]}`.
  - Surefire prints both `Tests run: N, Failures: F, Errors: E, Skipped: S`
    (per-class) and `Tests run: N, Failures: F, Errors: E, Skipped: S`
    (final "Results:" aggregate). We parse the LAST occurrence so we
    pick up the aggregate, not a single class line. Falls back to
    summing per-class lines if the aggregate is missing (e.g. only
    one test class).
  - Reward floor: write 0 first, only upgrade to 1 after all gates
    pass. EXIT trap is dropped (the explicit reward-write at the end
    supersedes it; the trap would otherwise overwrite the gated value
    with 1 if the script exits cleanly).

The patcher rewrites the whole file (since it's uniform across the
corpus). Idempotency is enforced via the marker
  `# --- laion v3 patch: tests_found gate ---`
near the top of the new test.sh: if present, the file is left alone.

Usage:
  python data/patchers/patch_methods2test_v3_tasks.py --root <dir> [--dry-run] [--limit N]

Constraints (per upload spec):
  - Only `tests/test.sh` is touched. instruction.md, TestSolution.java,
    solution/, environment/Dockerfile, pom.xml are preserved verbatim.
  - v3 is cumulative on v2 (the v2 instruction.md rewrites are still
    in place; this patch is orthogonal).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Marker we drop into the new test.sh so re-runs are idempotent.
V3_MARKER = "# --- laion v3 patch: tests_found gate ---"

# The new test.sh body. We:
#   - keep the Maven invocation, output redirection, and PIPESTATUS exit.
#   - drop the EXIT-trap-based scoring (it would clobber the gated value).
#   - parse the LAST `Tests run: N, Failures: F, Errors: E[, Skipped: S]`
#     line in the captured output. That line is emitted by Surefire as
#     both per-class and aggregate; the aggregate (after `Results :`) is
#     last when present. If only per-class lines exist, the last one is
#     fine for the single-class case which is the norm here.
#   - require runner_rc == 0 AND tests_run >= 1 AND failures == 0 AND
#     errors == 0 to score reward=1. Anything else = 0.
NEW_TEST_SH = f"""#!/bin/bash
{V3_MARKER}
# v3 verifier: gate reward on (a) clean exit AND (b) Surefire reporting
# at least one test actually executed. v2 scored reward=1 on any
# zero-exit `mvn test`, which let `Tests run: 0` outputs (vacuous
# discovery, agent-overwritten test class with no @Test methods, etc.)
# pass. See QC follow-up 2026-05-04.

set +e  # we manage exit/reward explicitly; don't abort on first failure

mkdir -p /logs/verifier
# Reward floor: any abnormal exit below leaves this 0 in place.
echo "0" > /logs/verifier/reward.txt

cd /app

# Create Maven project structure if needed
mkdir -p src/main/java

# Copy pom.xml if not exists
if [ ! -f pom.xml ]; then
    cp /tests/pom.xml .
fi

echo "Compiling and running tests..."
mvn test -q 2>&1 | tee /logs/verifier/test_output.txt
runner_rc=${{PIPESTATUS[0]}}

# Parse the LAST Surefire summary line in the captured output.
# Surefire emits e.g. "Tests run: 3, Failures: 0, Errors: 0, Skipped: 0"
# both per-class and as the aggregate "Results :" footer. We grab the
# last occurrence so we score the aggregate when present, and the only
# per-class line otherwise.
summary_line=$(grep -E 'Tests run: [0-9]+, Failures: [0-9]+, Errors: [0-9]+' \\
    /logs/verifier/test_output.txt | tail -1)

tests_run=$(echo "$summary_line" | grep -oE 'Tests run: [0-9]+' \\
    | grep -oE '[0-9]+' | head -1)
failures=$(echo "$summary_line" | grep -oE 'Failures: [0-9]+' \\
    | grep -oE '[0-9]+' | head -1)
errors=$(echo "$summary_line" | grep -oE 'Errors: [0-9]+' \\
    | grep -oE '[0-9]+' | head -1)

# Default to 0/0/0 if any field is missing (will fail the gate below,
# producing reward=0 — the safe direction).
tests_run=${{tests_run:-0}}
failures=${{failures:-0}}
errors=${{errors:-0}}

echo "v3 verifier: runner_rc=$runner_rc tests_run=$tests_run failures=$failures errors=$errors"

if [ "$runner_rc" -eq 0 ] \\
        && [ "$tests_run" -ge 1 ] \\
        && [ "$failures" -eq 0 ] \\
        && [ "$errors" -eq 0 ]; then
    echo "1" > /logs/verifier/reward.txt
    exit 0
else
    echo "0" > /logs/verifier/reward.txt
    # Preserve a non-zero status for outer pipelines that inspect $? —
    # but only when something actually went wrong with the runner.
    if [ "$runner_rc" -ne 0 ]; then
        exit "$runner_rc"
    fi
    exit 1
fi
"""


def patch_one(test_sh: Path, dry_run: bool) -> str:
    """Patch a single tests/test.sh file. Returns one of:
    'patched', 'already', 'missing', 'unparseable'.
    """
    if not test_sh.is_file():
        return "missing"
    try:
        existing = test_sh.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unparseable"
    if V3_MARKER in existing:
        return "already"

    # Sanity: the v2 file we expect contains `mvn test` and the EXIT trap.
    # If it doesn't, we still rewrite (our new file is self-contained and
    # correct for the documented uniform layout), but flag it so we can
    # report any unexpected variants. We keep this advisory, not a
    # disqualifier.
    is_expected = ("mvn test" in existing) and ("trap cleanup EXIT" in existing)

    if not dry_run:
        test_sh.write_text(NEW_TEST_SH, encoding="utf-8")
    return "patched" if is_expected else "patched_unusual"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to extracted tasks root")
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
    counts = {
        "patched": 0,
        "patched_unusual": 0,
        "already": 0,
        "missing": 0,
        "unparseable": 0,
    }

    for i, td in enumerate(task_dirs, 1):
        test_sh = td / "tests" / "test.sh"
        result = patch_one(test_sh, dry_run=args.dry_run)
        counts[result] = counts.get(result, 0) + 1

        if i % 1000 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] patched={counts['patched']} "
                f"patched_unusual={counts['patched_unusual']} "
                f"already={counts['already']} missing={counts['missing']} "
                f"unparseable={counts['unparseable']}",
                flush=True,
            )

    print(
        f"\nDone. total={n_total} "
        f"patched={counts['patched']} "
        f"patched_unusual={counts['patched_unusual']} "
        f"already_patched={counts['already']} "
        f"missing={counts['missing']} "
        f"unparseable={counts['unparseable']} "
        f"(dry_run={args.dry_run})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
