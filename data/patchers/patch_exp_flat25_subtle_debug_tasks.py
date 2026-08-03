#!/usr/bin/env python3
"""
exp_flat25_subtle_debug v2 patcher.

Background
----------
DCAgent/exp_flat25_subtle_debug is a prompt-engineering treatment of the
same upstream 5,000-task pool used by other ``exp_flat25_*`` variants
(``patch_rle_flat25_tasks.py`` already filters the same pool for the
``heavy_padding`` / ``speed_bonus`` / ``pseudocode`` / ``stackoverflow``
treatments). The ``subtle_debug`` cut was published WITHOUT applying that
filter, so its v1 (n=200 QC) reports 200/200 infra-ok but only 2/200
solved (1.0%).

QC findings (2026-05-14, 10/10 hand-inspected traces):
  - 8/10: verifier exits with ``ModuleNotFoundError`` for packages the
    container image does not preinstall and the task description does
    not request, e.g.

        from openeo.rest.connection import Connection
            -> ModuleNotFoundError: No module named 'openeo'
        from traveling_salesperson import __main__ as main
            -> ModuleNotFoundError: No module named 'traveling_salesperson'
        import cmd2 / from celery.result import EagerResult /
        from aiocron import asyncio / from illud.output import Output /
        from get_pr_info import ... / from soocii_pubsub_lib import ...
            -> ModuleNotFoundError

    Same disease as ``rle_flat25`` / ``exp_rle_detailed`` — synthesized
    ``tests/test_solution.py`` imports random third-party packages that
    aren't in the container image and aren't mentioned in
    ``instruction.md``. Unfixable without restructuring the test → drop.

  - 2/2 *passing* trials (reward=1) have pytest output:

        collected 3 items
        ../tests/test_solution.py::... SKIPPED [ 33%]
        ../tests/test_solution.py::... SKIPPED [ 66%]
        ../tests/test_solution.py::... SKIPPED [100%]
        ==== 3 skipped, 1 warning in 0.44s ====
        All tests passed!

    The v1 ``tests/test.sh`` invokes ``pytest`` and trusts the pytest
    exit code as the *only* signal. pytest exits 0 when 0 tests are
    collected OR when every collected test is skipped — so every
    "all-skipped" suite rubber-stamps reward=1 with NOTHING actually
    asserted. Cross-checked against ``exp_flat25_speed_bonus-v2``
    (already filtered, headline 8.5% solve): task ``5k-speed-0792`` is
    a "3 skipped → All tests passed!" run there too, i.e. this is a
    *systemic* gap in the WHITELIST-based ``test.sh`` template shared
    across the whole exp_flat25 family.

Fix (v2)
--------
Two complementary changes applied in one pass:

  1. **Drop pass** — Reuse the ``rle_flat25`` evaluator unchanged:
     remove any task whose ``tests/test_solution.py`` doesn't compile,
     or imports a top-level module that's neither stdlib, in the
     container pip whitelist, nor mentioned literally in
     ``instruction.md``. This kills the 8/10 ``ModuleNotFoundError``
     mode.

  2. **Mutate pass** — Rewrite ``tests/test.sh`` so the reward is
     gated on:
       - pytest exiting cleanly,
       - pytest reporting ``>= 1`` test actually PASSED (not skipped,
         deselected, or zero-collected),
       - zero failures and zero errors.

     The summary line that pytest emits has a uniform regex-able shape
     (``N passed[, M failed][, K skipped]... in X.YYs``), and pytest 7+
     supports ``--strict-markers`` + ``-ra`` but those don't change the
     exit code on all-skipped — we parse the summary line directly.

Layout (verified across `5k-baseline-*` traces, byte-identical MD5
for ``tests/test.sh`` across the corpus):
  - ``tests/test.sh`` creates a venv at ``/app/.venv``, installs the
    fixed WHITELIST pip packages, runs ``pytest /tests/test_solution.py
    -v --tb=short``, and uses an EXIT trap on ``$?`` for scoring (the
    rubber-stamp bug).

The new ``test.sh`` keeps the venv + WHITELIST install verbatim,
captures pytest output to ``/logs/verifier/pytest_output.txt``, drops
the EXIT trap, parses the summary line, and writes
``/logs/verifier/reward.txt`` based on the parsed counts.

Idempotency: the new file carries the marker
``# --- laion v2 patch: passed_count gate ---`` near the top; if
present, the patcher leaves the file alone.

If a task has no ``tests/test.sh`` or its existing ``test.sh`` does
not invoke ``pytest``, the file is left untouched and counted as
``skipped_no_pytest``.

Usage
-----
    python data/patchers/patch_exp_flat25_subtle_debug_tasks.py --root <dir>
    python data/patchers/patch_exp_flat25_subtle_debug_tasks.py --root <dir> --dry-run
    python data/patchers/patch_exp_flat25_subtle_debug_tasks.py --root <dir> --limit 200

Target HF repo: ``laion/exp_flat25_subtle_debug-v2``.

Constraints (per upload spec):
  - Only ``tests/test.sh`` is mutated. ``instruction.md`` /
    ``tests/test_solution.py`` / ``environment/Dockerfile`` are
    preserved verbatim on KEPT tasks.
  - Tasks that fail the import-filter are DROPPED (directory removed),
    matching the existing ``patch_rle_flat25_tasks.py`` behavior.
  - Daytona snapshot caps are HARD: this patcher cannot grow the task
    set; it can only drop. Mutation of ``test.sh`` doesn't change the
    Dockerfile so it doesn't change the snapshot identity.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Reuse the shared evaluator from the rle_flat25 patcher.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from patch_rle_flat25_tasks import evaluate_task  # noqa: E402

# ---------------------------------------------------------------------------
# Mutation: new tests/test.sh body
# ---------------------------------------------------------------------------

V2_MARKER = "# --- laion v2 patch: passed_count gate ---"

# New test.sh body. We:
#   - keep the venv creation + pip-install pytest + WHITELIST install loop
#     verbatim (these are the runtime behavior the corpus expects).
#   - drop the EXIT trap on $? (the rubber-stamp scoring; would clobber
#     our gated reward).
#   - capture pytest output and parse the summary line for the count of
#     PASSED tests. We score reward=1 iff:
#         * pytest exit code == 0
#         * the summary line is present
#         * passed_count >= 1
#         * failed_count == 0 AND error_count == 0
#   - "all-skipped" or "zero-collected" runs leave passed_count = 0 -> 0.
NEW_TEST_SH = r"""#!/bin/bash
""" + V2_MARKER + r"""
# v2 verifier: gate reward on (a) clean exit AND (b) pytest reporting
# at least one test actually PASSED. v1 trusted the pytest exit code,
# which is 0 for "0 collected" and "all skipped" runs - both let
# vacuous trials rubber-stamp reward=1.

set +e  # we manage reward/exit explicitly
mkdir -p /logs/verifier
# Reward floor: any abnormal exit below leaves this 0 in place.
echo "0" > /logs/verifier/reward.txt

# Create virtual environment if needed (PEP 668 on Python 3.12+).
if [ ! -d /app/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv /app/.venv
fi
source /app/.venv/bin/activate

# Install pytest if not available
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install --quiet pytest
fi

# Install whitelisted test dependencies (not agent-created packages)
WHITELIST="requests numpy pandas scipy scikit-learn sklearn torch tensorflow keras httpx aiohttp ddtrace django flask fastapi matplotlib seaborn pillow pydantic pytest-mock requests-mock faker pyyaml pytz cryptography bcrypt hypothesis"
for pkg in $WHITELIST; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || pip install --quiet "$pkg" 2>/dev/null || true
done

cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

echo "Running pytest..."
pytest /tests/test_solution.py -v --tb=short 2>&1 \
    | tee /logs/verifier/pytest_output.txt
PYTEST_EXIT=${PIPESTATUS[0]}

# Pytest summary line shapes (last "==... in X.YYs ==..." line):
#   "==== 7 passed in 0.44s ===="
#   "==== 2 failed, 5 passed in 0.51s ===="
#   "==== 3 skipped, 1 warning in 0.40s ====" (vacuous - the v1 bug)
#   "==== 1 error in 0.49s ===="              (collection error)
#   "==== 6 failed, 46 passed, 1 error in 0.92s ===="
#
# Strategy: grep the LAST line matching " in <number>s" (handles both
# "in 0.44s" and "in 0.44 s"), then count occurrences of " N passed",
# " N failed", " N error", " N errors" in it.
summary_line=$(grep -E '={3,}[^=]*\b[0-9]+(\.[0-9]+)?s\b[^=]*={3,}' \
    /logs/verifier/pytest_output.txt | tail -1)

extract() {
    # $1 = keyword (passed|failed|error|errors); echo the count or 0.
    local kw="$1"
    local n
    n=$(echo "$summary_line" | grep -oE "[0-9]+ ${kw}\b" \
        | grep -oE '[0-9]+' | head -1)
    echo "${n:-0}"
}

passed=$(extract passed)
failed=$(extract failed)
error_s=$(extract error)
errors_p=$(extract errors)
# pytest uses singular "error" for 1 and "errors" for >= 2; take max.
if [ "$errors_p" -gt "$error_s" ]; then errors=$errors_p; else errors=$error_s; fi

echo "v2 verifier: pytest_exit=$PYTEST_EXIT passed=$passed failed=$failed errors=$errors summary='${summary_line}'"

if [ "$PYTEST_EXIT" -eq 0 ] \
        && [ "$passed" -ge 1 ] \
        && [ "$failed" -eq 0 ] \
        && [ "$errors" -eq 0 ]; then
    echo "1" > /logs/verifier/reward.txt
    echo "All tests passed."
    exit 0
else
    echo "0" > /logs/verifier/reward.txt
    echo "Some tests failed or no tests passed."
    if [ "$PYTEST_EXIT" -ne 0 ]; then exit "$PYTEST_EXIT"; fi
    exit 1
fi
"""


def patch_test_sh(test_sh: Path, dry_run: bool) -> str:
    """Patch a single tests/test.sh. Returns one of:
    'patched', 'patched_unusual', 'already', 'missing',
    'skipped_no_pytest', 'unparseable'.
    """
    if not test_sh.is_file():
        return "missing"
    try:
        existing = test_sh.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unparseable"
    if V2_MARKER in existing:
        return "already"

    # Skip tasks whose test.sh doesn't actually invoke pytest. The
    # exp_flat25 corpus is uniform (pytest-based) but we guard anyway
    # so a stray non-pytest task doesn't get corrupted.
    if "pytest" not in existing.lower():
        return "skipped_no_pytest"

    # Sanity: the v1 file we expect uses `trap cleanup EXIT` for scoring.
    # If absent, we still rewrite (the new file is self-contained), but
    # flag for reporting.
    is_expected = "trap cleanup EXIT" in existing

    if not dry_run:
        test_sh.write_text(NEW_TEST_SH, encoding="utf-8")
    return "patched" if is_expected else "patched_unusual"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True, type=Path,
                   help="Directory containing extracted task folders.")
    p.add_argument("--dry-run", action="store_true",
                   help="Only report counts; do not delete dropped task folders.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only inspect the first N tasks (debug).")
    p.add_argument("--report-json", type=Path, default=None,
                   help="Write per-task verdict JSON to this file.")
    p.add_argument("--show-bad", type=int, default=20,
                   help="Print the first N bad-import samples.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"[patch_exp_flat25_subtle_debug] inspecting {len(task_dirs)} tasks under {root}")

    syntax_dropped = 0
    no_test_dropped = 0
    import_dropped = 0
    kept = 0

    patch_counts = {
        "patched": 0,
        "patched_unusual": 0,
        "already": 0,
        "missing": 0,
        "skipped_no_pytest": 0,
        "unparseable": 0,
    }

    bad_import_samples: list[tuple[str, list[str]]] = []
    verdicts: dict[str, dict] = {}

    for td in task_dirs:
        v = evaluate_task(td)
        verdicts[td.name] = v
        if v["kept"]:
            kept += 1
            # Mutation: patch tests/test.sh in-place.
            test_sh = td / "tests" / "test.sh"
            result = patch_test_sh(test_sh, dry_run=args.dry_run)
            patch_counts[result] = patch_counts.get(result, 0) + 1
            continue
        reason = v["reason"]
        if reason == "no_test_file":
            no_test_dropped += 1
        elif (reason.startswith("syntax_error")
              or reason.startswith("ast_syntax")
              or reason.startswith("compile_error")):
            syntax_dropped += 1
        elif reason == "missing_import":
            import_dropped += 1
            if len(bad_import_samples) < args.show_bad:
                bad_import_samples.append((td.name, v["bad_imports"]))

    total = len(task_dirs)

    print()
    print(f"[patch_exp_flat25_subtle_debug] total:          {total}")
    print(f"[patch_exp_flat25_subtle_debug] no test file:   {no_test_dropped}")
    print(f"[patch_exp_flat25_subtle_debug] syntax errors:  {syntax_dropped}")
    print(f"[patch_exp_flat25_subtle_debug] import dropped: {import_dropped}")
    print(f"[patch_exp_flat25_subtle_debug] kept:           {kept}")
    print()
    print("[patch_exp_flat25_subtle_debug] test.sh mutation (on KEPT tasks):")
    for k in ("patched", "patched_unusual", "already",
              "missing", "skipped_no_pytest", "unparseable"):
        print(f"  {k:<20}: {patch_counts[k]:>5}")

    if bad_import_samples:
        print()
        print(f"[patch_exp_flat25_subtle_debug] sample bad-import drops "
              f"(first {len(bad_import_samples)}):")
        for tid, bad in bad_import_samples:
            print(f"  {tid}: {bad}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_exp_flat25_subtle_debug] wrote per-task verdicts: {args.report_json}")

    if args.dry_run:
        print("[patch_exp_flat25_subtle_debug] dry-run: not deleting dropped tasks.")
        return

    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_exp_flat25_subtle_debug] removed {removed} dropped task directories.")


if __name__ == "__main__":
    main()
