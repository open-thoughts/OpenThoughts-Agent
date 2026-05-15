#!/usr/bin/env python3
"""
exp_rpt_stack-junit patcher.

Bug: the shared test.sh runs javac + standalone JUnit. Without pipefail,
`javac ... 2>&1 | tee` hides javac errors, then JUnit runs against an empty
classpath, prints `[0 tests found]`, and exits 0 -> reward=1.

Fix: replace the entire test.sh with a stricter version that
  1. floors /logs/verifier/reward.txt = 0 immediately
  2. enables `set -eo pipefail` so javac/tee failures propagate
  3. requires the JUnit output to contain `[ N tests found ]` with N >= 1
     AND `[ 0 tests failed ]` AND `[ 0 tests aborted ]` AND no
     `[ N containers failed ]` with N >= 1
  4. only writes reward=1 when those conditions hold

All 10000 tasks share the same test.sh (verified by md5sum), so the patch is
a single-file replacement applied uniformly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCHED_TEST_SH = r"""#!/bin/bash
# --- laion v2 verifier patch: env shims + reward floor + strict JUnit gating ---
mkdir -p /logs/verifier 2>/dev/null || true
echo 0 > /logs/verifier/reward.txt 2>/dev/null || true
# Always leave reward.txt populated even if the script aborts mid-way
trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT
# --- end laion v2 patch ---

set -eo pipefail

cd /app

verdict=0  # 0 = fail, 1 = pass

run_pipeline() {
    local jar=/junit/junit-platform-console-standalone.jar
    local out=/logs/verifier/test_output.txt
    local cmp=/logs/verifier/compile_output.txt

    if [ -f pom.xml ]; then
        timeout 300 mvn test 2>&1 | tee "$out"
        return ${PIPESTATUS[0]}
    fi

    mkdir -p /app/classes
    # Collect agent-supplied .java sources under /app (recursive so package
    # subdirectories like main/java/... are found).
    local java_files
    java_files=$(find /app -path /app/classes -prune -o -name '*.java' -print 2>/dev/null | tr '\n' ' ')

    echo "Compiling..."
    # shellcheck disable=SC2086
    javac -cp "$jar" -d /app/classes /tests/TestSolution.java $java_files 2>&1 | tee "$cmp"
    local jc=${PIPESTATUS[0]}
    if [ "$jc" -ne 0 ]; then
        echo "laion v2: javac failed (exit $jc), refusing to score as PASS"
        return "$jc"
    fi

    echo "Running tests..."
    timeout 300 java -jar "$jar" --class-path /app/classes --scan-class-path 2>&1 | tee "$out"
    return ${PIPESTATUS[0]}
}

if run_pipeline; then
    out=/logs/verifier/test_output.txt
    if [ -s "$out" ]; then
        # Use awk to extract counts from the JUnit summary table:
        #   [         N tests found      ]
        #   [         N tests failed     ]
        #   [         N containers failed]
        #   [         N tests aborted    ]
        found=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+tests found[[:space:]]*\]/, a){print a[1]; exit}' "$out")
        failed=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+tests failed[[:space:]]*\]/, a){print a[1]; exit}' "$out")
        aborted=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+tests aborted[[:space:]]*\]/, a){print a[1]; exit}' "$out")
        cfailed=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+containers failed[[:space:]]*\]/, a){print a[1]; exit}' "$out")
        : "${found:=0}"; : "${failed:=0}"; : "${aborted:=0}"; : "${cfailed:=0}"
        echo "laion v2 gate: tests_found=$found failed=$failed aborted=$aborted containers_failed=$cfailed"
        if [ "$found" -gt 0 ] && [ "$failed" -eq 0 ] && [ "$aborted" -eq 0 ] && [ "$cfailed" -eq 0 ]; then
            verdict=1
        fi
    else
        echo "laion v2 gate: no test output captured"
    fi
fi

if [ "$verdict" -eq 1 ]; then
    echo 1 > /logs/verifier/reward.txt
    exit 0
else
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    test_paths = sorted(root.glob("*/tests/test.sh"))
    if not test_paths:
        print(f"No tests/test.sh files under {root}", file=sys.stderr)
        return 2

    if args.limit:
        test_paths = test_paths[: args.limit]

    n_changed = 0
    n_total = len(test_paths)
    marker = "# --- laion v2 verifier patch:"
    for i, p in enumerate(test_paths, 1):
        current = p.read_text()
        if current == PATCHED_TEST_SH:
            continue  # already patched
        n_changed += 1
        if not args.dry_run:
            p.write_text(PATCHED_TEST_SH)
        if i % 1000 == 0 or i == n_total:
            print(f"[{i}/{n_total}] patched={n_changed}", flush=True)

    # Sanity: marker present in patched file
    if PATCHED_TEST_SH.find(marker) < 0:
        print("WARNING: patch marker not found in PATCHED_TEST_SH constant", file=sys.stderr)

    print(f"Done. {n_changed}/{n_total} files modified (dry_run={args.dry_run}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
