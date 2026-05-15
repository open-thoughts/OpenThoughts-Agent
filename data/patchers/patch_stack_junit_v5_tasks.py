#!/usr/bin/env python3
"""
exp_rpt_stack-junit patcher (v4 -> v5).

QC of the v4 dataset (`laion/exp_rpt_stack-junit-v4`) found 0% solve rate
across 200 trials despite the v4 testfile-rename patch landing correctly.
Whole-dataset failure mode breakdown:

  - 160/200  javac compile failure (real model-capability ceiling: tests
             reference 3rd-party libs the agent didn't write a stub for)
  - 11/200   tests COMPILED + RAN + ALL PASSED — but `awk: line 1: syntax
             error at or near ,`. Reward marked 0 anyway.
  - 28/200   AgentSetupTimeoutError / RuntimeError at Daytona setup
             (image-side tmux install hang, unrelated to verifier)
  - 1/200    other

The 11 silently-discarded passes are the bug this patcher fixes. The v4
test.sh uses gawk-only 3-arg `match($0, /pattern/, a)` to extract test
counts from the JUnit Console summary. Several Daytona base images use
mawk / busybox-awk / nawk which crash on this syntax with "syntax error
at or near ,". When awk crashes, all four `found/failed/aborted/cfailed`
variables stay empty, the `: "${VAR:=0}"` defaults kick in, the gate
`found > 0` fails (because found=0), and the verdict is FAIL.

This is the same fingerprint as v3->v4: the model is doing real work,
the test corpus is fine, but the verifier shell script is broken in a
non-portable way.

Fix
---
Replace the four gawk-3-arg `match()` calls with portable POSIX awk
that uses `gsub()` to strip non-digits and works on mawk/nawk/gawk
alike. We also keep the entire v4 verifier flow (testfile rename,
reward floor, idempotency marker, JUnit gating) verbatim — only the
parsing block changes.

The new awk replacements look like::

    found=$(awk '/[0-9]+[[:space:]]+tests found/{
                    line=$0
                    gsub(/[^0-9]/," ",line)
                    n=split(line,a," ")
                    print a[1]; exit
                 }' "$out")

This reads the first line that contains "tests found" (the JUnit
summary uses fixed labels), strips everything but digits + spaces,
splits on whitespace, and prints the first numeric token. Works on
all three awk dialects we've seen on Daytona images (mawk 1.3.4,
gawk 5.x, nawk).

We also write the parsed counts to a debug log line (`laion v5 gate:
...`) so the next round of QC can verify the parser fired.

Idempotency
-----------
Idempotent via marker `# --- laion v5 patch: portable awk parser ---`.
Re-running on a v5-patched task is a no-op. v4 markers are preserved
since v5's patch SUPERSEDES the v4 verifier block; we replace the
entire v4 test.sh body with the v5 version (preserving the v4
testfile rename — the on-disk Java file rename from v4 is not undone).

This patcher does NOT re-rename test files (v4 already did that).
The test_src path is read out of the existing v4 test.sh's
`local test_src=...` line.

CLI
---
::

    python data/patchers/patch_stack_junit_v5_tasks.py --root <dir> [--dry-run]

Apply ON TOP of an already-v4-patched extraction. After patching,
upload to ``laion/exp_rpt_stack-junit-v5``.

Expected outcome (whole-dataset, post-v5):
  - The 11 false-zero trials become reward=1 → solve rate ≥ 5.5%.
  - The 160 javac fails stay javac fails (separate v6 patch needed
    for those — would have to bundle mockito/assertj/etc. into the
    classpath).
  - The 28 AgentSetupTimeoutError trials stay infra failures
    (Daytona-side tmux install issue; not patcher-fixable).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

V5_MARKER = "# --- laion v5 patch: portable awk parser ---"
V4_MARKER = "# --- laion v4 patch: testfile renamed ---"

# Match the v4 test.sh's `local test_src=<path>` line so we can preserve
# whatever path v4 chose (post-rename). Tolerate quotes and whitespace.
_TEST_SRC_RE = re.compile(
    r"^\s*local\s+test_src\s*=\s*([^\s#]+)\s*$",
    flags=re.MULTILINE,
)

# Full v5 test.sh body. Renders the same logic as v4 but with portable
# awk for the post-test gate.
_V5_TEMPLATE = r"""#!/bin/bash
# --- laion v2 verifier patch: env shims + reward floor + strict JUnit gating ---
mkdir -p /logs/verifier 2>/dev/null || true
echo 0 > /logs/verifier/reward.txt 2>/dev/null || true
trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT
# --- end laion v2 patch ---
{V4_MARKER}
{V5_MARKER}

set -eo pipefail

cd /app

verdict=0  # 0 = fail, 1 = pass

# Portable awk parser: works on gawk, mawk, nawk, busybox-awk.
# Reads $1 (the path to test_output.txt) and prints, on stdout, four
# space-separated integers: tests_found tests_failed tests_aborted containers_failed.
parse_junit_counts() {{
    awk '
        /tests found/ && !found {{
            line=$0; gsub(/[^0-9]/," ",line); split(line, a, " "); found=a[1]+0
        }}
        /tests failed/ && !fdone {{
            line=$0; gsub(/[^0-9]/," ",line); split(line, a, " "); failed=a[1]+0; fdone=1
        }}
        /tests aborted/ && !adone {{
            line=$0; gsub(/[^0-9]/," ",line); split(line, a, " "); aborted=a[1]+0; adone=1
        }}
        /containers failed/ && !cdone {{
            line=$0; gsub(/[^0-9]/," ",line); split(line, a, " "); cfailed=a[1]+0; cdone=1
        }}
        END {{ printf "%d %d %d %d\n", found+0, failed+0, aborted+0, cfailed+0 }}
    ' "$1"
}}

run_pipeline() {{
    local jar=/junit/junit-platform-console-standalone.jar
    local out=/logs/verifier/test_output.txt
    local cmp=/logs/verifier/compile_output.txt
    local test_src={TEST_SRC_PATH}

    if [ -f pom.xml ]; then
        timeout 300 mvn test 2>&1 | tee "$out"
        return ${{PIPESTATUS[0]}}
    fi

    mkdir -p /app/classes
    local java_files
    java_files=$(find /app -path /app/classes -prune -o -name '*.java' -print 2>/dev/null | tr '\n' ' ')

    echo "Compiling..."
    # shellcheck disable=SC2086
    javac -cp "$jar" -d /app/classes "$test_src" $java_files 2>&1 | tee "$cmp"
    local jc=${{PIPESTATUS[0]}}
    if [ "$jc" -ne 0 ]; then
        echo "laion v5: javac failed (exit $jc), refusing to score as PASS"
        return "$jc"
    fi

    echo "Running tests..."
    timeout 300 java -jar "$jar" --class-path /app/classes --scan-class-path 2>&1 | tee "$out"
    return ${{PIPESTATUS[0]}}
}}

if run_pipeline; then
    out=/logs/verifier/test_output.txt
    if [ -s "$out" ]; then
        # Portable parse: avoids gawk-only `match($0, /re/, a)` 3-arg form.
        read -r found failed aborted cfailed <<< "$(parse_junit_counts "$out")"
        : "${{found:=0}}"; : "${{failed:=0}}"; : "${{aborted:=0}}"; : "${{cfailed:=0}}"
        echo "laion v5 gate: tests_found=$found failed=$failed aborted=$aborted containers_failed=$cfailed"
        if [ "$found" -gt 0 ] && [ "$failed" -eq 0 ] && [ "$aborted" -eq 0 ] && [ "$cfailed" -eq 0 ]; then
            verdict=1
        fi
    else
        echo "laion v5 gate: no test output captured"
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


def extract_test_src(test_sh_text: str) -> str | None:
    """Pull the v4 ``local test_src=<path>`` value, if any."""
    m = _TEST_SRC_RE.search(test_sh_text)
    if not m:
        return None
    return m.group(1)


def patch_one_task(task_dir: Path, dry_run: bool) -> str:
    """Patch a single task. Returns one of:

    - ``"already-v5"``  — already patched
    - ``"not-v4"``      — task hasn't been v4-patched (skip; run v4 first)
    - ``"patched"``     — patch applied
    - ``"no-test-sh"``  — no tests/test.sh
    - ``"err:<msg>"``   — anything else
    """
    test_sh = task_dir / "tests" / "test.sh"
    if not test_sh.is_file():
        return "no-test-sh"
    try:
        text = test_sh.read_text()
    except Exception as e:
        return f"err:read:{e}"

    if V5_MARKER in text:
        return "already-v5"

    if V4_MARKER not in text:
        # We require v4 to have run first because we copy v4's renamed
        # test_src path. Running v5 on a vanilla task would lose that.
        return "not-v4"

    test_src = extract_test_src(text)
    if not test_src:
        # Defensive: v4 always emits a test_src line.
        return "err:no-test-src-in-v4-test.sh"

    new_text = _V5_TEMPLATE.format(
        V4_MARKER=V4_MARKER,
        V5_MARKER=V5_MARKER,
        TEST_SRC_PATH=test_src,
    )

    if dry_run:
        return "patched"
    try:
        test_sh.write_text(new_text)
    except Exception as e:
        return f"err:write:{e}"
    return "patched"


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

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    counts: dict[str, int] = {}
    for i, td in enumerate(task_dirs, 1):
        verdict = patch_one_task(td, args.dry_run)
        counts[verdict] = counts.get(verdict, 0) + 1
        if i % 1000 == 0:
            print(f"[{i}/{len(task_dirs)}] {counts}", flush=True)

    print()
    print("=" * 60)
    print(f"Total tasks scanned: {len(task_dirs)}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<20}: {v}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
