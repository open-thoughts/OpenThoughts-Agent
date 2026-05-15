#!/usr/bin/env python3
"""
exp_rpt_stack-junit v4 patcher.

Bug (v3 → v4): the v3 patcher enriched ``instruction.md`` with a Test
contract (package, imports, public API), so the agent now knows what to
build — but solve rate is still **0/200**. Root cause is on the verifier
side, not the prompt side.

84.5% (169/200) of v3 trials fail with::

    /tests/TestSolution.java:N: error: class <ClassName> is public, should
        be declared in a file named <ClassName>.java

The v2 verifier's ``test.sh`` runs::

    javac -cp $jar -d /app/classes /tests/TestSolution.java $java_files

But the *contents* of ``/tests/TestSolution.java`` for ~85% of tasks
declare a public class with a different name (e.g. ``DelimiterTest``,
``Node_ESTest``, ``CountingMemoryCacheTest``). Per JLS §7.6, ``javac``
**requires** the file to be named ``<PublicClassName>.java``. So even when
the agent writes a perfect ``Delimiter.java`` matching the contract, the
test fixture itself can't compile and reward stays 0.

Fix (v4): for each task, parse the public class name from
``tests/TestSolution.java`` and:

  1. Rename ``tests/TestSolution.java`` → ``tests/<PublicClassName>.java``.
  2. Rewrite ``tests/test.sh`` so the ``javac`` line references the new
     filename. We keep the rest of the v2 verifier's gating intact (reward
     floor, ``set -eo pipefail``, JUnit summary parsing).
  3. If the public class is already named ``TestSolution``, skip
     (no-op — file is already correct).
  4. If we cannot find a public class declaration, leave the task
     untouched and report it.

Idempotent: looks for ``# --- laion v4 patch: testfile renamed ---`` in
``test.sh``. If present, skips the rewrite.

Note: this patcher does NOT alter ``instruction.md`` — the v3 enrichment
is preserved. Run v3 first if you have a fresh extraction; this v4
patcher assumes v3 has already done its job.

Usage::

    python data/patchers/patch_stack_junit_v4_tasks.py \
        --root /tmp/tasks_extracted/<repo> \
        [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Regex for finding the FIRST public class/interface/enum declaration. We
# only care about the public top-level type, since javac's filename rule
# only applies to public types.
# --------------------------------------------------------------------------- #
_PUBLIC_TYPE_RE = re.compile(
    r"\bpublic\s+(?:final\s+|abstract\s+|strictfp\s+)?"
    r"(?:class|interface|enum|record)\s+(\w+)"
)

# Marker that identifies a v4-patched test.sh.
_V4_MARKER = "# --- laion v4 patch: testfile renamed ---"

# The patched test.sh template — same logic as v2 but parameterised by
# the renamed test file path.
_PATCHED_TEST_SH_TEMPLATE = r"""#!/bin/bash
# --- laion v2 verifier patch: env shims + reward floor + strict JUnit gating ---
mkdir -p /logs/verifier 2>/dev/null || true
echo 0 > /logs/verifier/reward.txt 2>/dev/null || true
trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT
# --- end laion v2 patch ---
{V4_MARKER}

set -eo pipefail

cd /app

verdict=0  # 0 = fail, 1 = pass

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
        echo "laion v4: javac failed (exit $jc), refusing to score as PASS"
        return "$jc"
    fi

    echo "Running tests..."
    timeout 300 java -jar "$jar" --class-path /app/classes --scan-class-path 2>&1 | tee "$out"
    return ${{PIPESTATUS[0]}}
}}

if run_pipeline; then
    out=/logs/verifier/test_output.txt
    if [ -s "$out" ]; then
        found=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+tests found[[:space:]]*\]/, a){{print a[1]; exit}}' "$out")
        failed=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+tests failed[[:space:]]*\]/, a){{print a[1]; exit}}' "$out")
        aborted=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+tests aborted[[:space:]]*\]/, a){{print a[1]; exit}}' "$out")
        cfailed=$(awk 'match($0, /\[[[:space:]]*([0-9]+)[[:space:]]+containers failed[[:space:]]*\]/, a){{print a[1]; exit}}' "$out")
        : "${{found:=0}}"; : "${{failed:=0}}"; : "${{aborted:=0}}"; : "${{cfailed:=0}}"
        echo "laion v4 gate: tests_found=$found failed=$failed aborted=$aborted containers_failed=$cfailed"
        if [ "$found" -gt 0 ] && [ "$failed" -eq 0 ] && [ "$aborted" -eq 0 ] && [ "$cfailed" -eq 0 ]; then
            verdict=1
        fi
    else
        echo "laion v4 gate: no test output captured"
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


def find_public_class_name(java_src: str) -> str | None:
    """Return the simple name of the first ``public`` top-level type, or None."""
    # Strip block + line comments + strings to avoid false hits inside docs.
    src = re.sub(r"/\*.*?\*/", " ", java_src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    m = _PUBLIC_TYPE_RE.search(src)
    if m:
        return m.group(1)
    return None


def patch_one_task(task_dir: Path, dry_run: bool) -> str:
    """Patch a single task. Return one of:

    - ``"renamed"``       — file renamed + test.sh rewritten
    - ``"already_v4"``    — already patched
    - ``"already_named"`` — public class is already ``TestSolution``
    - ``"no_test"``       — no ``tests/TestSolution.java``
    - ``"no_public"``     — could not find a public class declaration
    - ``"no_test_sh"``    — no ``tests/test.sh``
    """
    test_java = task_dir / "tests" / "TestSolution.java"
    test_sh = task_dir / "tests" / "test.sh"

    if not test_sh.is_file():
        return "no_test_sh"

    # Check idempotency on test.sh first — the v4 patch may have already
    # been applied (and the .java file already renamed to a different name).
    sh_text = test_sh.read_text(encoding="utf-8", errors="replace")
    if _V4_MARKER in sh_text:
        return "already_v4"

    if not test_java.is_file():
        # The test file may already have been renamed by a partial v4 run.
        # In that case, sh_text would contain `_V4_MARKER` and we'd have
        # returned above — so this is a genuinely-missing test.
        return "no_test"

    java_text = test_java.read_text(encoding="utf-8", errors="replace")
    public_name = find_public_class_name(java_text)
    if public_name is None:
        return "no_public"

    if public_name == "TestSolution":
        # Already correctly named. Still need to drop the v4 marker into
        # test.sh so we don't keep re-checking on subsequent runs.
        new_test_src_path = "/tests/TestSolution.java"
    else:
        new_test_src_path = f"/tests/{public_name}.java"

    if not dry_run:
        # Rename the file (only if needed).
        if public_name != "TestSolution":
            target = task_dir / "tests" / f"{public_name}.java"
            if target.exists():
                # Defensive: don't overwrite a pre-existing file with the same
                # name. Append a suffix if absolutely necessary, but in practice
                # this should never trigger for our corpus.
                target = task_dir / "tests" / f"{public_name}__renamed.java"
                new_test_src_path = f"/tests/{public_name}__renamed.java"
            shutil.move(str(test_java), str(target))

        # Always rewrite test.sh from the template (idempotent).
        new_sh = _PATCHED_TEST_SH_TEMPLATE.format(
            V4_MARKER=_V4_MARKER,
            TEST_SRC_PATH=f'"{new_test_src_path}"',
        )
        test_sh.write_text(new_sh, encoding="utf-8")

    if public_name == "TestSolution":
        return "already_named"
    return "renamed"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Directory of extracted task folders (each with tests/)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--examples", type=int, default=10)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not task_dirs:
        print(f"No task dirs under {root}", file=sys.stderr)
        return 2
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    for i, td in enumerate(task_dirs, 1):
        verdict = patch_one_task(td, args.dry_run)
        counts[verdict] = counts.get(verdict, 0) + 1
        ex = examples.setdefault(verdict, [])
        if len(ex) < args.examples:
            ex.append(td.name)
        if i % 500 == 0 or i == len(task_dirs):
            sample = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"[{i}/{len(task_dirs)}] {sample}", flush=True)

    total = sum(counts.values())
    print(f"\nDone. {total} task dirs processed (dry_run={args.dry_run}).")
    for k in sorted(counts):
        pct = counts[k] / total * 100
        print(f"  {k:<14}: {counts[k]:>5} ({pct:5.1f}%)")
        for name in examples.get(k, []):
            print(f"      {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
