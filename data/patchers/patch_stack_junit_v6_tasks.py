#!/usr/bin/env python3
"""
exp_rpt_stack-junit patcher (v5 -> v6).

QC of v5 (`laion/exp_rpt_stack-junit-v5`) lifted whole-dataset solve from
0% to 3.5% by fixing the gawk-3-arg `match()` parsing bug. The remaining
96.5% failures are *all* `javac error: package X does not exist`: the
test files import third-party libraries that the verifier classpath
doesn't ship — Spring, Netty, HBase, Mockito, TestNG, Reactor, Neo4j,
Akka, AssertJ, Apache Hadoop, Apache Dubbo, etc. The verifier ships only
``junit-platform-console-standalone-1.10.0.jar`` (which transitively
provides JUnit Jupiter, JUnit Vintage / 4.x, JUnit 3 ``junit.framework``
shim, and Hamcrest core). The model has no realistic path to satisfy
those tests because the harness builds its classpath from that single
jar.

The 7 passing v5 trials (out of 200) are tasks whose test files import
*only* ``org.junit.*``, ``java.*`` / ``java.util.*``, and at most one
project-internal class. v6 makes that the structural property of the
dataset: drop any task whose ``tests/*.java`` references a package
outside the verifier's classpath.

Filter
------
Drop a task if **any** of its ``tests/*.java`` files contains an
``import`` for a fully-qualified package not in this allowlist:

  - ``java.*`` / ``javax.*``                 — JDK / Java EE shim, always present
  - ``org.junit.*``                          — JUnit 4.x and JUnit 5 (Jupiter, Vintage, Platform)
  - ``junit.*`` (e.g. ``junit.framework.*``) — JUnit 3 shim bundled with Vintage
  - ``org.hamcrest.*``                       — bundled in junit-platform-console-standalone
  - Single-component imports (``import Foo;``) — project-internal,
    the model can implement these under ``/app``.

Ranges aside, this matches what the v5 console-standalone jar actually
exposes on the classpath (verified against the v5 Dockerfile, which
ships ``junit-platform-console-standalone-1.10.0.jar`` and nothing else
beyond Maven + the JDK).

Sampling 500 of the 9999 v5 task trees showed ~10.6% of tasks pass this
filter (roughly 1.0k surviving tasks). The dropped tasks are an even
mix of Apache (HBase, Hadoop, Lucene, Camel, Dubbo, ...), Spring,
Mockito/AssertJ-only test stacks, Android, and a long tail of
project-specific frameworks (zipkin, herddb, marquez, gaffer, ...).

Idempotency
-----------
Idempotent via a ``"v6_filter": "kept"`` field in each surviving task's
``metadata.json``. Re-running on a v6-patched task is a no-op (we don't
re-evaluate, we just skip). Dropping is destructive (``shutil.rmtree``);
re-runs see only kept tasks and leave them alone.

Apply ON TOP of an already-v5-patched extraction (the v5 patcher's
test.sh body is preserved verbatim — we don't touch test.sh). After
patching, upload to ``laion/exp_rpt_stack-junit-v6``.

CLI
---
::

    python data/patchers/patch_stack_junit_v6_tasks.py \\
        --root /path/to/extracted_tasks_dir \\
        [--dry-run] [--limit N] [--drop-log path.tsv]

Mirrors the convention of ``patch_stack_pytest_synthetic_v3_tasks.py``:
destructive filter, dry-run preview, optional drop log for diagnostics.
Progress prints every 200 tasks (the spec says 200 here; the v5 patcher
prints every 1000 — we go finer because a destructive filter benefits
from earlier visibility into the kill rate).

Expected outcome (whole-dataset, post-v6):
  - Survivors: ~10-11% of v5 (~1.0k of 9999).
  - The 96.5% javac fails should drop accordingly: only tasks whose
    test imports are inside the v5 jar's classpath remain, so
    ``javac: package X does not exist`` should approach 0.
  - The new failure modes will be (a) genuine model-capability misses
    on tasks whose test file references a project-internal class with
    a non-trivial spec, and (b) the residual Daytona-side
    AgentSetupTimeoutErrors (not patcher-fixable, ~14% of trials in
    v5 QC).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

V6_MARKER_KEY = "v6_filter"
V6_MARKER_VALUE = "kept"

# Match a Java import line (single-line): captures the FQN (no trailing `.*`).
#
#   import com.foo.Bar;
#   import static com.foo.Bar.baz;
#   import   org.junit.jupiter.api.* ;
#
# Group 1 = the dotted name (without trailing ".*", without trailing ";").
_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:static\s+)?([\w.]+?)(?:\.\*)?\s*;",
)


def _import_is_allowed(fqn: str) -> bool:
    """Return True if a Java import FQN is on the v6 allowlist."""
    # Single-component (``import Foo;``) — project-internal; the model
    # can implement this under /app. Java requires top-level imports to
    # be in a package, so single-component imports are vanishingly rare
    # in real code, but harmless if present.
    if "." not in fqn:
        return True

    top = fqn.split(".", 1)[0]

    # JDK
    if top in ("java", "javax"):
        return True

    # JUnit 5 (Jupiter, Vintage, Platform) and JUnit 4
    if fqn == "org.junit" or fqn.startswith("org.junit."):
        return True

    # Hamcrest core/library — bundled in junit-platform-console-standalone
    if fqn == "org.hamcrest" or fqn.startswith("org.hamcrest."):
        return True

    # JUnit 3 ``junit.framework.*`` shim (re-exported by JUnit Vintage)
    if top == "junit":
        return True

    return False


def _disallowed_imports(java_text: str) -> list[str]:
    """Return the FQN of every disallowed import in a Java source file."""
    bad: list[str] = []
    for line in java_text.splitlines():
        m = _IMPORT_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        if not _import_is_allowed(name):
            bad.append(name)
    return bad


def _read_metadata(task_dir: Path) -> dict | None:
    """Return parsed metadata.json, or None if missing/unparseable."""
    meta = task_dir / "metadata.json"
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text())
    except Exception:
        return None


def _write_metadata_marker(task_dir: Path) -> None:
    """Stamp ``v6_filter: kept`` into metadata.json. Creates the file
    if missing (preserving any existing keys)."""
    meta_path = task_dir / "metadata.json"
    data: dict = {}
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text())
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    data[V6_MARKER_KEY] = V6_MARKER_VALUE
    meta_path.write_text(json.dumps(data, indent=2) + "\n")


def filter_one_task(task_dir: Path) -> tuple[str, str]:
    """Decide keep vs drop for one task.

    Returns ``(verdict, reason)``:

    - ``("keep", "already-v6")``    — already marked, no work
    - ``("keep", "")``               — passes the allowlist
    - ``("drop", "no-tests-dir")``   — ``tests/`` missing
    - ``("drop", "no-test-java")``   — ``tests/`` has no ``*.java``
    - ``("drop", "import:<fqn>")``   — first disallowed import found
    - ``("drop", "err:<msg>")``      — read/parse failure
    """
    # Idempotency: short-circuit on prior v6 stamp.
    meta = _read_metadata(task_dir)
    if meta and meta.get(V6_MARKER_KEY) == V6_MARKER_VALUE:
        return "keep", "already-v6"

    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        return "drop", "no-tests-dir"

    java_files = sorted(tests_dir.glob("*.java"))
    if not java_files:
        return "drop", "no-test-java"

    # Fail-fast on the first disallowed import across any of the test
    # files in tests/. (In practice these tasks have exactly one test
    # .java; we tolerate more for safety.)
    for jf in java_files:
        try:
            text = jf.read_text(errors="replace")
        except Exception as e:
            return "drop", f"err:read:{jf.name}:{e}"
        bad = _disallowed_imports(text)
        if bad:
            return "drop", f"import:{bad[0]}"

    return "keep", ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--drop-log", type=str, default=None)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    counts = {
        "keep": 0,
        "keep_already_v6": 0,
        "drop_no_tests_dir": 0,
        "drop_no_test_java": 0,
        "drop_import": 0,
        "drop_other": 0,
    }
    drop_log_lines: list[str] = []
    examples: dict[str, list[str]] = {}

    total = len(task_dirs)
    for i, td in enumerate(task_dirs, 1):
        verdict, reason = filter_one_task(td)

        if verdict == "keep":
            if reason == "already-v6":
                counts["keep_already_v6"] += 1
            else:
                counts["keep"] += 1
                if not args.dry_run:
                    try:
                        _write_metadata_marker(td)
                    except Exception as e:
                        # Non-fatal: surviving task without marker is
                        # still semantically correct, but loses
                        # idempotency. Surface it.
                        print(f"warn: marker write failed for {td.name}: {e}",
                              file=sys.stderr)
        else:
            if reason == "no-tests-dir":
                key = "drop_no_tests_dir"
            elif reason == "no-test-java":
                key = "drop_no_test_java"
            elif reason.startswith("import:"):
                key = "drop_import"
            else:
                key = "drop_other"
            counts[key] += 1
            drop_log_lines.append(f"{td.name}\t{reason}")
            examples.setdefault(key, [])
            if len(examples[key]) < 5:
                examples[key].append(f"{td.name}: {reason}")
            if not args.dry_run:
                shutil.rmtree(td, ignore_errors=True)

        if i % 200 == 0 or i == total:
            print(f"[{i}/{total}] {counts}", flush=True)

    kept_total = counts["keep"] + counts["keep_already_v6"]
    yld = kept_total / total * 100 if total else 0.0
    print()
    print("=" * 60)
    print(f"Total: {total}  Kept: {kept_total} ({yld:.1f}%)  Dry={args.dry_run}")
    for k in (
        "keep",
        "keep_already_v6",
        "drop_no_tests_dir",
        "drop_no_test_java",
        "drop_import",
        "drop_other",
    ):
        v = counts[k]
        print(f"  {k:<22}: {v:>5}")
        for ex in examples.get(k, [])[:5]:
            print(f"      {ex}")

    if args.drop_log and drop_log_lines:
        Path(args.drop_log).write_text("\n".join(drop_log_lines) + "\n")
        print(f"\nDrop log: {args.drop_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
