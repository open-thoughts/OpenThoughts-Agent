#!/usr/bin/env python3
"""
exp_rpt_stack-go patcher (v3 -> v4).

QC of v3 (`laion/exp_rpt_stack-go-v3`) lifted whole-dataset solve from
the v2 baseline up to 6.5% (13/200 trials) by adding ``go mod init`` +
``go mod tidy`` (with a ``go get ./...`` fallback) to ``test.sh``. The
remaining 187 v3 failures are dominated by a single failure mode:

    "no required module provides package <X>"  — 98 / 187  (52%)

These are tasks whose ``tests/solution_test.go`` imports private,
niche-org, or otherwise-unfetchable third-party Go packages. ``go mod
tidy`` either can't resolve them at all (private orgs, internal-only
modules) or pulls a wrong version that doesn't expose the symbols the
test expects. Concrete examples from the v3 fail log:

  - ``github.com/openshift/api/...``
  - ``k8s.io/legacy-cloud-providers/...``
  - ``github.com/Azure/azure-sdk-for-go/...``
  - ``github.com/atlassian/smith/...``
  - ``github.com/envoyproxy/go-control-plane/...``

The other 89 failures break down as:

  - 46 (25%)  project-internal ``undefined:`` errors (model-capability,
              not patcher-fixable)
  - 8  (4%)   wrong-fork type-mismatches (e.g. ``github.com/studyzy/
              crypto/ecdsa`` vs the stdlib ``crypto/ecdsa``)
  - 34 (18%)  long-tail other (compiler errors, build-tag misses, ...)
  - 1  (0.5%) Go-runtime test source

Filter
------
The 13 *passing* v3 trials' test files import overwhelmingly from a
small, sandbox-reliable allowlist: the Go stdlib, ``stretchr/testify``,
``google/go-cmp``, and ``gopkg.in/check.v1``. v4 makes that the
structural property of the dataset: drop any task whose
``tests/solution_test.go`` references an import path outside this
allowlist. This intentionally drops some v3 *passers* (e.g. tasks
importing ``github.com/joho/godotenv`` or ``github.com/coinbase/step``)
— the goal here is reliability and a crisp, actionable failure
distribution, not recall.

Allowlist (for ``tests/solution_test.go`` imports):

  - **Go stdlib**: any import path with no dot in its first component
    (e.g. ``fmt``, ``testing``, ``crypto/ecdsa``, ``encoding/json``).
    The Go module-path spec requires a dot in the first component for
    any non-stdlib path, so this is a sound discriminator.
  - **``github.com/stretchr/testify``** and any subpath
    (``assert``, ``require``, ``mock``, ``suite``).
  - **``github.com/google/go-cmp/cmp``** (and any subpath under
    ``github.com/google/go-cmp``) — ubiquitous diff helper.
  - **``gopkg.in/check.v1``** — third common test framework.

Every other dotted import path is treated as "third-party fetchable but
unreliable in our sandbox" and the task is dropped. Be conservative:
when uncertain about a path, drop.

Sampling 500 of the v3 task trees (drawing from ``tasks_success_ya7m9feq``
and the broader v3 extraction) shows ~12-14% of tasks pass this filter.
The dropped tasks are overwhelmingly the cluster of failed v3 trials
listed above (k8s, openshift, Azure SDK, envoyproxy, model-capability
projects with deep internal-package imports, ...).

Idempotency
-----------
Idempotent via a ``"v4_filter": "kept"`` field in each surviving task's
``metadata.json`` (mirrors the v6 stack-junit pattern). Re-running on a
v4-patched task is a no-op: kept tasks are skipped, dropped tasks are
already gone. Dropping is destructive (``shutil.rmtree``).

Apply ON TOP of an already-v3-patched extraction (the v3 patcher's
``test.sh`` body — ``go mod init`` + ``go mod tidy || go get ./...`` — is
preserved verbatim). After patching, upload the surviving subset to
``laion/exp_rpt_stack-go-v4``.

CLI
---
::

    python data/patchers/patch_stack_go_v4_tasks.py \\
        --root /path/to/extracted_tasks_dir \\
        [--dry-run] [--limit N] [--drop-log path.tsv]

Mirrors the convention of ``patch_stack_junit_v6_tasks.py``: destructive
filter, dry-run preview, optional drop log, progress every 200 tasks.

Expected outcome (whole-dataset, post-v4):
  - Survivors: ~12-14% of v3 (~1.2k-1.4k of 9999, depending on the
    full-set distribution vs. the 200-task QC sample).
  - The "no required module provides package" failure mode should
    approach 0 — only stdlib + testify + go-cmp + check.v1 survives,
    all of which ``go mod tidy`` (or even just ``go get``) handles
    reliably.
  - Residual failures will be the ~25% project-internal ``undefined:``
    bucket (genuine model-capability misses, not patcher-fixable) and
    a thin tail of build/runtime issues. Solve rate target: > 30%.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

V4_MARKER_KEY = "v4_filter"
V4_MARKER_VALUE = "kept"

# Match a Go single-line import:
#
#     import "fmt"
#     import _ "github.com/foo/bar"
#     import alias "github.com/foo/bar"
#     import . "github.com/foo/bar"
#
# Group 1 = the import path (without quotes).
_IMPORT_SINGLE_RE = re.compile(
    r'^\s*import\s+(?:[A-Za-z_][\w]*\s+|\.\s+|_\s+)?"([^"]+)"\s*$',
)

# Match the start of an import block:
#
#     import (
#
_IMPORT_BLOCK_OPEN_RE = re.compile(r"^\s*import\s*\(\s*(?://.*)?$")

# Match one entry inside an import block. Same alias forms as single-line,
# but no leading "import" keyword. Group 1 = the path.
#
#     "fmt"
#     _ "github.com/foo/bar"
#     alias "github.com/foo/bar"
#     . "github.com/foo/bar"
#
# Trailing line comments (// ...) are tolerated.
_IMPORT_BLOCK_ENTRY_RE = re.compile(
    r'^\s*(?:[A-Za-z_][\w]*\s+|\.\s+|_\s+)?"([^"]+)"\s*(?://.*)?$',
)


def _import_is_allowed(path: str) -> bool:
    """Return True if a Go import path is on the v4 allowlist."""
    # Go stdlib: the first path component has no dot. The Go module-path
    # spec requires a dot in the first component for non-stdlib paths,
    # so this is a sound discriminator (e.g. ``fmt``, ``crypto/ecdsa``,
    # ``encoding/json`` — all stdlib).
    first = path.split("/", 1)[0]
    if "." not in first:
        return True

    # stretchr/testify — assert, require, mock, suite.
    if path == "github.com/stretchr/testify" or path.startswith(
        "github.com/stretchr/testify/"
    ):
        return True

    # google/go-cmp — ubiquitous diff helper for tests.
    if path == "github.com/google/go-cmp" or path.startswith(
        "github.com/google/go-cmp/"
    ):
        return True

    # gopkg.in/check.v1 — third common Go test framework.
    if path == "gopkg.in/check.v1" or path.startswith("gopkg.in/check.v1/"):
        return True

    return False


def _extract_imports(go_text: str) -> list[str]:
    """Scan a Go source file and return every imported path.

    Handles both single-line ``import "x"`` and multi-line
    ``import ( ... )`` blocks, plus aliased / blank / dot imports. Skips
    line and block comments inside import blocks defensively (rare in
    practice but cheap to handle).
    """
    paths: list[str] = []
    lines = go_text.splitlines()
    i = 0
    n = len(lines)
    in_block = False
    in_block_comment = False

    while i < n:
        line = lines[i]
        i += 1

        # Strip /* ... */ block-comment state (defensive — rare in
        # import blocks but possible).
        if in_block_comment:
            end = line.find("*/")
            if end == -1:
                continue
            line = line[end + 2 :]
            in_block_comment = False

        if not in_block:
            # Skip top-of-file block comments / package decl.
            stripped = line.lstrip()

            # Open of /* ... */ that doesn't close on this line.
            if "/*" in line and "*/" not in line[line.index("/*") + 2 :]:
                in_block_comment = True
                continue

            m = _IMPORT_SINGLE_RE.match(line)
            if m:
                paths.append(m.group(1))
                continue

            if _IMPORT_BLOCK_OPEN_RE.match(line):
                in_block = True
                continue

            # Once we see a non-import top-level decl after the package
            # statement, no more imports will appear in valid Go source.
            # Cheap early-exit: stop on `func`, `type`, `var`, `const`
            # at column zero.
            if stripped.startswith(("func ", "type ", "var ", "const ")):
                break
        else:
            # Inside ``import ( ... )`` block.
            if "/*" in line and "*/" not in line[line.index("/*") + 2 :]:
                in_block_comment = True
                continue

            stripped = line.strip()
            if stripped == ")":
                in_block = False
                continue
            if not stripped or stripped.startswith("//"):
                continue

            m = _IMPORT_BLOCK_ENTRY_RE.match(line)
            if m:
                paths.append(m.group(1))
            # Unrecognized lines inside an import block are tolerated
            # silently (defensive — malformed Go isn't our concern).

    return paths


def _disallowed_imports(go_text: str) -> list[str]:
    """Return every disallowed import path in a Go test source file."""
    return [p for p in _extract_imports(go_text) if not _import_is_allowed(p)]


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
    """Stamp ``v4_filter: kept`` into metadata.json. Creates the file
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
    data[V4_MARKER_KEY] = V4_MARKER_VALUE
    meta_path.write_text(json.dumps(data, indent=2) + "\n")


def filter_one_task(task_dir: Path) -> tuple[str, str]:
    """Decide keep vs drop for one task.

    Returns ``(verdict, reason)``:

    - ``("keep", "already-v4")``    — already marked, no work
    - ``("keep", "")``               — passes the allowlist
    - ``("drop", "no-tests-dir")``   — ``tests/`` missing
    - ``("drop", "no-test-go")``     — ``tests/solution_test.go`` missing
    - ``("drop", "import:<path>")``  — first disallowed import found
    - ``("drop", "err:<msg>")``      — read/parse failure
    """
    # Idempotency: short-circuit on prior v4 stamp.
    meta = _read_metadata(task_dir)
    if meta and meta.get(V4_MARKER_KEY) == V4_MARKER_VALUE:
        return "keep", "already-v4"

    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        return "drop", "no-tests-dir"

    test_go = tests_dir / "solution_test.go"
    if not test_go.is_file():
        return "drop", "no-test-go"

    try:
        text = test_go.read_text(errors="replace")
    except Exception as e:
        return "drop", f"err:read:{test_go.name}:{e}"

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
        "keep_already_v4": 0,
        "drop_no_tests_dir": 0,
        "drop_no_test_go": 0,
        "drop_import": 0,
        "drop_other": 0,
    }
    drop_log_lines: list[str] = []
    examples: dict[str, list[str]] = {}

    total = len(task_dirs)
    for i, td in enumerate(task_dirs, 1):
        verdict, reason = filter_one_task(td)

        if verdict == "keep":
            if reason == "already-v4":
                counts["keep_already_v4"] += 1
            else:
                counts["keep"] += 1
                if not args.dry_run:
                    try:
                        _write_metadata_marker(td)
                    except Exception as e:
                        # Non-fatal: surviving task without marker is
                        # still semantically correct, but loses
                        # idempotency. Surface it.
                        print(
                            f"warn: marker write failed for {td.name}: {e}",
                            file=sys.stderr,
                        )
        else:
            if reason == "no-tests-dir":
                key = "drop_no_tests_dir"
            elif reason == "no-test-go":
                key = "drop_no_test_go"
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

    kept_total = counts["keep"] + counts["keep_already_v4"]
    yld = kept_total / total * 100 if total else 0.0
    print()
    print("=" * 60)
    print(f"Total: {total}  Kept: {kept_total} ({yld:.1f}%)  Dry={args.dry_run}")
    for k in (
        "keep",
        "keep_already_v4",
        "drop_no_tests_dir",
        "drop_no_test_go",
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
