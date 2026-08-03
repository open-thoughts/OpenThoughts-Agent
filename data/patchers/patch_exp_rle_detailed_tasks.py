#!/usr/bin/env python3
"""
exp_rle_detailed v3 patcher.

Background
----------
DCAgent/exp_rle_detailed is a prompt-engineering treatment of the same
upstream 5,000-task pool used by exp_rle_heavy_padding /
exp_flat25_speed_bonus / exp_flat25_pseudocode / exp_flat25_stackoverflow
(those are covered by ``patch_rle_flat25_tasks.py``).

QC observation (2026-05-14, v2 → v3):
  Smoke test of laion/exp_rle_detailed-v2 (773 surviving tasks after the
  v2 patcher) reported 200/200 infra-ok but only 12/200 (6.0%) solved.
  Re-triage of all 200 traces:

    78  import_error          ← 39%  ImportError / ModuleNotFoundError
                                       at test collection (test.sh installs
                                       only a fixed whitelist; many tests
                                       import beyond that)
    42  errors_only           ← mostly real test fails or downstream import bugs
    18  mixed_pass_fail       ← partial pass — legitimate hard case
    17  all_failed            ← agent solution didn't satisfy
    14  collect_error_other   ← ImportError name-not-found, torch CUDA blowup,
                                       missing fixture / file
    12  solved
    12  other                 ← misc fixture / mark errors
     4  collect_zero_other    ← `collected 0 items` (no tests defined)
     3  rubber_stamp_all_skipped  ← `pytest.importorskip("foo")` for a
                                       package not in whitelist → all tests
                                       skipped → verifier exits 0 only if
                                       the test runner is happy; here they
                                       returned exit nonzero

  The v2 evaluator (= shared `patch_rle_flat25_tasks.evaluate_task`) had
  three structural gaps:

  1. **Submodule imports.** It only checks the top-level package name
     against the container whitelist. `from pandas.util.testing import …`
     passes because `pandas` is whitelisted, but
     `pandas.util.testing` was removed in pandas 2.x. Same for
     `sklearn.utils.testing` (renamed to `_testing`), `tensorflow.python`
     (private — not shipped in pip wheels), `ddtrace.compat` (removed in
     ddtrace 1.x+), and `pandas.util._test_decorators` /
     `pandas._testing` private namespaces.

  2. **Bare local-name imports** masked by ``instr.md`` mentions.
     Tests routinely import ``from tests import X``, ``from helpers import
     Y``, ``from main import Z`` — bare-name modules the agent is supposed
     to create at /app/. Because instruction.md inevitably mentions
     "tests" or "helpers" as English words, the v2 evaluator exempted
     them. These near-always fail at verifier time because /app
     contains nothing by that name.

  3. **`pytest.importorskip(unwhitelisted_module)`** at module level.
     If the imported module isn't in the container whitelist, pytest
     skips every test in the file. The verifier sees `0 items / N skipped`
     and reports reward=0. We can detect the AST pattern and drop those
     tasks — they are effectively rubber-stamp impossible.

  Plus several smaller gaps:

  4. **`tzdata` / `google.api_core`** — not in the container whitelist.
  5. **pytest plugins** (`pytest.mark.asyncio` → pytest-asyncio,
     `pytest.lazy_fixture` → pytest-lazy-fixture) — referenced in tests
     but not installed.
  6. **Whole-file unconditional skip** — `pytestmark = pytest.mark.skip(...)`
     or every test func wearing `@pytest.mark.skip` — verifier rubber-stamps
     these (one case in v2 produced a fake reward=1.0). Drop.

Fix
---
v3 keeps the v2 import-allowlist check (via the shared ``evaluate_task``)
*as a first pass*, then layers on:

  • a blocklist of known-broken submodules (``KNOWN_BROKEN_SUBMODULES``)
    that triggers on any deep import path matching the prefix;
  • a blocklist of bare local-package names that overrides the
    instr-mentioned exemption (``LOCAL_NAME_BLOCKLIST``);
  • detection of ``pytest.importorskip("X")`` calls for X outside the
    container whitelist;
  • detection of ``pytestmark = pytest.mark.skip(...)`` and the
    "every-test-decorated-with-@skip" rubber-stamp pattern.

Idempotency: the patcher only removes task directories.

Usage
-----
  python data/patchers/patch_exp_rle_detailed_tasks.py --root <tasks_dir>
  python data/patchers/patch_exp_rle_detailed_tasks.py --root <tasks_dir> --dry-run
  python data/patchers/patch_exp_rle_detailed_tasks.py --root <tasks_dir> --limit 200
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

# Reuse the shared evaluator from the rle_flat25 patcher.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from patch_rle_flat25_tasks import (  # noqa: E402
    ALWAYS_INSTALLED,
    PIP_WHITELIST,
    STDLIB,
    _instruction_tokens,
    _top_level_imports,
)


# ---------------------------------------------------------------------------
# v3 additional config
# ---------------------------------------------------------------------------

# Submodule paths that are inside an installed package but don't exist
# (removed in modern versions, private namespace not shipped, etc.).
# Match if the imported dotted path equals an entry or starts with
# ``<entry>.``.
KNOWN_BROKEN_SUBMODULES: frozenset[str] = frozenset({
    # pandas
    "pandas.util.testing",           # removed in pandas 2.x (→ pandas.testing)
    "pandas.tests",                  # private — not shipped in wheels
    "pandas.core.index",             # private
    "pandas.compat",                 # mostly private; specific symbols vanish
    "pandas.util._test_decorators",  # private — flaky across versions
    "pandas._testing",               # private — symbols renamed/removed
    # scikit-learn
    "sklearn.utils.testing",         # removed (→ sklearn.utils._testing)
    # tensorflow / keras
    "tensorflow.python",             # private internals
    "keras.utils.test_utils",        # removed in modern keras
    "keras.utils.testing_utils",     # removed
    # numpy
    "numpy.core",                    # private (removed/relocated in numpy 2.x)
    "numpy.compat",                  # removed in numpy 2.x
    "numpy.lib.shape_base",          # private
    "numpy.core._rational_tests",    # internal
    "numpy.core._multiarray_tests",  # internal
    # ddtrace
    "ddtrace.compat",                # removed in ddtrace 1.x+
    # pandas test test-decorators namespace variants
    "test.utils",                    # ambiguous — usually agent-local; safe drop
})

# Bare local-package names that override the instr-mentioned exemption.
# Tests doing ``from <name> import X`` for these always need the agent to
# materialize a /app/<name>.py, which rarely happens reliably. Even when
# instruction.md mentions the word (e.g. "place your tests under tests/"),
# the test-time import fails.
LOCAL_NAME_BLOCKLIST: frozenset[str] = frozenset({
    "tests", "test", "tests_helpers",
    "helpers", "helper",
    "utils", "util",
    "models", "model",
    "main",
    "app", "apps",
    "src", "source",
    "rules", "rule",
    "section", "sections",
    "api", "apis",
    "common", "core", "lib",
    "config", "settings", "conf",
    "view", "views",
    "form", "forms",
    "schema", "schemas",
})

# Top-level pypi modules the container does NOT pre-install (verified by
# failure traces). Top-level imports of these are bad even if instr.md
# mentions them (e.g. "google" appears in many tutorials).
EXTRA_MISSING_TOPLEVEL: frozenset[str] = frozenset({
    "tzdata",
    "google",      # google.api_core, google.cloud, etc.
})

# pytest plugins whose decorators / fixtures appear in tests but aren't
# installed in the container.
KNOWN_PYTEST_PLUGINS: dict[str, str] = {
    "asyncio": "pytest-asyncio",
    "trio": "pytest-trio",
    "twisted": "pytest-twisted",
    "django": "pytest-django",
    "tornado": "pytest-tornado",
    "app_settings": "guillotina-pytest",  # extremely niche
}

_IMPORTORSKIP_RE = re.compile(r"pytest\.importorskip\s*\(\s*['\"]([^'\"]+)['\"]")
_PYTEST_MARK_RE = re.compile(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)")
_PYTESTMARK_SKIP_RE = re.compile(
    r"^\s*pytestmark\s*=\s*pytest\.mark\.skip\s*[\(\s]", re.M
)
# Match @pytest.mark.skip but NOT @pytest.mark.skipif (conditional)
_SKIP_DECORATOR_RE = re.compile(r"@pytest\.mark\.skip(?!\w)")
_TEST_FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_\w+", re.M)


def _deep_import_paths(tree: ast.AST) -> set[str]:
    """Return full dotted-path imports (e.g. 'pandas.util.testing').

    Unlike ``_top_level_imports`` (which strips to the first segment), this
    preserves the full ``a.b.c`` so we can match against
    ``KNOWN_BROKEN_SUBMODULES``.
    """
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    paths.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative — local-only
            if node.module:
                paths.add(node.module)
    return paths


def _matches_broken_submodule(path: str) -> str | None:
    """Return the matching broken-submodule entry, or None."""
    for broken in KNOWN_BROKEN_SUBMODULES:
        if path == broken or path.startswith(broken + "."):
            return broken
    return None


def evaluate_task_v3(task_dir: Path) -> dict:
    """v3 verdict for a task.

    Returns ``{"kept": bool, "reason": str, "details": list[str]}``.

    Drop reasons (in ``details``):
      • ``top:<name>(local)``       — bare local-package name
      • ``top:<name>(missing)``     — known-missing top-level package
      • ``top:<name>``               — top-level package not in whitelist /
                                       instr.md (same as v2)
      • ``submod:<path>``            — known-broken submodule
      • ``plugin:<name>``            — pytest plugin referenced via mark
      • ``plugin:lazy_fixture``      — pytest-lazy-fixture references
      • ``importorskip:<mod>``       — module-level importorskip of an
                                       un-whitelisted package
      • ``rubber_stamp:pytestmark_skip``  — whole-file unconditional skip
      • ``rubber_stamp:all_skip``    — every test func wears @pytest.mark.skip
    """
    test_file = task_dir / "tests" / "test_solution.py"
    instr_file = task_dir / "instruction.md"

    if not test_file.exists():
        return {"kept": False, "reason": "no_test_file", "details": []}

    try:
        src = test_file.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"kept": False, "reason": f"read_error:{type(exc).__name__}", "details": []}

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"kept": False, "reason": "syntax_error", "details": [str(exc)[:80]]}

    imports_top = _top_level_imports(tree)
    imports_deep = _deep_import_paths(tree)
    instr_text = (
        instr_file.read_text(encoding="utf-8", errors="replace")
        if instr_file.exists() else ""
    )
    instr_toks = _instruction_tokens(instr_text)

    bad: list[str] = []

    # (1) top-level import availability
    for name in sorted(imports_top):
        if name in STDLIB:
            continue
        if name in ALWAYS_INSTALLED:
            continue
        if name in PIP_WHITELIST:
            continue
        if name in EXTRA_MISSING_TOPLEVEL:
            bad.append(f"top:{name}(missing)")
            continue
        if name in LOCAL_NAME_BLOCKLIST:
            bad.append(f"top:{name}(local)")
            continue
        if name.lower() in instr_toks:
            continue
        bad.append(f"top:{name}")

    # (2) submodule blocklist
    for path in sorted(imports_deep):
        m = _matches_broken_submodule(path)
        if m is not None:
            bad.append(f"submod:{path}")

    # (3) pytest plugins via @pytest.mark.X
    for m in _PYTEST_MARK_RE.finditer(src):
        name = m.group(1)
        if name in KNOWN_PYTEST_PLUGINS:
            bad.append(f"plugin:{name}")
            break  # one is enough — don't multi-count the same file
    # pytest.lazy_fixture (pytest-lazy-fixture not installed)
    if "pytest.lazy_fixture" in src or "pytest_lazy_fixtures" in src:
        bad.append("plugin:lazy_fixture")

    # (4) pytest.importorskip("foo") with foo outside whitelist
    for m in _IMPORTORSKIP_RE.finditer(src):
        mod = m.group(1).split(".")[0]
        if (mod in STDLIB or mod in PIP_WHITELIST or mod in ALWAYS_INSTALLED):
            continue
        bad.append(f"importorskip:{m.group(1)}")
        break  # one is enough

    # (5) rubber-stamp patterns
    if _PYTESTMARK_SKIP_RE.search(src):
        bad.append("rubber_stamp:pytestmark_skip")
    else:
        tests_count = len(_TEST_FUNC_RE.findall(src))
        skip_decorators = len(_SKIP_DECORATOR_RE.findall(src))
        # Only flag if there are tests AND every one is decorated with
        # an unconditional skip. skipif (conditional) is fine.
        if tests_count > 0 and skip_decorators >= tests_count:
            bad.append("rubber_stamp:all_skip")

    if bad:
        return {"kept": False, "reason": "drop", "details": bad}
    return {"kept": True, "reason": "", "details": []}


# Back-compat name (legacy callers import `evaluate_task`)
evaluate_task = evaluate_task_v3


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

    print(f"[patch_exp_rle_detailed v3] inspecting {len(task_dirs)} tasks under {root}")

    no_test_dropped = 0
    syntax_dropped = 0
    drop_counts: dict[str, int] = {}
    kept = 0

    bad_samples: list[tuple[str, list[str]]] = []
    verdicts: dict[str, dict] = {}

    for td in task_dirs:
        v = evaluate_task_v3(td)
        verdicts[td.name] = v
        if v["kept"]:
            kept += 1
            continue
        reason = v["reason"]
        if reason == "no_test_file":
            no_test_dropped += 1
        elif reason == "syntax_error" or reason.startswith("read_error"):
            syntax_dropped += 1
        else:
            for d in v["details"]:
                prefix = d.split(":", 1)[0]
                drop_counts[prefix] = drop_counts.get(prefix, 0) + 1
            if len(bad_samples) < args.show_bad:
                bad_samples.append((td.name, v["details"]))

    total = len(task_dirs)

    print()
    print(f"[patch_exp_rle_detailed v3] total:          {total}")
    print(f"[patch_exp_rle_detailed v3] no test file:   {no_test_dropped}")
    print(f"[patch_exp_rle_detailed v3] syntax errors:  {syntax_dropped}")
    if drop_counts:
        print("[patch_exp_rle_detailed v3] drop reasons (by detail prefix):")
        for k, v in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {v:5d}  {k}")
    print(f"[patch_exp_rle_detailed v3] kept:           {kept}")

    if bad_samples:
        print()
        print(f"[patch_exp_rle_detailed v3] sample drops (first {len(bad_samples)}):")
        for tid, det in bad_samples:
            print(f"  {tid}: {det}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_exp_rle_detailed v3] wrote per-task verdicts: {args.report_json}")

    if args.dry_run:
        print("[patch_exp_rle_detailed v3] dry-run: not deleting dropped tasks.")
        return

    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_exp_rle_detailed v3] removed {removed} dropped task directories.")


if __name__ == "__main__":
    main()
