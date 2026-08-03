#!/usr/bin/env python3
"""
exp_rpt_stack-pytest-gpt5mini patcher (DCAgent -> laion v2).

Source dataset: ``DCAgent/exp_rpt_stack-pytest-gpt5mini`` (n=10000 tasks).
QC (n=200): 187/200 infra-ok (0.935), 5 solved (0.025).

This corpus is the SAME ``stack-pytest-synthetic-NNNN`` task family the
v3 / v4 synthetic patchers target — same task scaffolding (test.sh that
``source``s ``/app/.venv``, ``pip install pytest``, optional missing-import
auto-installer with ``|| true``, then runs ``pytest /tests/test_solution.py``)
— just regenerated with ``gpt-5-mini`` instead of ``gpt-5-nano``. The
failure distribution in the QC traces confirms this: ``ModuleNotFoundError``
for numpy (16), pandas (3), django (3), torch (3), homeassistant (3),
azure (4), starlette/transformers/yaml (1 each), repo-private modules
(smartva, ukkeylib, source_paystack, ...) that won't exist in any pip
container, AND ``ImportError: attempted relative import with no known
parent package`` (13).

The 13 ``failures/harbor/`` task dirs are transient
``DaytonaError: Failed to get session command`` infra issues — not
patchable in the data layer.

The dataset's built-in ``MISSING=$(python3 -c ...)`` auto-installer in
``test.sh`` swallows pip errors via ``|| true`` and uses the unmodified
import name as the package name — so ``import yaml`` tries
``pip install yaml`` (fails: package is ``pyyaml``), ``from PIL import``
tries ``pip install PIL`` (fails: package is ``pillow``), and
``import sklearn`` tries ``pip install sklearn`` (deprecated alias).
Solution: pre-resolve the correct PyPI package names at patch time
(via the inverted import-to-package map) and inject an explicit
``pip3 install --break-system-packages <pkgs>`` line into ``test.sh``.

We delegate the actual implementation to the v4 synthetic patcher's
functions (filter passes + ``patch_test_sh`` mutation): same task
family, same scaffold, same fix. This module is a thin driver that
imports and re-exports the v4 logic with a gpt5mini-specific marker so
re-runs are idempotent and don't collide with v4-patched tasks.

CLI mirrors ``patch_stack_pytest_synthetic_v4_tasks.py``::

    python data/patchers/patch_exp_rpt_stack_pytest_gpt5mini_tasks.py \\
        --root /path/to/extracted_tasks_dir \\
        [--dry-run] [--limit N] [--drop-log path.tsv]

Target HF repo: ``laion/exp_rpt_stack-pytest-gpt5mini-v2``.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Reuse the synthetic-v4 filter + mutation logic verbatim — same task
# family, same scaffold. v4's docstring is the canonical design note.
from data.patchers.patch_stack_pytest_synthetic_v4_tasks import (  # noqa: E402
    V4_MARKER,
    filter_one_task,
    _detect_pip_packages,
)


# Distinct marker so this patch doesn't collide with an earlier v4 pass
# applied to a different extraction of the same family. Marker is fenced
# by the same shape of injection block as v4, so patch_test_sh's
# existing ``V4_MARKER in text`` short-circuit still gives us
# idempotency across our OWN re-runs — we just additionally guard
# against a v4 marker being there from a different patcher.
GPT5MINI_MARKER = "# --- laion gpt5mini patch: install detected test imports ---"


def _patch_test_sh_with_marker(test_sh: Path, pkgs: list[str]) -> tuple[bool, str]:
    """Wrapper around v4's ``patch_test_sh`` that uses a gpt5mini-specific
    marker. The injected text is otherwise identical (same pip line, same
    fallback chain).

    Idempotent: if EITHER the v4 marker OR our marker is already present,
    leaves the file alone.
    """
    if not test_sh.is_file():
        return False, "no-test.sh"
    text = test_sh.read_text()
    if GPT5MINI_MARKER in text:
        return False, "already-patched"
    if V4_MARKER in text:
        # Some other run of a different patcher already injected the same
        # install line — leave it alone. Functionally equivalent.
        return False, "already-patched-v4"
    if not pkgs:
        return False, "no-pkgs"

    # Defer to v4's logic; v4 short-circuits on its OWN marker so we
    # temporarily rewrite the marker constant for this call by writing
    # the resolved block ourselves to keep behaviors aligned.
    import re

    pkg_str = " ".join(pkgs)
    block = (
        f"{GPT5MINI_MARKER}\n"
        f"pip3 install --break-system-packages --quiet {pkg_str} 2>/dev/null"
        f" || pip3 install --quiet {pkg_str} 2>/dev/null || true\n"
    )

    lines = text.splitlines(keepends=True)
    pip_pytest_re = re.compile(r"pip3?\s+install[^\n]*\bpytest\b")
    insert_at: int | None = None
    for i, ln in enumerate(lines):
        if pip_pytest_re.search(ln):
            insert_at = i + 1
            break
    if insert_at is None:
        return False, "no-pytest-install-line"

    indent = re.match(r"[ \t]*", lines[insert_at - 1]).group(0)
    indented_block = "".join(indent + ln for ln in block.splitlines(keepends=True))
    new_lines = lines[:insert_at] + [indented_block] + lines[insert_at:]
    test_sh.write_text("".join(new_lines))
    return True, f"patched:{','.join(pkgs)}"


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
        "drop_syntax": 0,
        "drop_import": 0,
        "drop_fixture": 0,
        "drop_relative": 0,
        "drop_other": 0,
        "patched": 0,
        "patch_skipped": 0,
    }
    drop_log_lines: list[str] = []
    examples: dict[str, list[str]] = {}

    total = len(task_dirs)
    for i, td in enumerate(task_dirs, 1):
        verdict, reason = filter_one_task(td)
        if verdict == "keep":
            counts["keep"] += 1
            test_sh = td / "tests" / "test.sh"
            test_solution = td / "tests" / "test_solution.py"
            pkgs = _detect_pip_packages(test_solution)
            if not args.dry_run:
                changed, _why = _patch_test_sh_with_marker(test_sh, pkgs)
                if changed:
                    counts["patched"] += 1
                else:
                    counts["patch_skipped"] += 1
            else:
                if pkgs and test_sh.is_file():
                    text = test_sh.read_text()
                    if GPT5MINI_MARKER not in text and V4_MARKER not in text:
                        counts["patched"] += 1
                    else:
                        counts["patch_skipped"] += 1
                else:
                    counts["patch_skipped"] += 1
        else:
            if reason.startswith("syntax"):
                key = "drop_syntax"
            elif reason.startswith("import"):
                key = "drop_import"
            elif reason.startswith("fixture"):
                key = "drop_fixture"
            elif reason.startswith("relative"):
                key = "drop_relative"
            else:
                key = "drop_other"
            counts[key] += 1
            drop_log_lines.append(f"{td.name}\t{reason}")
            examples.setdefault(key, [])
            if len(examples[key]) < 5:
                examples[key].append(f"{td.name}: {reason}")
            if not args.dry_run:
                shutil.rmtree(td, ignore_errors=True)

        if i % 1000 == 0 or i == total:
            print(f"[{i}/{total}] {counts}", flush=True)

    print()
    print("=" * 60)
    yld = counts["keep"] / total * 100 if total else 0.0
    print(f"Total: {total}  Kept: {counts['keep']} ({yld:.1f}%)  Dry={args.dry_run}")
    for k in ("keep", "drop_syntax", "drop_import", "drop_fixture",
              "drop_relative", "drop_other", "patched", "patch_skipped"):
        v = counts[k]
        print(f"  {k:<14}: {v:>5}")
        for ex in examples.get(k, [])[:5]:
            print(f"      {ex}")

    if args.drop_log and drop_log_lines:
        Path(args.drop_log).write_text("\n".join(drop_log_lines) + "\n")
        print(f"\nDrop log: {args.drop_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
