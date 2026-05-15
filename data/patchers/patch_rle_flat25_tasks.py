#!/usr/bin/env python3
"""
Filter rle / flat25 tasks (5k-task-pool variants) to drop tasks whose
``tests/test_solution.py`` cannot run in the daytona sandbox image.

Background
----------
DCAgent/exp_rle_heavy_padding, exp_flat25_speed_bonus, exp_flat25_pseudocode,
and exp_flat25_stackoverflow are four prompt-engineering treatments of the
same upstream 5,000-task pool. QC sampling found ~9/10 traces fail because
``test_solution.py`` imports random third-party packages that are not in the
container image and are usually unrelated to the task description (e.g. a
"Flask Blueprint" task whose tests import ``from circuit import Circuit``).

The container image (``environment/Dockerfile``) is just ``python3 + pytest``.
The runtime hook (``tests/test.sh``) pip-installs a fixed whitelist of
common libraries before running the test:

    requests numpy pandas scipy scikit-learn sklearn torch tensorflow keras
    httpx aiohttp ddtrace django flask fastapi matplotlib seaborn pillow
    pydantic pytest-mock requests-mock faker pyyaml pytz cryptography bcrypt
    hypothesis

Filter logic
------------
For each task:
  1. ``py_compile`` ``tests/test_solution.py`` → drop on syntax error.
  2. AST-parse top-level imports. Drop the task if ANY top-level import is
     not in:
       - Python stdlib (``sys.stdlib_module_names``)
       - The container's pip whitelist (above)
       - A token mentioned literally in ``instruction.md`` (i.e. the task
         description names the module as part of the requirement).
  3. Otherwise keep the task untouched. The patcher never mutates tasks; it
     only drops them.

Usage
-----
    python data/patchers/patch_rle_flat25_tasks.py --root /path/to/tasks_dir
    python data/patchers/patch_rle_flat25_tasks.py --root /path/to/tasks_dir --dry-run
    python data/patchers/patch_rle_flat25_tasks.py --root /path/to/tasks_dir --limit 200
"""

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import re
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

# Python standard library modules (CPython >= 3.10 has sys.stdlib_module_names)
STDLIB: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", set()))

# Always-installed in the container (pytest is installed unconditionally,
# plus pytest's own runtime helpers).
ALWAYS_INSTALLED: frozenset[str] = frozenset({
    "pytest",
    "_pytest",
    "py",                # legacy pytest namespace
    "pluggy",
    "iniconfig",
    "packaging",
    "tomli",
    "exceptiongroup",
})

# Whitelist of pip packages that ``tests/test.sh`` installs before pytest runs.
# Source of truth: the WHITELIST loop in environment/test.sh of these datasets.
# Map pip-name → top-level import name(s).
WHITELIST_PIP_TO_IMPORTS: dict[str, tuple[str, ...]] = {
    "requests": ("requests",),
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "scipy": ("scipy",),
    "scikit-learn": ("sklearn",),
    "sklearn": ("sklearn",),
    "torch": ("torch",),
    "tensorflow": ("tensorflow", "tf"),
    "keras": ("keras",),
    "httpx": ("httpx",),
    "aiohttp": ("aiohttp",),
    "ddtrace": ("ddtrace",),
    "django": ("django",),
    "flask": ("flask",),
    "fastapi": ("fastapi",),
    "matplotlib": ("matplotlib",),
    "seaborn": ("seaborn",),
    "pillow": ("PIL",),
    "pydantic": ("pydantic",),
    "pytest-mock": ("pytest_mock",),
    "requests-mock": ("requests_mock",),
    "faker": ("faker",),
    "pyyaml": ("yaml",),
    "pytz": ("pytz",),
    "cryptography": ("cryptography",),
    "bcrypt": ("bcrypt",),
    "hypothesis": ("hypothesis",),
}

PIP_WHITELIST: frozenset[str] = frozenset(
    name
    for imports in WHITELIST_PIP_TO_IMPORTS.values()
    for name in imports
)

# Modules whose presence in instruction.md is sometimes only via dotted path
# (e.g. "ddtrace/compat.py" → top-level package "ddtrace"). We strip slashes
# and dots when checking instruction.md mentions.

# Heuristic: identifier-like tokens in instruction.md. The check is
# "does the top-level import name appear as a substring/token in instruction.md".
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_level_imports(tree: ast.AST) -> set[str]:
    """Return the set of top-level package names imported by an AST module.

    For ``import a.b.c`` and ``from a.b import c`` we return ``a``.
    For ``from . import x`` (relative) we return nothing (relative imports
    can't reach an external package).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import — local package only
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def _instruction_tokens(text: str) -> set[str]:
    """Lowercased identifier-like tokens from an instruction.md."""
    return {t.lower() for t in TOKEN_RE.findall(text)}


def evaluate_task(task_dir: Path) -> dict:
    """Return a per-task verdict dict.

    Keys:
      kept: bool  — task survives the filter
      reason: str — short reason if dropped
      bad_imports: list[str] — imports that triggered the drop (if any)
    """
    test_file = task_dir / "tests" / "test_solution.py"
    instruction_file = task_dir / "instruction.md"

    if not test_file.exists():
        return {"kept": False, "reason": "no_test_file", "bad_imports": []}

    # 1. py_compile
    try:
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
            py_compile.compile(
                str(test_file), cfile=tmp.name, doraise=True
            )
    except py_compile.PyCompileError as exc:
        return {"kept": False, "reason": f"syntax_error: {exc.msg.splitlines()[0][:80]}", "bad_imports": []}
    except Exception as exc:
        return {"kept": False, "reason": f"compile_error: {type(exc).__name__}", "bad_imports": []}

    # 2. AST imports
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return {"kept": False, "reason": f"ast_syntax: {exc.msg[:60]}", "bad_imports": []}

    imports = _top_level_imports(tree)

    # 3. Instruction.md tokens
    instr_tokens: set[str] = set()
    if instruction_file.exists():
        instr_tokens = _instruction_tokens(
            instruction_file.read_text(encoding="utf-8", errors="replace")
        )

    bad: list[str] = []
    for name in sorted(imports):
        if name in STDLIB:
            continue
        if name in ALWAYS_INSTALLED:
            continue
        if name in PIP_WHITELIST:
            continue
        # Mentioned in the task description (case-insensitive token match)
        if name.lower() in instr_tokens:
            continue
        bad.append(name)

    if bad:
        return {"kept": False, "reason": "missing_import", "bad_imports": bad}

    return {"kept": True, "reason": "", "bad_imports": []}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    print(f"[patch_rle_flat25] inspecting {len(task_dirs)} tasks under {root}")

    syntax_dropped = 0
    no_test_dropped = 0
    import_dropped = 0
    kept = 0

    bad_import_samples: list[tuple[str, list[str]]] = []
    verdicts: dict[str, dict] = {}

    for td in task_dirs:
        v = evaluate_task(td)
        verdicts[td.name] = v
        if v["kept"]:
            kept += 1
            continue
        if v["reason"] == "no_test_file":
            no_test_dropped += 1
        elif v["reason"].startswith("syntax_error") or v["reason"].startswith("ast_syntax") or v["reason"].startswith("compile_error"):
            syntax_dropped += 1
        elif v["reason"] == "missing_import":
            import_dropped += 1
            if len(bad_import_samples) < args.show_bad:
                bad_import_samples.append((td.name, v["bad_imports"]))

    total = len(task_dirs)
    syntax_passed = total - syntax_dropped - no_test_dropped
    import_passed = syntax_passed - import_dropped

    print()
    print(f"[patch_rle_flat25] total:          {total}")
    print(f"[patch_rle_flat25] no test file:   {no_test_dropped}")
    print(f"[patch_rle_flat25] syntax errors:  {syntax_dropped}")
    print(f"[patch_rle_flat25] syntax passed:  {syntax_passed}")
    print(f"[patch_rle_flat25] import dropped: {import_dropped}")
    print(f"[patch_rle_flat25] import passed:  {import_passed}")
    print(f"[patch_rle_flat25] kept:           {kept}")

    if bad_import_samples:
        print()
        print(f"[patch_rle_flat25] sample bad-import drops (first {len(bad_import_samples)}):")
        for tid, bad in bad_import_samples:
            print(f"  {tid}: {bad}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_rle_flat25] wrote per-task verdicts: {args.report_json}")

    if args.dry_run:
        print("[patch_rle_flat25] dry-run: not deleting dropped tasks.")
        return

    # Apply: remove dropped task directories.
    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_rle_flat25] removed {removed} dropped task directories.")


if __name__ == "__main__":
    main()
