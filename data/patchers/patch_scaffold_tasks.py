#!/usr/bin/env python3
"""
Patch every instruction.md under a Harbor `exp_rpt_scaffold` task tree so the
agent knows which file path to write its implementation to.

The bug
-------
Task prompts say "modify the scaffold under /app/" or "modify the existing
module" but the verifier's test does `from <module> import ...` with a
specific module name (e.g. `from solution import ...`,
`import app.h245_fsm_sim as h245`). The agent has no way to know the required
module name, so most sampled traces had reward=0 due to spec ambiguity.

The fix
-------
For each task dir we:

  1. Parse `tests/test_scaffold.py` and find the FIRST non-stdlib top-level
     `from X import ...` or `import X` statement. The first dotted component
     is the required module path X. (We skip imports inside functions/conditional
     blocks since those are typically dynamic loaders.)

  2. Append a single line to `instruction.md` (the original content is
     preserved):

         Important: place your implementation at `/app/<module_path>.py` so
         the tests can `import <X>`.

     where `<module_path>` is `X` with dots converted to slashes. For
     example, `app.h245_fsm_sim` becomes `app/h245_fsm_sim.py` and `solution`
     becomes `solution.py`.

If a task has no parseable top-level user-module import (e.g. the test uses
`importlib.util` to dynamically locate the file, or the test file is missing),
the task is skipped and counted under `n_skipped`.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Optional

# Modules that count as "framework / stdlib" — we never treat them as the
# user's target module. sys.stdlib_module_names already covers the stdlib;
# the extra entries below cover common test-time imports that are not stdlib
# (pytest et al.) and are never the agent's implementation file.
EXTRA_FRAMEWORK_MODULES: set[str] = {
    "pytest",
    "_pytest",
    "numpy",
    "np",
    "scipy",
    "pandas",
    "torch",
    "tensorflow",
    "tf",
    "yaml",
    "toml",
    "click",
    "typer",
    "requests",
    "httpx",
    "hypothesis",
    "freezegun",
    "mock",
    "pydantic",
    "attr",
    "attrs",
    "dataclasses_json",
    "tabulate",
    "sympy",
    "sklearn",
    "matplotlib",
    "PIL",
    "cv2",
    "h5py",
    "pyarrow",
    "fastapi",
    "starlette",
    "flask",
    "django",
    "sqlalchemy",
    "redis",
    "boto3",
    "transformers",
    "datasets",
    "tqdm",
    "rich",
    "lxml",
    "bs4",
    "beautifulsoup4",
    "openai",
    "anthropic",
    "regex",
    "dateutil",
    "pytz",
    "tzlocal",
    "joblib",
    "networkx",
    "pygame",
    "fastapi",
    "uvicorn",
    "starlette",
    "aiohttp",
    "asyncio",  # also stdlib, defensive
    "ujson",
    "orjson",
    "msgpack",
}

FRAMEWORK_MODULES: set[str] = set(sys.stdlib_module_names) | EXTRA_FRAMEWORK_MODULES

PATCH_MARKER = "Important: place your implementation at"


def _is_framework(top_module: str) -> bool:
    """Return True if `top_module` is stdlib or a known test-framework module."""
    return top_module in FRAMEWORK_MODULES


def find_target_module(test_text: str) -> Optional[str]:
    """
    Parse `test_text` and return the first non-framework module path X that
    appears in `import X` or `from X import ...` anywhere in the AST (in
    document order). This includes:
      - top-level imports
      - imports inside `try/except` blocks (a common scaffold-test pattern:
        try the canonical name, fall back to `app.<canonical>`)
      - imports inside `if/else`, `with`, function bodies, etc.

    We deliberately ignore relative imports (level > 0) and `import` calls
    that go through `importlib.import_module(...)` since those typically
    represent dynamic loaders that try multiple candidates.

    Returns None if no literal `import X` / `from X import` for a user module
    is found (e.g. the test uses only `importlib.util.spec_from_file_location`
    against a heuristically located file).
    """
    try:
        tree = ast.parse(test_text)
    except SyntaxError:
        return None

    def _check(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name:
                    continue
                top = alias.name.split(".")[0]
                if not _is_framework(top):
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                return None
            if node.module is None:
                return None
            top = node.module.split(".")[0]
            if not _is_framework(top):
                return node.module
        return None

    # Recursive in-order traversal that preserves source order — ast.walk
    # uses BFS which would shuffle nested imports relative to siblings.
    def _visit(node: ast.AST) -> Optional[str]:
        hit = _check(node)
        if hit is not None:
            return hit
        for child in ast.iter_child_nodes(node):
            hit = _visit(child)
            if hit is not None:
                return hit
        return None

    return _visit(tree)


def _module_to_path(module: str) -> str:
    """`app.h245_fsm_sim` -> `app/h245_fsm_sim.py`; `solution` -> `solution.py`."""
    return module.replace(".", "/") + ".py"


def patch_instruction(text: str, module: str) -> tuple[str, bool]:
    """Return (patched_text, changed). Idempotent."""
    if PATCH_MARKER in text:
        return text, False

    module_path = _module_to_path(module)
    suffix = (
        f"\n\n"
        f"Important: place your implementation at `/app/{module_path}` "
        f"so the tests can `import {module}`.\n"
    )

    # Preserve original trailing whitespace style — strip a single trailing
    # newline before appending so we don't end up with three blank lines.
    base = text.rstrip("\n")
    return base + suffix, True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Tasks dir (extracted parquet)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Patch at most N tasks (0 = all)")
    p.add_argument(
        "--skiplist-out",
        default=None,
        help="Optional path to write skipped task names + reason (one per line).",
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_changed = 0
    n_already = 0
    n_skipped = 0
    skip_reasons: list[str] = []

    for i, td in enumerate(task_dirs, 1):
        instr_path = td / "instruction.md"
        test_path = td / "tests" / "test_scaffold.py"

        if not instr_path.is_file():
            n_skipped += 1
            skip_reasons.append(f"{td.name}\tno_instruction_md")
            continue
        if not test_path.is_file():
            n_skipped += 1
            skip_reasons.append(f"{td.name}\tno_test_scaffold_py")
            continue

        try:
            test_text = test_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            n_skipped += 1
            skip_reasons.append(f"{td.name}\tread_error: {exc}")
            continue

        module = find_target_module(test_text)
        if module is None:
            n_skipped += 1
            skip_reasons.append(f"{td.name}\tno_top_level_user_import")
            continue

        instr_text = instr_path.read_text()
        patched, changed = patch_instruction(instr_text, module)
        if not changed:
            n_already += 1
            continue

        n_changed += 1
        if not args.dry_run:
            instr_path.write_text(patched)

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} already={n_already} "
                f"skipped={n_skipped}",
                flush=True,
            )

    print(
        f"Done. changed={n_changed}/{n_total}, already_patched={n_already}, "
        f"skipped={n_skipped}, dry_run={args.dry_run}"
    )

    if args.skiplist_out and skip_reasons:
        Path(args.skiplist_out).write_text("\n".join(skip_reasons) + "\n")
        print(f"Wrote skiplist to {args.skiplist_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
