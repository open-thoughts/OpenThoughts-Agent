#!/usr/bin/env python3
"""
exp_rpt_stack-pytest-synthetic-gpt5nano patcher (v2 -> v3).

QC of v2 (`laion/exp_rpt_stack-pytest-synthetic-gpt5nano-v2`) found:
  - 92/200 (46%) trials infra-failure (90 AgentSetupTimeoutError +
    2 RuntimeError) — all Daytona tmux-install hang, NOT patcher-fixable.
  - 108/200 reached the verifier; 0 of those passed.
  - The 0% pass rate among the 108 ran trials traces to one of:
      a) Test imports a pip-only package not in the daytona base image
         (e.g. ``cookiecutter``, ``grpc``, ``dataclasses_avroschema``,
         ``pyoxenmq``, ``dit``) that the agent's solution shouldn't ship.
         When the test does ``from cookiecutter.exceptions import ...``
         pytest aborts at collection time with ModuleNotFoundError.
      b) Test uses pytest fixtures the agent's solution can't provide
         (the test-level ``app`` fixture / conftest is absent).

This is the same disease as the rle/flat25 corpus before
`patch_rle_flat25_tasks.py` landed. We adopt the same filtering logic:
drop tasks where the test imports a non-stdlib, non-builtin, non-pytest,
non-task-mentioned package.

We also, unlike rle_flat25, run an extra check: drop tasks whose
``test_solution.py`` contains pytest-fixture-using test functions
(``def test_xxx(some_arg):`` where ``some_arg`` is not a stdlib pytest
fixture or in the file's own ``conftest.py``). The agent has no way to
satisfy these.

Filter passes (in order):

  1. ``py_compile`` ``tests/test_solution.py`` → drop on syntax error.
  2. AST-parse top-level imports. Drop the task if ANY top-level
     import is not in:
       - Python stdlib (``sys.stdlib_module_names``)
       - The container's pip whitelist (a reasonable superset; we err
         toward keeping)
       - A token mentioned literally in ``instruction.md``
  3. AST-parse top-level test functions. If any test function has at
     least one positional parameter that's not a built-in pytest
     fixture (and there's no ``conftest.py`` in the same dir
     defining that fixture), drop the task.

CLI mirrors `patch_stack_pytest_synthetic_tasks.py` (the v2 patcher)
plus the extra fixture-check pass.

Idempotent: dropping is destructive (rmtree), re-running is safe
(already-dropped tasks aren't present). The patcher does NOT mutate
surviving tasks.

Usage::

    python data/patchers/patch_stack_pytest_synthetic_v3_tasks.py \\
        --root /path/to/extracted_tasks_dir \\
        [--dry-run] [--limit N] [--drop-log path.tsv]

Apply ON TOP of (or in place of) the v2 PEP-668 unblock — the v2
patcher only fixes pip's --break-system-packages, which is orthogonal
to what we filter here. Either order is fine. Target HF repo:
``laion/exp_rpt_stack-pytest-synthetic-gpt5nano-v3``.
"""
from __future__ import annotations

import argparse
import ast
import py_compile
import re
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist (same shape as patch_rle_flat25_tasks.py — keep them in sync if
# you ever decide to share the impl).
# ---------------------------------------------------------------------------

STDLIB: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", set()))

ALWAYS_INSTALLED: frozenset[str] = frozenset({
    "pytest",
    "_pytest",
    "py",
    "pluggy",
    "iniconfig",
    "packaging",
    "tomli",
    "exceptiongroup",
    # Conventional agent-provided entrypoint module: tests in this corpus
    # almost universally do ``from solution import X`` to import what the
    # agent produced at /app/solution.py. Treat these as always-allowed
    # so we don't drop tasks that follow the convention.
    "solution",
    "app",
    "src",
})

# Top-level imports we know are present after pip install in the v2
# verifier's pre-test step. (The v2 test.sh of pytest-synthetic only
# installs pytest itself; a solution-implementing agent might add more.
# We keep this allowlist conservative.)
WHITELIST_PIP_TO_IMPORTS: dict[str, tuple[str, ...]] = {
    "requests": ("requests",),
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "scipy": ("scipy",),
    "scikit-learn": ("sklearn",),
    "sklearn": ("sklearn",),
    "httpx": ("httpx",),
    "aiohttp": ("aiohttp",),
    "django": ("django",),
    "flask": ("flask",),
    "fastapi": ("fastapi",),
    "pydantic": ("pydantic",),
    "pyyaml": ("yaml",),
    "pytz": ("pytz",),
    "cryptography": ("cryptography",),
    "pillow": ("PIL",),
    "hypothesis": ("hypothesis",),
    "click": ("click",),
    "jinja2": ("jinja2",),
    "lxml": ("lxml",),
    "beautifulsoup4": ("bs4",),
}

WHITELIST_IMPORTS: frozenset[str] = frozenset(
    name for imports in WHITELIST_PIP_TO_IMPORTS.values() for name in imports
)

# Built-in pytest fixtures (pytest 7+/8+/9+). Any test param matching
# one of these is fine; anything else needs a conftest.py.
PYTEST_BUILTIN_FIXTURES: frozenset[str] = frozenset({
    "cache", "capfd", "capfdbinary", "caplog", "capsys", "capsysbinary",
    "capteesys", "doctest_namespace", "monkeypatch", "pytestconfig",
    "record_property", "record_testsuite_property", "record_xml_attribute",
    "recwarn", "subtests", "tmp_path", "tmp_path_factory", "tmpdir",
    "tmpdir_factory", "request", "testdir", "pytester",
})


def _normalize_token(s: str) -> str:
    return s.strip().lower()


def _tokens_in_text(text: str) -> set[str]:
    """Words in ``text`` (lowercased), used to fuzzy-match imports
    against the task description."""
    return {_normalize_token(t) for t in re.findall(r"[A-Za-z0-9_\-\.]+", text)}


def _import_is_allowed(name: str, instruction_tokens: set[str]) -> bool:
    """Return True if a top-level import is OK to keep."""
    top = name.split(".")[0]
    if top in STDLIB:
        return True
    if top in ALWAYS_INSTALLED:
        return True
    if top in WHITELIST_IMPORTS:
        return True
    # If the instruction explicitly names this module, the agent might
    # plausibly be expected to provide it (or to install it).
    if _normalize_token(top) in instruction_tokens:
        return True
    if _normalize_token(name) in instruction_tokens:
        return True
    return False


def _conftest_fixtures(test_dir: Path) -> set[str]:
    """Parse ``conftest.py`` next to test_solution.py for ``@pytest.fixture``
    function names. Returns empty set if no conftest."""
    cf = test_dir / "conftest.py"
    if not cf.is_file():
        return set()
    try:
        tree = ast.parse(cf.read_text(), filename=str(cf))
    except Exception:
        return set()
    fixtures: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                # @pytest.fixture or @fixture
                if (isinstance(dec, ast.Attribute) and dec.attr == "fixture") or \
                   (isinstance(dec, ast.Name) and dec.id == "fixture") or \
                   (isinstance(dec, ast.Call) and (
                       (isinstance(dec.func, ast.Attribute) and dec.func.attr == "fixture") or
                       (isinstance(dec.func, ast.Name) and dec.func.id == "fixture"))):
                    fixtures.add(node.name)
                    break
    return fixtures


def _test_function_param_names(tree: ast.AST) -> dict[str, list[str]]:
    """Return ``{test_name: [param_names]}`` for top-level test_* functions."""
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name.startswith("test_"):
            params: list[str] = []
            args = node.args
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                if a.arg in ("self", "cls"):
                    continue
                params.append(a.arg)
            out[node.name] = params
    return out


# ---------------------------------------------------------------------------
# Filter passes
# ---------------------------------------------------------------------------

def syntax_passes(test_solution: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(test_solution), doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, f"syntax:{e.msg.splitlines()[0] if e.msg else 'error'}"
    except Exception as e:
        return False, f"compile:{e}"


def imports_pass(
    test_solution: Path,
    instruction_tokens: set[str],
) -> tuple[bool, str]:
    try:
        tree = ast.parse(test_solution.read_text(), filename=str(test_solution))
    except Exception as e:
        return False, f"parse:{e}"
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_is_allowed(alias.name, instruction_tokens):
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                # Relative imports — fine, they're internal.
                continue
            if not _import_is_allowed(mod, instruction_tokens):
                bad.append(mod)
    if bad:
        return False, f"import:{bad[0]}"
    return True, ""


def fixtures_pass(test_solution: Path) -> tuple[bool, str]:
    """Drop tasks where any test_* function has a non-builtin, non-conftest
    fixture as a positional arg."""
    try:
        tree = ast.parse(test_solution.read_text(), filename=str(test_solution))
    except Exception as e:
        return False, f"parse:{e}"
    conftest = _conftest_fixtures(test_solution.parent)
    funcs = _test_function_param_names(tree)
    for name, params in funcs.items():
        for p in params:
            if p in PYTEST_BUILTIN_FIXTURES:
                continue
            if p in conftest:
                continue
            return False, f"fixture:{name}({p})"
    return True, ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def filter_one_task(task_dir: Path) -> tuple[str, str]:
    """Run all three passes. Return ``(verdict, reason)``.

    ``verdict ∈ {"keep", "drop"}``. ``reason`` is a short tag.
    """
    test_solution = task_dir / "tests" / "test_solution.py"
    instruction = task_dir / "instruction.md"

    if not test_solution.is_file():
        return "drop", "no-test_solution"
    instr_text = instruction.read_text() if instruction.is_file() else ""
    instr_tokens = _tokens_in_text(instr_text)

    ok, why = syntax_passes(test_solution)
    if not ok:
        return "drop", why

    ok, why = imports_pass(test_solution, instr_tokens)
    if not ok:
        return "drop", why

    ok, why = fixtures_pass(test_solution)
    if not ok:
        return "drop", why

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

    counts = {"keep": 0, "drop_syntax": 0, "drop_import": 0,
              "drop_fixture": 0, "drop_other": 0}
    drop_log_lines: list[str] = []
    examples: dict[str, list[str]] = {}

    total = len(task_dirs)
    for i, td in enumerate(task_dirs, 1):
        verdict, reason = filter_one_task(td)
        if verdict == "keep":
            counts["keep"] += 1
        else:
            if reason.startswith("syntax"):
                key = "drop_syntax"
            elif reason.startswith("import"):
                key = "drop_import"
            elif reason.startswith("fixture"):
                key = "drop_fixture"
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
    for k in ("keep", "drop_syntax", "drop_import", "drop_fixture", "drop_other"):
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
