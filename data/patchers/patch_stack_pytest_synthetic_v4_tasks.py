#!/usr/bin/env python3
"""
exp_rpt_stack-pytest-synthetic-gpt5nano patcher (v3 -> v4).

QC of v3 (`laion/exp_rpt_stack-pytest-synthetic-gpt5nano-v3`, n=200) found
a 0.5% solve rate. The v3 patcher's `imports_pass` KEPT tasks whose test
imports were in a small "container will have it" allowlist
(``WHITELIST_PIP_TO_IMPORTS``: requests/numpy/pandas/scipy/sklearn/...),
but the verifier's ``tests/test.sh`` only does
``pip3 install --quiet pytest`` — none of the allowlisted packages are
actually present in the daytona base image. As a result, the kept tasks
abort at pytest collection with ``ModuleNotFoundError`` for numpy /
pandas / flask / sympy / sqlalchemy / etc.

Failing-trace top-misses on the v3 dataset (n=200): numpy (34), pandas
(5), tvm (4), sympy (4), mock (4), flask (3), sqlalchemy (2), networkx
(2), neomodel (2), glyphsLib (2), ecdsa (2). All of these except tvm /
glyphsLib / neomodel are pip-installable.

Two ways to close the "import allowlist vs. actually-installed" gap:

  A. Make ``test.sh`` pip-install the imports the test ACTUALLY needs.
     Compute them by AST-parsing the test file's top-level imports and
     mapping them through the inverted ``WHITELIST_PIP_TO_IMPORTS``.
     Inject a ``pip install --quiet ... 2>/dev/null || true`` line into
     ``test.sh`` immediately after the existing pytest install. Marker-
     guarded for idempotency.

  B. Pre-bake a fat base image with the union of common deps.

We implement Option A: it's cheap, repo-local, and re-runnable.

Filter passes (in order):

  1. ``py_compile`` ``tests/test_solution.py`` → drop on syntax error.
  2. AST-parse top-level imports. Drop the task if ANY top-level import
     is not in:
       - Python stdlib (``sys.stdlib_module_names``)
       - The container's pip whitelist (expanded relative to v3 — see
         ``WHITELIST_PIP_TO_IMPORTS`` below)
       - A token mentioned literally in ``instruction.md``
  3. AST-parse top-level test functions. If any test function has at
     least one positional parameter that's not a built-in pytest
     fixture (and there's no ``conftest.py`` in the same dir defining
     that fixture), drop the task.
  4. **NEW vs. v3:** Drop tasks whose ``test_solution.py`` has a
     top-level relative import (``from .X import ...`` or
     ``from .. import X``) at file scope. The verifier runs the test
     file via ``pytest /tests/test_solution.py`` — it has no parent
     package, so relative imports raise
     ``ImportError: attempted relative import with no known parent
     package`` at collection. Unfixable without restructuring.

Mutation pass (the actual v4 fix), applied to surviving tasks:

  5. AST-parse the test file's top-level imports, compute the set of
     PyPI package names needed (via inverted import-to-package map),
     and inject

         # --- laion v4 patch: install detected test imports ---
         pip3 install --break-system-packages --quiet <pkgs> 2>/dev/null \\
             || pip3 install --quiet <pkgs> 2>/dev/null || true

     into ``tests/test.sh`` immediately after the existing
     ``pip3 install ... pytest`` line. Marker-guarded; re-runs are
     idempotent.

CLI mirrors `patch_stack_pytest_synthetic_v3_tasks.py`:

    python data/patchers/patch_stack_pytest_synthetic_v4_tasks.py \\
        --root /path/to/extracted_tasks_dir \\
        [--dry-run] [--limit N] [--drop-log path.tsv]

Apply ON TOP of v3 or v2 — either works; the only mutation is the
test.sh injection, which is fenced by the v4 marker. Target HF repo:
``laion/exp_rpt_stack-pytest-synthetic-gpt5nano-v4``.
"""
from __future__ import annotations

import argparse
import ast
import py_compile
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist (expanded from v3 — see module docstring).
#
# Adds: mock, sympy, networkx, werkzeug, traitlets, xarray, ecdsa,
# sqlalchemy. These all appeared in v3 failure traces and are real PyPI
# packages installable inside the daytona base image. We DO NOT add the
# niche packages that surfaced in v3 failures (``tvm``, ``glyphsLib``,
# ``neomodel``, etc.) — those have low solve probability and are kept on
# the implicit drop list (i.e., not in the allowlist; the imports_pass
# filter will drop tasks that import them, unless they're named in
# instruction.md).
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

# Mapping: PyPI distribution name -> the import names it exposes.
# Used in TWO directions:
#   (forward) build WHITELIST_IMPORTS for the imports_pass filter
#   (inverted) when injecting pip-installs into test.sh, look up which
#              PyPI package supplies a given import name
WHITELIST_PIP_TO_IMPORTS: dict[str, tuple[str, ...]] = {
    # carry-over from v3
    "requests": ("requests",),
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "scipy": ("scipy",),
    "scikit-learn": ("sklearn",),
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
    # NEW in v4 — sourced from v3 failure-trace top-misses
    "mock": ("mock",),
    "sympy": ("sympy",),
    "networkx": ("networkx",),
    "werkzeug": ("werkzeug",),
    "traitlets": ("traitlets",),
    "xarray": ("xarray",),
    "ecdsa": ("ecdsa",),
    "sqlalchemy": ("sqlalchemy",),
}

WHITELIST_IMPORTS: frozenset[str] = frozenset(
    name for imports in WHITELIST_PIP_TO_IMPORTS.values() for name in imports
)

# Inverted: import-name -> PyPI package name (lowercased import key).
# ``sklearn`` -> ``scikit-learn``, ``bs4`` -> ``beautifulsoup4``,
# ``PIL`` -> ``pillow``, ``yaml`` -> ``pyyaml``, etc.
IMPORT_TO_PIP_PKG: dict[str, str] = {
    imp: pip_pkg
    for pip_pkg, imports in WHITELIST_PIP_TO_IMPORTS.items()
    for imp in imports
}

# Built-in pytest fixtures (pytest 7+/8+/9+). Any test param matching
# one of these is fine; anything else needs a conftest.py.
PYTEST_BUILTIN_FIXTURES: frozenset[str] = frozenset({
    "cache", "capfd", "capfdbinary", "caplog", "capsys", "capsysbinary",
    "capteesys", "doctest_namespace", "monkeypatch", "pytestconfig",
    "record_property", "record_testsuite_property", "record_xml_attribute",
    "recwarn", "subtests", "tmp_path", "tmp_path_factory", "tmpdir",
    "tmpdir_factory", "request", "testdir", "pytester",
})

# Idempotency marker for the injected test.sh block.
V4_MARKER = "# --- laion v4 patch: install detected test imports ---"


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


def _top_level_imports(tree: ast.AST) -> tuple[list[str], bool]:
    """Walk the AST and return ``(top_level_module_names, has_relative)``.

    ``top_level_module_names`` are dotted module names (e.g. ``django.http``,
    ``smlb.learners``); the caller is responsible for taking ``.split(".")[0]``
    when checking against allowlists.

    ``has_relative`` is True if ANY top-level (file-scope) ImportFrom has
    ``level > 0`` (i.e. ``from .X import ...`` or ``from .. import Y``).
    Unlike ``imports_pass`` in v3 — which silently allowed relative imports
    by ``continue``-ing — v4 reports them so the driver can drop the task.
    """
    names: list[str] = []
    has_relative = False
    for node in tree.body:  # only file-scope imports — matches v3 semantics
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                has_relative = True
                continue
            names.append(node.module or "")
    return names, has_relative


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
                # Relative imports — handled by the dedicated relative-
                # import filter; not the imports-allowlist's concern.
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


def relative_imports_pass(test_solution: Path) -> tuple[bool, str]:
    """Drop tasks with ``from .X import ...`` at file scope. Pytest invokes
    ``test_solution.py`` as a stand-alone file with no parent package, so
    relative imports raise at collection."""
    try:
        tree = ast.parse(test_solution.read_text(), filename=str(test_solution))
    except Exception as e:
        return False, f"parse:{e}"
    _, has_relative = _top_level_imports(tree)
    if has_relative:
        return False, "relative_import"
    return True, ""


# ---------------------------------------------------------------------------
# Mutation: inject pip-installs into test.sh
# ---------------------------------------------------------------------------

def _detect_pip_packages(test_solution: Path) -> list[str]:
    """Return a deterministic, deduplicated list of PyPI package names that
    map (via ``IMPORT_TO_PIP_PKG``) to the test file's top-level imports.

    Imports that are stdlib, in ``ALWAYS_INSTALLED``, or unknown to the
    inverted map are skipped — the v4 fix only installs what we KNOW is on
    PyPI under a specific name; everything else is left to the existing
    test-time error path."""
    try:
        tree = ast.parse(test_solution.read_text(), filename=str(test_solution))
    except Exception:
        return []
    names, _ = _top_level_imports(tree)
    pkgs: list[str] = []
    seen: set[str] = set()
    for full in names:
        top = full.split(".")[0]
        if not top:
            continue
        if top in STDLIB or top in ALWAYS_INSTALLED:
            continue
        pip_pkg = IMPORT_TO_PIP_PKG.get(top)
        if pip_pkg is None:
            continue
        if pip_pkg in seen:
            continue
        seen.add(pip_pkg)
        pkgs.append(pip_pkg)
    return pkgs


def _injection_block(pkgs: list[str]) -> str:
    """Build the marker-fenced lines we inject into test.sh."""
    pkg_str = " ".join(pkgs)
    # PEP-668 unblock + soft fallback + ``|| true`` so a transient PyPI hiccup
    # doesn't kill the whole verifier — pytest collection will report the
    # ModuleNotFoundError naturally if anything's still missing.
    return (
        f"{V4_MARKER}\n"
        f"pip3 install --break-system-packages --quiet {pkg_str} 2>/dev/null"
        f" || pip3 install --quiet {pkg_str} 2>/dev/null || true\n"
    )


def patch_test_sh(test_sh: Path, pkgs: list[str]) -> tuple[bool, str]:
    """Inject the pip-install line into ``test.sh`` after the existing
    ``pytest`` install. Returns ``(changed, reason)``.

    Idempotent: if the V4 marker is already present, leaves the file alone
    (returns ``False, "already-patched"``). If ``pkgs`` is empty, no-op
    (returns ``False, "no-pkgs"``).
    """
    if not test_sh.is_file():
        return False, "no-test.sh"
    text = test_sh.read_text()
    if V4_MARKER in text:
        return False, "already-patched"
    if not pkgs:
        return False, "no-pkgs"

    lines = text.splitlines(keepends=True)
    # Find the first line that pip-installs pytest (the v2 patch line).
    # Match either the PEP-668-unblocked line (``--break-system-packages``)
    # or a plain ``pip3 install ... pytest``.
    insert_at: int | None = None
    pip_pytest_re = re.compile(r"pip3?\s+install[^\n]*\bpytest\b")
    for i, ln in enumerate(lines):
        if pip_pytest_re.search(ln):
            insert_at = i + 1
            break
    if insert_at is None:
        return False, "no-pytest-install-line"

    block = _injection_block(pkgs)
    # Indent the block to match the surrounding line's indent (the v2 patch
    # line lives inside an ``if ! command -v pytest``-style block in some
    # tasks; preserve that).
    indent = re.match(r"[ \t]*", lines[insert_at - 1]).group(0)
    indented_block = "".join(indent + ln for ln in block.splitlines(keepends=True))

    new_lines = lines[:insert_at] + [indented_block] + lines[insert_at:]
    test_sh.write_text("".join(new_lines))
    return True, f"patched:{','.join(pkgs)}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def filter_one_task(task_dir: Path) -> tuple[str, str]:
    """Run all four filter passes. Return ``(verdict, reason)``.

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

    ok, why = relative_imports_pass(test_solution)
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
            # Mutation: patch test.sh in-place.
            test_sh = td / "tests" / "test.sh"
            test_solution = td / "tests" / "test_solution.py"
            pkgs = _detect_pip_packages(test_solution)
            if not args.dry_run:
                changed, _why = patch_test_sh(test_sh, pkgs)
                if changed:
                    counts["patched"] += 1
                else:
                    counts["patch_skipped"] += 1
            else:
                # Still surface what WOULD be done.
                if pkgs and test_sh.is_file() and V4_MARKER not in test_sh.read_text():
                    counts["patched"] += 1
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
