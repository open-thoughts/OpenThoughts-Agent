#!/usr/bin/env python3
"""
Filter ``DCAgent/exp_rle_github_issue`` tasks (5k-github-issue-* pool) to drop
tasks whose ``tests/test_solution.py`` cannot run usefully in the daytona
sandbox image.

Background
----------
v2 (initial patch, 5000 → 739): caught the obvious cases — non-whitelisted
top-level imports, syntax errors, and unconditional pytest.mark.skip[if] on
every test. QC of 200 v2 trials showed:

  * 200/200 verifier infra-OK, 14/200 (7.0%) ``reward == 1``.
  * 186/200 failures break down (re-triage 2026-05-15):
      - 67 (36%) **fixture-not-found** at runtime: tests reference fixtures
        (``testdir``, ``client``, ``app``, ``host``, ``idx``, ``index``,
        ``tz_naive_fixture``, etc.) but no conftest.py is shipped with the
        task. v2 patcher missed these because it only checked imports.
      - ~30 (15%) **broken sub-imports from whitelisted packages** —
        ``from pandas.compat import lrange``,
        ``from pandas.core.base import SpecificationError``,
        ``from pandas.util import testing``,
        ``from pandas import Float64Index``,
        ``pydantic.datetime_parse`` etc.
        v2 passed these because top-level ``pandas`` / ``pydantic`` are in
        the whitelist, but the SUB-attribute was removed in modern pandas/
        pydantic.
      - 10 ``tzdata`` ModuleNotFound (not in whitelist).
      - 7 ``from tests.X import Y`` (the test imports an external ``tests``
        package that ships with upstream test suite but not the task).
      - 4 module-level AttributeError on whitelisted pkgs
        (``pandas.util._test_decorators.skip_if_np_lt``, etc.).
      - 12 zero-test-collected (empty file / module-level
        ``pytest.importorskip``).
      - 2 single-test rubber-stamp skips via ``pytest.importorskip()`` at
        module-level (not catchable by AST scan of ``pytest.mark.skip``).
      - 14 genuine FAILED tests (agent could in theory solve; we keep these).

v3 (this patcher) adds:

  5. **Sub-attribute / sub-module checks**: drop if test imports a known-
     removed attribute from ``pandas``, ``pandas.compat``, ``pandas.util``,
     ``pandas.core.*``, ``pydantic`` (v1 APIs), ``sklearn.utils``,
     ``scipy.spatial.distance``, ``httpx``.
  6. **Missing-fixture scan**: walk all ``def test_*(self?, *args)``
     signatures, collect every parameter name that's not ``self`` or a
     standard pytest fixture (``tmp_path``, ``capsys``, ``monkeypatch``,
     ``request``, ``caplog``, …) or a parameter introduced via
     ``@pytest.mark.parametrize(...)``. Drop the task if any such name is
     referenced but no local ``@pytest.fixture`` defines it AND no
     ``conftest.py`` ships with the task.
  7. **Module-level ``pytest.importorskip(...)``**: if a top-level
     statement (or top-level Assign whose value is) calls
     ``pytest.importorskip(...)``, the package may be missing at runtime and
     pytest would emit ``1 skipped`` (rubber-stamp reward=1 is also possible
     when there's only one such call and no tests). Drop these.
  8. **Module-level ``os.chdir(literal_path)`` / ``os.environ[KEY]`` /
     ``ctypes.CDLL(literal_path)``** that reference paths/env-vars not in
     the task's instruction → drop.
  9. **``from tests.X import Y``** at top-level → drop (``tests`` is not a
     pip package and the task doesn't ship a sibling ``tests/`` package
     beyond the single test_solution.py).

Container image (``environment/Dockerfile``) is ``python3 + pytest``. The
runtime hook (``tests/test.sh``) pip-installs the fixed whitelist:

    requests numpy pandas scipy scikit-learn sklearn torch tensorflow keras
    httpx aiohttp ddtrace django flask fastapi matplotlib seaborn pillow
    pydantic pytest-mock requests-mock faker pyyaml pytz cryptography bcrypt
    hypothesis

(This is the same whitelist as the rle_flat25 family — see
``patch_rle_flat25_tasks.py``.)

Filter logic
------------
For each task:

  1. ``py_compile`` ``tests/test_solution.py`` — drop on syntax error.
  2. AST-parse top-level imports. Drop if ANY top-level import is not in:
       * Python stdlib (``sys.stdlib_module_names``)
       * Always-present pytest helpers
       * The container's pip whitelist (above)
       * A token mentioned literally in ``instruction.md``
  3. Drop if every ``def test_*`` / ``Test*`` method is decorated with
     ``@pytest.mark.skip`` or guarded by a module-level ``pytestmark =
     pytest.mark.skipif(...)``.
  4. v3 extra checks (5)-(9) above.
  5. Otherwise keep the task untouched. The patcher never mutates tasks; it
     only deletes (drops) entire task directories.

The script writes a ``.patcher_marker_exp_rle_github_issue_v3`` file at the
root of the tasks directory after a successful run; subsequent runs against
the same dir are a no-op (the v2 marker is ignored — v3 re-checks tasks
even if they passed the v2 marker).

Usage
-----
    python data/patchers/patch_exp_rle_github_issue_tasks.py --root /path/to/tasks_dir
    python data/patchers/patch_exp_rle_github_issue_tasks.py --root /path/to/tasks_dir --dry-run
    python data/patchers/patch_exp_rle_github_issue_tasks.py --root /path/to/tasks_dir --limit 200
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
# Allowlist (matches environment/test.sh)
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
})

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

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

MARKER_NAME = ".patcher_marker_exp_rle_github_issue_v3"

# ---------------------------------------------------------------------------
# v3: known-removed sub-attributes / sub-modules of whitelisted packages.
#
# Each entry is "dotted.attr.path" — if the test file imports this path (via
# ``from a.b import c`` where the dotted form ``a.b.c`` matches, or via
# attribute access ``a.b.c`` at module level), the task is dropped. Built from
# the 7%-solve trace re-triage (2026-05-15).
# ---------------------------------------------------------------------------
KNOWN_BROKEN_SUB_ATTRS: frozenset[str] = frozenset({
    # pandas — removed in pandas >= 2.x
    "pandas.compat.lrange",
    "pandas.compat.PY37",
    "pandas.compat.np_datetime64_compat",
    "pandas.compat.np_version_under1p19",
    "pandas.util.testing",
    "pandas._np_version_under1p14",
    "pandas.Float64Index",
    "pandas.Int64Index",
    "pandas.UInt64Index",
    "pandas.core.base.SpecificationError",
    "pandas.core.groupby.groupby.SpecificationError",
    "pandas.core.arrays.PandasArray",
    "pandas.core.indexing.maybe_numeric_slice",
    "pandas.core.algorithms.quantile",
    "pandas.tests.io.pytables.common._maybe_remove",
    "pandas.tests.arrays.categorical.common",
    "pandas.tests.extension.base.BaseNoReduceTests",
    "pandas.util._test_decorators.skip_if_np_lt",
    "pandas.util._test_decorators.skip_if_no_ne",
    "pandas.util._test_decorators.skip_if_no_scipy",
    # pydantic v1 APIs removed in v2
    "pydantic.datetime_parse",
    "pydantic.NoneBytes",
    # sklearn — internal/removed APIs
    "sklearn.utils._IS_32BIT",
    "sklearn.utils._testing.assert_warns_message",
    "sklearn.ensemble._weight_boosting._samme_proba",
    # scipy
    "scipy.spatial.distance.kulsinski",
    # httpx — removed/renamed
    "httpx.HTTPProxy",
})

# Top-level packages that are STDLIB-shaped but NOT in sys.stdlib_module_names
# on every Python version. Also the standard pytest-provided fixtures whose
# names appear as fn arguments without a local @pytest.fixture definition.
STANDARD_PYTEST_FIXTURES: frozenset[str] = frozenset({
    "request", "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory",
    "capsys", "capsysbinary", "capfd", "capfdbinary", "caplog",
    "monkeypatch", "recwarn", "doctest_namespace", "pytestconfig",
    "record_property", "record_xml_attribute", "record_testsuite_property",
    "cache", "mocker",  # mocker is from pytest-mock (in whitelist)
    "requests_mock",    # from requests-mock (in whitelist)
    "faker",            # from faker (in whitelist)
    "anyio_backend",    # from anyio
    "self", "cls",
})

# Names that are NOT fixtures even though they appear as fn args (these are
# parametrize values, lambdas via @pytest.fixture(params=...) elsewhere, etc.).
# Keep this very small — be conservative; missing-fixture rule should err on
# the side of dropping.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_level_imports(tree: ast.AST) -> set[str]:
    """Return top-level package names imported by an AST module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative — local package
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def _instruction_tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def _is_skip_decorator(decorator: ast.AST) -> bool:
    """True if the decorator looks like @pytest.mark.skip or @pytest.mark.skipif."""
    # @pytest.mark.skip / @pytest.mark.skipif / @pytest.mark.skip(...)
    target = decorator
    if isinstance(target, ast.Call):
        target = target.func
    # Walk dotted attribute name into a string like "pytest.mark.skip"
    parts: list[str] = []
    cur = target
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    dotted = ".".join(reversed(parts))
    return dotted.endswith("pytest.mark.skip") or dotted.endswith("pytest.mark.skipif")


def _module_level_skipif(tree: ast.Module) -> bool:
    """True if module sets ``pytestmark = pytest.mark.skipif(...)`` at module level.

    Conservative: only triggers on a literal assignment to ``pytestmark`` whose
    value is a Call/Attribute matching pytest.mark.skip[if]. A list of marks is
    also handled by checking each element.
    """
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        names = []
        for t in stmt.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
        if "pytestmark" not in names:
            continue
        value = stmt.value
        candidates: list[ast.AST] = []
        if isinstance(value, (ast.List, ast.Tuple)):
            candidates.extend(value.elts)
        else:
            candidates.append(value)
        for c in candidates:
            if _is_skip_decorator(c):
                return True
    return False


def _all_tests_skipped(tree: ast.Module) -> bool:
    """True if every test in the module is unconditionally skipped.

    A "test" is a top-level ``def test_*`` or a method ``def test_*`` /
    ``def Test*`` in a top-level class whose name starts with ``Test``. If the
    module has no tests at all, this returns False (filtering is left to the
    import / fixture checks — empty test files are caught separately).
    """
    if _module_level_skipif(tree):
        return True

    test_count = 0
    skipped_count = 0

    def _func_is_test(fn: ast.AST, in_test_class: bool) -> bool:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if fn.name.startswith("test_") or fn.name == "test":
            return True
        if in_test_class and fn.name.startswith("test"):
            return True
        return False

    def _is_skipped(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return any(_is_skip_decorator(d) for d in fn.decorator_list)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _func_is_test(node, False):
            test_count += 1
            if _is_skipped(node):
                skipped_count += 1
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_skipped = any(_is_skip_decorator(d) for d in node.decorator_list)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and _func_is_test(sub, True):
                    test_count += 1
                    if class_skipped or _is_skipped(sub):
                        skipped_count += 1

    if test_count == 0:
        return False
    return skipped_count == test_count


# ---------------------------------------------------------------------------
# v3 helpers
# ---------------------------------------------------------------------------

def _from_import_dotted_paths(tree: ast.Module) -> set[str]:
    """Return the set of fully-dotted import targets in ``from X import Y``.

    For ``from pandas.compat import lrange``, this yields
    ``"pandas.compat.lrange"``. For ``from pandas import (Float64Index, X)``,
    yields ``"pandas.Float64Index"`` and ``"pandas.X"``. For
    ``from .X import Y``, returns nothing (relative imports are not flagged).
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level and node.level > 0:
            continue
        if not node.module:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            out.add(f"{node.module}.{alias.name}")
    return out


def _attribute_dotted_chains(tree: ast.AST) -> set[str]:
    """Return every fully-qualified dotted name accessed via attribute chains
    rooted at a Name. E.g. ``pandas.util._test_decorators.skip_if_no_ne`` →
    ``"pandas.util._test_decorators.skip_if_no_ne"`` (and all proper prefixes
    of length >= 2). Useful for catching module-level
    ``@td.skip_if_no_ne`` etc. — though that's via a Name alias, not a chain;
    the chain version catches direct ``pandas.x.y.z`` references.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # walk leftwards
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            continue
        parts.append(cur.id)
        parts.reverse()
        for L in range(2, len(parts) + 1):
            out.add(".".join(parts[:L]))
    return out


def _known_broken_subattrs(tree: ast.Module) -> list[str]:
    """Intersection of import / attribute dotted paths with KNOWN_BROKEN_SUB_ATTRS."""
    hits: set[str] = set()
    fdotted = _from_import_dotted_paths(tree)
    hits.update(fdotted & KNOWN_BROKEN_SUB_ATTRS)
    # Also: an Import (``import pandas.compat.lrange``) — rare.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in KNOWN_BROKEN_SUB_ATTRS:
                    hits.add(alias.name)
    # Attribute chains (``pandas.compat.lrange()`` at top level — rare but cheap)
    attrs = _attribute_dotted_chains(tree)
    hits.update(attrs & KNOWN_BROKEN_SUB_ATTRS)
    return sorted(hits)


def _has_module_level_importorskip(tree: ast.Module) -> bool:
    """Detect ``foo = pytest.importorskip(...)`` (or bare ``pytest.importorskip(...)``)
    at module scope. These tasks rubber-stamp or zero-collect.
    """
    for stmt in tree.body:
        # bare expression
        if isinstance(stmt, ast.Expr):
            val = stmt.value
        elif isinstance(stmt, ast.Assign):
            val = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            val = stmt.value
        else:
            continue
        if not isinstance(val, ast.Call):
            continue
        # match pytest.importorskip(...)
        f = val.func
        if isinstance(f, ast.Attribute) and f.attr == "importorskip":
            if isinstance(f.value, ast.Name) and f.value.id == "pytest":
                return True
        # match imported alias ``from pytest import importorskip``
        if isinstance(f, ast.Name) and f.id == "importorskip":
            return True
    return False


def _module_level_chdir_or_env(tree: ast.Module, instr_tokens: set[str]) -> str | None:
    """Detect module-level ``os.chdir(LIT)``, ``os.environ[LIT]``,
    ``ctypes.CDLL(LIT)`` that point at a path/env-var not in the instruction.

    Returns a short reason string or None.
    """
    for stmt in tree.body:
        # Walk both Expr (call) and Assign (assignment to call)
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # os.chdir("...") / os.chdir(<name>)
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "chdir"
                and isinstance(f.value, ast.Name)
                and f.value.id == "os"
            ):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    target = node.args[0].value
                    # If the literal is "tests" or some absolute path not in instr,
                    # drop.
                    leaf = target.strip("/").split("/")[-1].lower()
                    if leaf and leaf not in instr_tokens and target not in instr_tokens:
                        return f"module_chdir:{target}"
                else:
                    # chdir(<name>) — fixture-pattern, can't tell; let other rules handle.
                    pass
            # ctypes.CDLL("...")
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "CDLL"
                and isinstance(f.value, ast.Name)
                and f.value.id == "ctypes"
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    return f"module_ctypes_cdll:{node.args[0].value}"

        # os.environ["KEY"] subscript at module-level Assign / Expr
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr)):
            for sub in ast.walk(stmt):
                if not isinstance(sub, ast.Subscript):
                    continue
                v = sub.value
                if (
                    isinstance(v, ast.Attribute)
                    and v.attr == "environ"
                    and isinstance(v.value, ast.Name)
                    and v.value.id == "os"
                ):
                    # extract key
                    slc = sub.slice
                    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                        key = slc.value
                        if key.lower() not in instr_tokens:
                            return f"module_env:{key}"
    return None


def _from_tests_import(tree: ast.Module) -> bool:
    """``from tests.X import Y`` or ``from tests import X`` at top level."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module == "tests" or (node.module and node.module.startswith("tests.")):
                return True
    return False


def _local_fixtures(tree: ast.Module) -> set[str]:
    """Names of functions decorated with @pytest.fixture (or @fixture, or
    ``@pytest.fixture(...)``) anywhere in the file (top-level OR nested in
    classes). Returns the function names.
    """
    fixtures: set[str] = set()

    def is_fixture_dec(d: ast.AST) -> bool:
        target = d
        if isinstance(target, ast.Call):
            target = target.func
        parts: list[str] = []
        cur = target
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        dotted = ".".join(reversed(parts))
        return dotted.endswith("pytest.fixture") or dotted == "fixture" or dotted.endswith("pytest.yield_fixture") or dotted == "yield_fixture"

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(is_fixture_dec(d) for d in node.decorator_list):
                fixtures.add(node.name)
    return fixtures


def _parametrize_arg_names(decorators: list[ast.AST]) -> set[str]:
    """Extract argnames declared via @pytest.mark.parametrize on a fn/class.

    Supports both forms:
      @pytest.mark.parametrize("a,b", [...])
      @pytest.mark.parametrize(("a", "b"), [...])
      @pytest.mark.parametrize(["a", "b"], [...])
    """
    out: set[str] = set()
    for d in decorators:
        target = d
        if not isinstance(target, ast.Call):
            continue
        f = target.func
        parts: list[str] = []
        cur = f
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        dotted = ".".join(reversed(parts))
        if not dotted.endswith("parametrize"):
            continue
        if not target.args:
            continue
        first = target.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            for n in re.split(r"[,\s]+", first.value):
                n = n.strip()
                if n:
                    out.add(n)
        elif isinstance(first, (ast.Tuple, ast.List)):
            for elt in first.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
    return out


def _looks_like_real_test(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Heuristic: a test function returns implicitly (None) or via bare ``return``.
    Helpers that ``return X`` are not real tests even if named ``test_*``.

    Walk the function body's top-level Return statements only. Real tests can
    have early ``return`` (no value), but should not consistently return values.
    Returns False if the function has any Return with a non-None value at the
    function's *top* statement level (not deep inside nested blocks).
    """
    for stmt in fn.body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return False
    return True


def _missing_fixtures(tree: ast.Module, has_conftest: bool, instr_tokens: set[str]) -> list[str]:
    """Detect test function parameters that look like external fixtures with
    no local @pytest.fixture and no conftest.py.

    Skip if the task ships a conftest.py — fixture might be defined there.
    Returns a sorted list of unknown fixture names (empty if all clear).
    """
    if has_conftest:
        return []
    local_fixtures = _local_fixtures(tree)
    unknown: set[str] = set()

    def _collect_test_methods(parent_decorators: list[ast.AST]) -> None:
        pass  # placeholder, walking below

    # Walk classes first to know class-level parametrize markers
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_parametrize = _parametrize_arg_names(node.decorator_list)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    sub.name.startswith("test_") or sub.name.startswith("test")
                ):
                    if not _looks_like_real_test(sub):
                        continue
                    fn_parametrize = _parametrize_arg_names(sub.decorator_list)
                    args = [a.arg for a in sub.args.args]
                    for a in args:
                        if a in {"self", "cls"}: continue  # noqa: E701
                        if a in STANDARD_PYTEST_FIXTURES: continue  # noqa: E701
                        if a in local_fixtures: continue  # noqa: E701
                        if a in class_parametrize or a in fn_parametrize: continue  # noqa: E701
                        if a.lower() in instr_tokens: continue  # noqa: E701
                        unknown.add(a)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node not in tree.body:
                continue
            if not (node.name.startswith("test_") or node.name == "test"):
                continue
            if not _looks_like_real_test(node):
                continue
            fn_parametrize = _parametrize_arg_names(node.decorator_list)
            args = [a.arg for a in node.args.args]
            for a in args:
                if a in {"self", "cls"}: continue  # noqa: E701
                if a in STANDARD_PYTEST_FIXTURES: continue  # noqa: E701
                if a in local_fixtures: continue  # noqa: E701
                if a in fn_parametrize: continue  # noqa: E701
                if a.lower() in instr_tokens: continue  # noqa: E701
                unknown.add(a)
    return sorted(unknown)


def evaluate_task(task_dir: Path) -> dict:
    """Return a per-task verdict dict.

    Keys:
      kept: bool
      reason: str  — one of: '', 'no_test_file', 'syntax_error', 'compile_error',
                     'missing_import', 'all_skipped'
      bad_imports: list[str]
    """
    test_file = task_dir / "tests" / "test_solution.py"
    instruction_file = task_dir / "instruction.md"

    if not test_file.exists():
        return {"kept": False, "reason": "no_test_file", "bad_imports": []}

    try:
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=True) as tmp:
            py_compile.compile(
                str(test_file), cfile=tmp.name, doraise=True
            )
    except py_compile.PyCompileError as exc:
        return {"kept": False, "reason": f"syntax_error: {exc.msg.splitlines()[0][:80]}", "bad_imports": []}
    except Exception as exc:
        return {"kept": False, "reason": f"compile_error: {type(exc).__name__}", "bad_imports": []}

    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return {"kept": False, "reason": f"ast_syntax: {exc.msg[:60]}", "bad_imports": []}

    imports = _top_level_imports(tree)
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
        if name.lower() in instr_tokens:
            continue
        bad.append(name)

    if bad:
        return {"kept": False, "reason": "missing_import", "bad_imports": bad}

    if _all_tests_skipped(tree):
        return {"kept": False, "reason": "all_skipped", "bad_imports": []}

    # --- v3 extra checks ---

    # (a) ``from tests.X import Y`` at top level
    if _from_tests_import(tree):
        return {"kept": False, "reason": "from_tests_import", "bad_imports": []}

    # (b) known-broken sub-attributes of whitelisted packages
    broken_sub = _known_broken_subattrs(tree)
    if broken_sub:
        return {"kept": False, "reason": "broken_subattr", "bad_imports": broken_sub}

    # (c) module-level ``pytest.importorskip(...)`` — rubber-stamp or
    #     zero-collect at runtime
    if _has_module_level_importorskip(tree):
        return {"kept": False, "reason": "module_importorskip", "bad_imports": []}

    # (d) module-level chdir / environ / ctypes.CDLL on paths not in
    #     instruction (the env-var / file is almost never present).
    mlchdir = _module_level_chdir_or_env(tree, instr_tokens)
    if mlchdir is not None:
        return {"kept": False, "reason": mlchdir, "bad_imports": []}

    # (e) missing fixtures referenced as fn args (and no conftest.py shipped)
    has_conftest = any((task_dir / "tests").rglob("conftest.py")) if (task_dir / "tests").exists() else False
    missing_fix = _missing_fixtures(tree, has_conftest, instr_tokens)
    if missing_fix:
        return {"kept": False, "reason": "missing_fixtures", "bad_imports": missing_fix}

    return {"kept": True, "reason": "", "bad_imports": []}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    p.add_argument("--force", action="store_true",
                   help="Re-run even if marker file is present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    marker = root / MARKER_NAME
    if marker.exists() and not args.force and not args.dry_run:
        print(f"[patch_exp_rle_github_issue] marker present at {marker}; skipping (use --force to override)")
        return

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"[patch_exp_rle_github_issue] inspecting {len(task_dirs)} tasks under {root}")

    syntax_dropped = 0
    no_test_dropped = 0
    import_dropped = 0
    skip_dropped = 0
    # v3 buckets
    from_tests_dropped = 0
    broken_subattr_dropped = 0
    importorskip_dropped = 0
    chdir_env_dropped = 0
    missing_fix_dropped = 0
    kept = 0

    bad_import_samples: list[tuple[str, list[str]]] = []
    verdicts: dict[str, dict] = {}

    for td in task_dirs:
        v = evaluate_task(td)
        verdicts[td.name] = v
        if v["kept"]:
            kept += 1
            continue
        r = v["reason"]
        if r == "no_test_file":
            no_test_dropped += 1
        elif r.startswith("syntax_error") or r.startswith("ast_syntax") or r.startswith("compile_error"):
            syntax_dropped += 1
        elif r == "missing_import":
            import_dropped += 1
            if len(bad_import_samples) < args.show_bad:
                bad_import_samples.append((td.name, v["bad_imports"]))
        elif r == "all_skipped":
            skip_dropped += 1
        elif r == "from_tests_import":
            from_tests_dropped += 1
        elif r == "broken_subattr":
            broken_subattr_dropped += 1
            if len(bad_import_samples) < args.show_bad:
                bad_import_samples.append((td.name, v["bad_imports"]))
        elif r == "module_importorskip":
            importorskip_dropped += 1
        elif r.startswith("module_chdir") or r.startswith("module_env") or r.startswith("module_ctypes_cdll"):
            chdir_env_dropped += 1
        elif r == "missing_fixtures":
            missing_fix_dropped += 1
            if len(bad_import_samples) < args.show_bad:
                bad_import_samples.append((td.name, v["bad_imports"]))

    total = len(task_dirs)
    print()
    print(f"[patch_exp_rle_github_issue] total:           {total}")
    print(f"[patch_exp_rle_github_issue] no test file:    {no_test_dropped}")
    print(f"[patch_exp_rle_github_issue] syntax dropped:  {syntax_dropped}")
    print(f"[patch_exp_rle_github_issue] import dropped:  {import_dropped}")
    print(f"[patch_exp_rle_github_issue] all-skip dropped:{skip_dropped}")
    print(f"[patch_exp_rle_github_issue] from-tests drop: {from_tests_dropped}")
    print(f"[patch_exp_rle_github_issue] broken-subattr:  {broken_subattr_dropped}")
    print(f"[patch_exp_rle_github_issue] importorskip:    {importorskip_dropped}")
    print(f"[patch_exp_rle_github_issue] chdir/env/CDLL:  {chdir_env_dropped}")
    print(f"[patch_exp_rle_github_issue] missing fixture: {missing_fix_dropped}")
    print(f"[patch_exp_rle_github_issue] kept:            {kept}")

    if bad_import_samples:
        print()
        print(f"[patch_exp_rle_github_issue] sample bad-import drops (first {len(bad_import_samples)}):")
        for tid, bad in bad_import_samples:
            print(f"  {tid}: {bad}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_exp_rle_github_issue] wrote per-task verdicts: {args.report_json}")

    if args.dry_run:
        print("[patch_exp_rle_github_issue] dry-run: not deleting dropped tasks.")
        return

    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_exp_rle_github_issue] removed {removed} dropped task directories.")

    marker.write_text(json.dumps({
        "kept": kept,
        "total": total,
        "no_test": no_test_dropped,
        "syntax": syntax_dropped,
        "missing_import": import_dropped,
        "all_skipped": skip_dropped,
        "from_tests_import": from_tests_dropped,
        "broken_subattr": broken_subattr_dropped,
        "module_importorskip": importorskip_dropped,
        "module_chdir_env_cdll": chdir_env_dropped,
        "missing_fixtures": missing_fix_dropped,
    }, indent=2))
    print(f"[patch_exp_rle_github_issue] wrote idempotency marker: {marker}")


if __name__ == "__main__":
    main()
