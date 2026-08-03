#!/usr/bin/env python3
"""
exp_flat25_subtle_debug v3 patcher.

Background
----------
v2 (``patch_exp_flat25_subtle_debug_tasks.py``) addressed two failure
modes in v1:

  1. ``ModuleNotFoundError`` on synthesized ``tests/test_solution.py``
     that imported third-party packages not in the container image —
     filtered by ``patch_rle_flat25_tasks.evaluate_task`` which checks
     each test's top-level imports against ``stdlib + container pip
     whitelist + instruction.md tokens``.
  2. Vacuous "all skipped" pytest exits getting reward=1 — patched by
     rewriting ``tests/test.sh`` to gate reward on a parsed
     ``N passed`` count rather than the raw pytest exit code.

QC re-triage of laion/exp_flat25_subtle_debug-v2 (n=200 traces, headline
3.5% solve, 2026-05-15) shows the same disease leaks through under
several finer-grained patterns that v2's top-level import filter does
not catch. From the v2-kept 794 tasks:

  41.0%  ImportError (verifier ``e>=1, p=0, f=0``)
    e.g. ``from pandas.util.testing import ...`` (whitelist top-level
    ``pandas`` passes; ``pandas.util.testing`` was removed in pandas
    1.0)
    e.g. ``from sklearn.utils.testing import ...`` (renamed to
    ``_testing`` in sklearn 0.22)
    e.g. ``from tensorflow.python.keras.engine.input_layer import ...``
    (private TF API; gone)
    e.g. ``from scipy.spatial.distance import wminkowski`` (removed in
    scipy 1.11)
    e.g. ``from ddtrace.compat import ...`` (removed)

    Also "local-package" imports the v2 filter passes through because
    they appear as tokens in ``instruction.md``:
      - ``import app`` (task 5k-subtle-2603 instruction.md describes a
        pandas Series.quantile fix at ``/app/starter_code.py``; the
        synthesized test imports an unrelated Flask ``app`` from a
        different upstream project)
      - ``import helpers``, ``from tests.fixtures import ...``,
        ``from adapt.utils import *``

  33.5%  Collection/setup errors (``e>=1, p=0, f=0``) from
         pytest-fixture references with no resolvable definition.
    The test file ``test_solution.py`` references fixtures (via
    function parameters or ``@pytest.mark.usefixtures("name")``) that
    must be defined in a ``conftest.py`` at or above the test's
    directory. The task tarball ships ``tests/test_solution.py`` alone
    — no conftest — so the fixtures (``uk_tokenizer``, ``revert_etc``,
    ``ctre``, ``testapp``, ``host``, ``shell``, ``page``, etc.) can
    never be resolved.

  13.0%  Real test failures (``f>=1``).
  8.5%   All-skipped (``p=0, f=0, e=0``) — already scored 0 by v2.
  3.5%   Real wins.

Fix (v3)
--------
Three additive changes on top of v2's structure:

  (a) **Submodule denylist for whitelist packages.** Walk the AST and
      check the *full dotted path* of every ``import`` /
      ``from ... import`` against a denylist of submodules that no
      longer exist in the pip-whitelist-installed package versions
      (pandas >= 2, sklearn >= 1.3, tensorflow >= 2.16, scipy >= 1.11,
      numpy >= 2, ddtrace >= 2). Tasks importing these are dropped.

  (b) **Strict local-name drop.** v2's filter allowed top-level imports
      that appeared as tokens in ``instruction.md``. This is too
      lenient for names like ``app``, ``helpers``, ``tests``, ``test``,
      ``utils``, ``conftest`` — these are project-local module names
      that the agent cannot materialize at the import location
      (``/tests/test_solution.py`` cannot find ``/app/app.py`` as a
      top-level ``app`` package because ``/app`` is not on
      ``PYTHONPATH``, and the file at ``/app/starter_code.py`` is named
      differently from what the test expects). Drop on any such name
      regardless of instruction.md mentions.

  (c) **Fixture-orphan drop.** If ``test_solution.py`` references a
      pytest fixture (via test function parameter or
      ``@pytest.mark.usefixtures``) that is neither defined in the
      file (``@pytest.fixture``) nor a known pytest / pytest-plugin
      builtin (from pytest, pytest-mock, pytest-django, etc.) nor a
      ``parametrize`` parameter, drop the task. These tasks would
      always fail at collection time with
      ``fixture 'XXX' not found``.

Idempotency, layout, and the ``test.sh`` mutation are unchanged from
v2 — we re-import v2's ``patch_test_sh`` / ``NEW_TEST_SH`` /
``V2_MARKER``. Since v3 only tightens the *drop* filter, tasks that
survive v3 are a strict subset of tasks that survived v2, and the
``tests/test.sh`` file the trainer reads is byte-identical to v2's.
The v3 marker in the patched test.sh stays the same string so the
existing detector still treats the file as "already patched".

Usage
-----
    python data/patchers/patch_exp_flat25_subtle_debug_tasks_v3.py --root <dir>
    python data/patchers/patch_exp_flat25_subtle_debug_tasks_v3.py --root <dir> --dry-run

Target HF repo: ``laion/exp_flat25_subtle_debug-v3``.

Daytona snapshot caps are HARD: this patcher only drops; it never
grows the task set and never changes the Dockerfile, so snapshot
identity is preserved across v2 → v3.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

# Reuse v2's tests/test.sh rewrite (and the import-filter scaffold).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from patch_exp_flat25_subtle_debug_tasks import (  # noqa: E402
    patch_test_sh,
)

# ---------------------------------------------------------------------------
# Tightened drop filter
# ---------------------------------------------------------------------------

STDLIB: frozenset[str] = frozenset(getattr(sys, "stdlib_module_names", set()))

ALWAYS_INSTALLED: frozenset[str] = frozenset({
    "pytest", "_pytest", "py", "pluggy", "iniconfig", "packaging",
    "tomli", "exceptiongroup",
})

WHITELIST_IMPORTS: frozenset[str] = frozenset({
    "requests", "numpy", "pandas", "scipy", "sklearn", "torch",
    "tensorflow", "tf", "keras", "httpx", "aiohttp", "ddtrace",
    "django", "flask", "fastapi", "matplotlib", "seaborn", "PIL",
    "pydantic", "pytest_mock", "requests_mock", "faker", "yaml",
    "pytz", "cryptography", "bcrypt", "hypothesis",
})

# Dotted-path denylist: submodules of whitelist packages that no longer
# exist in the pip-whitelist-installed versions. Match is "full ==
# entry" OR "full.startswith(entry + '.')" so e.g.
# ``pandas.util.testing.assert_frame_equal`` matches
# ``pandas.util.testing``.
KNOWN_BAD_DOTTED: frozenset[str] = frozenset({
    # sklearn (>= 1.3)
    "sklearn.utils.testing",
    "sklearn.utils.mocking",
    "sklearn.externals",
    "sklearn.utils._IS_32BIT",
    # pandas (>= 2.0)
    "pandas.util.testing",
    "pandas.compat.lrange",
    "pandas.compat.zip",
    "pandas.compat.StringIO",
    "pandas.compat.PY37",
    "pandas.compat.PY2",
    "pandas.compat.PY3",
    "pandas.NumericIndex",
    "pandas.Float64Index",
    "pandas.Int64Index",
    "pandas.UInt64Index",
    "pandas.core.api.Float64Index",
    "pandas.core.api.NumericIndex",
    "pandas._libs.tslib.NaT",
    "pandas.core.dtypes.base.registry",
    "pandas.core.base.SpecificationError",
    # tensorflow (>= 2.16)
    "tensorflow.python",
    "tensorflow.contrib",
    "tensorflow.keras.wrappers.scikit_learn",
    # scipy (>= 1.11)
    "scipy.spatial.distance.wminkowski",
    "scipy.spatial.distance.kulsinski",
    "scipy.spatial.distance.matching",
    "scipy.spatial.distance.sokalmichener",
    "scipy.spatial.distance.sokalsneath",
    "scipy.signal.spectral",
    "scipy.optimize.optimize",
    # numpy (>= 2)
    "numpy.core._rational_tests",
    "numpy.compat",
    # ddtrace (>= 2)
    "ddtrace.compat",
})

# Local module names that we *always* drop. These can never be resolved
# in the container's import path because they aren't pip-installable and
# the agent's solution lives elsewhere (e.g. ``/app/starter_code.py``,
# not ``/tests/app/__init__.py``). v2 allowed these if they appeared as
# a token in instruction.md; that's too lax for these very-common
# substring tokens.
LOCAL_NAMES: frozenset[str] = frozenset({
    "app", "helpers", "tests", "test", "utils", "conftest",
    "adapt", "returns",
})

# Pytest builtin fixtures plus fixtures provided by whitelist plugins
# (pytest-mock, pytest-django, faker, anyio, etc.).
PYTEST_BUILTIN_FIXTURES: frozenset[str] = frozenset({
    # core pytest
    "request", "tmp_path", "tmp_path_factory", "tmpdir",
    "tmpdir_factory", "monkeypatch", "capsys", "capsysbinary",
    "capfd", "capfdbinary", "caplog", "recwarn", "pytestconfig",
    "cache", "doctest_namespace", "record_property",
    "record_xml_attribute", "record_testsuite_property",
    "testdir", "pytester",
    # pytest-mock
    "mocker", "class_mocker", "module_mocker", "package_mocker",
    "session_mocker",
    # pytest-requests-mock
    "requests_mock", "rm", "requests_mocker",
    # pytest-django (django in WHITELIST -> pytest-django not installed
    # by default in test.sh, but tests use these names; keep liberal)
    "db", "transactional_db", "admin_user", "admin_client",
    "client", "rf", "settings", "live_server",
    # faker
    "faker",
    # anyio / pytest-asyncio
    "event_loop", "anyio_backend",
    # hypothesis
    "data",
    # method receivers (not fixtures, but appear as first arg of test
    # methods on classes)
    "self", "cls",
})

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _all_imports(tree: ast.AST) -> list[tuple[str, str]]:
    """Return ``[(full_dotted_path, top_level_pkg)]`` for absolute imports."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.append((alias.name, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                out.append((node.module, node.module.split(".")[0]))
    return out


def _defined_fixtures(tree: ast.AST) -> set[str]:
    """Names defined via ``@pytest.fixture`` (or ``@pytest.fixture(...)``).

    Honors ``name=`` keyword to override the function name.
    """
    fixtures: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            name_node = dec.func if isinstance(dec, ast.Call) else dec
            is_fixture = False
            if isinstance(name_node, ast.Attribute) and name_node.attr == "fixture":
                is_fixture = True
            elif isinstance(name_node, ast.Name) and name_node.id == "fixture":
                is_fixture = True
            if is_fixture:
                fixtures.add(node.name)
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if (kw.arg == "name"
                                and isinstance(kw.value, ast.Constant)
                                and isinstance(kw.value.value, str)):
                            fixtures.add(kw.value.value)
    return fixtures


def _test_fn_params(tree: ast.AST) -> set[str]:
    """Parameter names of ``def test_*`` functions (recursively).

    Strips the first parameter of methods inside ``class``-bodies (``self``).
    """
    out: list[str] = []

    def visit(node: ast.AST, in_class: bool) -> None:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit(child, True)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") or node.name == "test":
                args = [a.arg for a in node.args.args]
                if in_class and args and args[0] == "self":
                    args = args[1:]
                out.extend(args)
            for child in node.body:
                visit(child, in_class)

    for child in tree.body:
        visit(child, False)
    return set(out)


def _usefixtures(tree: ast.AST) -> set[str]:
    """Fixture names referenced by ``@pytest.mark.usefixtures(...)``."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "usefixtures"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.add(arg.value)
    return out


def _parametrize_names(tree: ast.AST) -> set[str]:
    """Parameter names declared via ``@pytest.mark.parametrize(...)``.

    Supports the three argspec shapes:
      - ``"a, b"`` (string)
      - ``("a", "b")`` (tuple)
      - ``["a", "b"]`` (list)
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "parametrize"):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            for n in re.split(r"[,\s]+", arg.value):
                if n:
                    out.add(n)
        elif isinstance(arg, (ast.Tuple, ast.List)):
            for el in arg.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    out.add(el.value)
    return out


def evaluate_task_v3(task_dir: Path) -> dict:
    """Return a per-task verdict dict.

    Keys:
      kept: bool
      reason: str — one of:
        "no_test_file"
        "syntax_error: <msg>"
        "missing_import"            (v2-style: top-level not whitelist & no
                                     instruction mention)
        "bad_submodule"             (deprecated submodule of whitelist pkg)
        "local_name"                (always-local import name)
        "orphan_fixtures"           (test references unresolvable fixtures)
      bad_imports: list[str]
      orphan_fixtures: list[str]
    """
    test_file = task_dir / "tests" / "test_solution.py"
    instruction_file = task_dir / "instruction.md"

    if not test_file.exists():
        return {"kept": False, "reason": "no_test_file",
                "bad_imports": [], "orphan_fixtures": []}

    try:
        src = test_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"kept": False, "reason": f"read_error: {type(exc).__name__}",
                "bad_imports": [], "orphan_fixtures": []}

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"kept": False,
                "reason": f"syntax_error: {(exc.msg or '')[:60]}",
                "bad_imports": [], "orphan_fixtures": []}

    instr_tokens: set[str] = set()
    if instruction_file.exists():
        try:
            instr_text = instruction_file.read_text(
                encoding="utf-8", errors="replace")
            instr_tokens = {t.lower() for t in _TOKEN_RE.findall(instr_text)}
        except OSError:
            pass

    # 1. Import filter
    imports = _all_imports(tree)
    bad_imports: list[str] = []
    bad_kind: str | None = None

    for full, top in imports:
        if top in STDLIB:
            continue
        if top in ALWAYS_INSTALLED:
            continue
        # (a) submodule denylist
        flagged_sub = None
        for kbd in KNOWN_BAD_DOTTED:
            if full == kbd or full.startswith(kbd + "."):
                flagged_sub = full
                break
        if flagged_sub:
            bad_imports.append(flagged_sub)
            bad_kind = "bad_submodule"
            break
        if top in WHITELIST_IMPORTS:
            continue
        # (b) strict local-name drop
        if top.lower() in LOCAL_NAMES:
            bad_imports.append(top)
            bad_kind = "local_name"
            break
        # v2 fallback: token in instruction.md
        if top.lower() in instr_tokens:
            continue
        bad_imports.append(top)
        bad_kind = "missing_import"
        break

    if bad_imports:
        return {"kept": False, "reason": bad_kind, "bad_imports": bad_imports,
                "orphan_fixtures": []}

    # 2. (c) Fixture-orphan filter
    defined = _defined_fixtures(tree)
    used_marks = _usefixtures(tree)
    fn_params = _test_fn_params(tree)
    parametrize = _parametrize_names(tree)

    needed = (fn_params | used_marks) - defined - PYTEST_BUILTIN_FIXTURES - parametrize
    if needed:
        return {"kept": False, "reason": "orphan_fixtures", "bad_imports": [],
                "orphan_fixtures": sorted(needed)}

    return {"kept": True, "reason": "", "bad_imports": [], "orphan_fixtures": []}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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
                   help="Print the first N bad-import / orphan samples.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"[patch_exp_flat25_subtle_debug_v3] inspecting "
          f"{len(task_dirs)} tasks under {root}")

    counts: dict[str, int] = {
        "no_test_file": 0,
        "syntax_error": 0,
        "missing_import": 0,
        "bad_submodule": 0,
        "local_name": 0,
        "orphan_fixtures": 0,
        "kept": 0,
        "other": 0,
    }
    patch_counts: dict[str, int] = {
        "patched": 0,
        "patched_unusual": 0,
        "already": 0,
        "missing": 0,
        "skipped_no_pytest": 0,
        "unparseable": 0,
    }

    bad_samples: list[tuple[str, str, list[str]]] = []
    orphan_samples: list[tuple[str, list[str]]] = []
    verdicts: dict[str, dict] = {}

    for td in task_dirs:
        v = evaluate_task_v3(td)
        verdicts[td.name] = v
        reason = v["reason"]
        if v["kept"]:
            counts["kept"] += 1
            test_sh = td / "tests" / "test.sh"
            result = patch_test_sh(test_sh, dry_run=args.dry_run)
            patch_counts[result] = patch_counts.get(result, 0) + 1
            continue
        if reason == "no_test_file":
            counts["no_test_file"] += 1
        elif reason.startswith("syntax_error") or reason.startswith("read_error"):
            counts["syntax_error"] += 1
        elif reason == "missing_import":
            counts["missing_import"] += 1
            if len(bad_samples) < args.show_bad:
                bad_samples.append((td.name, "missing_import", v["bad_imports"]))
        elif reason == "bad_submodule":
            counts["bad_submodule"] += 1
            if len(bad_samples) < args.show_bad:
                bad_samples.append((td.name, "bad_submodule", v["bad_imports"]))
        elif reason == "local_name":
            counts["local_name"] += 1
            if len(bad_samples) < args.show_bad:
                bad_samples.append((td.name, "local_name", v["bad_imports"]))
        elif reason == "orphan_fixtures":
            counts["orphan_fixtures"] += 1
            if len(orphan_samples) < args.show_bad:
                orphan_samples.append((td.name, v["orphan_fixtures"]))
        else:
            counts["other"] += 1

    total = len(task_dirs)
    print()
    print(f"[patch_exp_flat25_subtle_debug_v3] total:           {total}")
    for k in ("no_test_file", "syntax_error", "missing_import",
              "bad_submodule", "local_name", "orphan_fixtures",
              "other", "kept"):
        print(f"[patch_exp_flat25_subtle_debug_v3] {k:<16}: {counts[k]:>5}")
    print()
    print("[patch_exp_flat25_subtle_debug_v3] test.sh mutation (on KEPT tasks):")
    for k in ("patched", "patched_unusual", "already",
              "missing", "skipped_no_pytest", "unparseable"):
        print(f"  {k:<20}: {patch_counts.get(k, 0):>5}")

    if bad_samples:
        print()
        print(f"[patch_exp_flat25_subtle_debug_v3] sample drop reasons (first {len(bad_samples)}):")
        for tid, kind, bad in bad_samples:
            print(f"  {tid}: ({kind}) {bad}")
    if orphan_samples:
        print()
        print(f"[patch_exp_flat25_subtle_debug_v3] sample orphan fixtures (first {len(orphan_samples)}):")
        for tid, orphans in orphan_samples:
            print(f"  {tid}: {orphans}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_exp_flat25_subtle_debug_v3] wrote per-task verdicts: "
              f"{args.report_json}")

    if args.dry_run:
        print("[patch_exp_flat25_subtle_debug_v3] dry-run: not deleting dropped tasks.")
        return

    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_exp_flat25_subtle_debug_v3] removed {removed} dropped task directories.")


if __name__ == "__main__":
    main()
