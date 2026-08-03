#!/usr/bin/env python3
"""
exp_rle_error_report **v3** patcher.

Background
----------
laion/exp_rle_error_report-v2 was the v1-filtered output of the upstream
5,000-task pool (DCAgent/exp_rle_error_report). v1 applied
``patch_exp_rle_error_report_tasks.py`` (= the shared rle_flat25 filter):
drop tasks whose ``tests/test_solution.py`` imports an unknown top-level
module that is neither stdlib, in the container whitelist, nor named in
``instruction.md``. v1 produced ``laion/exp_rle_error_report-v2`` with
759 surviving tasks.

QC re-triage (2026-05-15) — 200 RL traces against v2:
  100% infra-ok, 5.0% solve (10/200). 190 fails were sampled and
  bucketed:

    fixture_not_found        60   30.0%
    module_not_found         39   19.5%
    import_name_not_found    36   18.0%
    uncategorized            19    9.5%   (mostly skipped-only or
                                            partial-pass)
    attr_not_found           12    6.0%
    partial_pass             10    5.0%
    filenotfound_subproc      4    2.0%
    pandas_v2_freq            4    2.0%
    torch_so_missing          3    1.5%
    no_output                 3    1.5%
    django_not_configured     1    0.5%
    pytest_warns_none         1    0.5%
    skeleton_notimpl          1    0.5%

  Three structural classes of leak survived v1:

    1. ``tests/test_solution.py`` uses pytest fixtures defined in an
       upstream ``conftest.py`` that is NOT shipped to the sandbox
       (60/200 = 30%). The agent has no way to provide e.g. a fixture
       called ``nomad_setup`` or ``ray_start_stop`` — pytest hard-errors
       at setup time. The verifier reports reward=0 deterministically;
       no amount of agent skill can solve it.

    2. ``test_solution.py`` imports a *deprecated/removed* submodule or
       symbol from a whitelisted top-level package (so v1's top-level
       allowlist passes, but the actual import line fails on the modern
       pinned version in the sandbox). Examples observed in v2 traces:
         * ``from pandas.util.testing import ...``
         * ``from pandas.core.index import ...``
         * ``import pandas.util._test_decorators as td`` →
           ``td.skip_if_no_scipy`` removed in modern pandas
         * ``from sklearn.utils.testing import ...``
         * ``from scipy.spatial.distance import kulsinski, wminkowski``
         * ``from scipy.signal.spectral import _spectral_helper``
         * ``import scipy._lib._numpy_compat``
         * ``from pydantic.datetime_parse import parse_date``
         * ``import tensorflow.python.X`` /
           ``import tensorflow.contrib.X``

    3. The test calls deprecated pytest APIs that hard-error on modern
       pytest, runs ``subprocess`` against system binaries not in the
       sandbox image (``ray``, ``docker``, ``kubectl``, ...), imports
       ``django.contrib.*`` without ``settings.configure()``, or imports
       a clearly-local upstream module name (``tests``, ``helpers``,
       ``apis``, ``conftest``, ``fixtures``).

Filter logic (cumulative — v3 = v1 + these new rules)
-----------------------------------------------------
A task is **dropped** if any of the following are true (in order):

  R1  (v1)  Top-level import not stdlib/whitelist/instruction-mentioned.
            Plus a small extra blocklist of upstream-style local names
            (``tests``, ``helpers``, ``apis``, ``conftest``, ``fixtures``,
            ``test_utils``, ``test_common``, ``test_helpers``,
            ``project``).
  R2  (v3)  Imports a known-deprecated/removed dotted submodule
            (``pandas.util.testing``, ``scipy.signal.spectral``,
            ``tensorflow.python``, ``pydantic.datetime_parse``,
            ``matplotlib.tests``, ``numpy.testing.decorators``, ...).
  R3  (v3)  ``from <module> import <name>`` where the symbol is known
            to have been removed (e.g. ``scipy.spatial.distance.kulsinski``,
            ``scipy.spatial.distance.wminkowski``).
  R4  (v3)  ``pytest.warns(None)`` — pytest>=8 raises TypeError.
  R5  (v3)  Imports ``django.contrib.*`` but never calls
            ``settings.configure()`` / sets ``DJANGO_SETTINGS_MODULE``
            / calls ``django.setup()``.
  R6  (v3)  Test calls ``subprocess.run([\"ray\", ...])`` or similar for
            a binary not in the sandbox (``ray``, ``docker``,
            ``kubectl``, ``minio``, ``redis-server``, ``mongod``,
            ``nomad``, ``consul``, ``helm``).
  R7  (v3)  Test function (or pytest-fixture function) requests a
            fixture by argument name, where the fixture is neither
            defined in the test file itself nor a built-in / well-known
            pytest fixture, and the argname is not a
            ``@pytest.mark.parametrize`` argument.
            Args with default values are explicitly excluded (pytest
            doesn't request fixtures for defaulted args).

Note that ``pytest_warns_none``/``django_unconfigured`` rules each only
catch 1 task in the 200-trial sample, but their precision is high
(they don't drop any solves). The biggest contributors are R7
(missing_fixture, 345 in full pool) and R2 (deprecated_module, 133).

Validation against 200-trial v2 sample
--------------------------------------
  Predicted drop: 140 of 190 fails (73.7% fail removal)
  Solves kept:    10 / 10  (100% recall on solves)
  Predicted kept solve rate: 16.7%  (vs 5.0% v2 baseline = **3.33x lift**)
  Full v2 → v3 pool: keep 261 / 759

Idempotency: this patcher only removes task directories; re-running on
an already-filtered tree is a no-op.

Usage
-----
  python data/patchers/patch_exp_rle_error_report_v3_tasks.py --root <tasks_dir>
  python data/patchers/patch_exp_rle_error_report_v3_tasks.py --root <tasks_dir> --dry-run
  python data/patchers/patch_exp_rle_error_report_v3_tasks.py --root <tasks_dir> --limit 200
  python data/patchers/patch_exp_rle_error_report_v3_tasks.py --root <tasks_dir> --report-json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

# Reuse R1 (the v1 top-level whitelist + instruction-mention check) from
# the existing shared evaluator. We import it for the constants only and
# re-implement the driver because v3 needs richer per-rule diagnostics
# than the v1 ``evaluate_task`` returns.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Re-use the shared constants.
from patch_rle_flat25_tasks import (  # noqa: E402
    STDLIB,
    ALWAYS_INSTALLED,
    PIP_WHITELIST,
    TOKEN_RE,
)

# ---------------------------------------------------------------------------
# v3 additional rules
# ---------------------------------------------------------------------------

# R1 extension — upstream-project-style local module names. A
# ``test_solution.py`` that does ``from tests import X`` will fail because
# the sandbox /tests dir only contains ``test_solution.py`` itself.
LIKELY_LOCAL_MODS: frozenset[str] = frozenset({
    "tests",
    "test_utils",
    "test_common",
    "test_helpers",
    "helpers",
    "apis",
    "conftest",
    "fixtures",
    "project",
})

# R2 — dotted-submodule prefixes that no longer exist in modern releases
# of whitelisted packages.
DEPRECATED_DOTTED_PREFIXES: tuple[str, ...] = (
    "pandas.util.testing",
    "pandas.core.index",
    "pandas.tests",
    "pandas.util._test_decorators",
    "pandas.io.json.normalize",
    "pandas.io.formats.terminal",
    "pandas.io.formats.printing",
    "pandas.compat.numpy",
    "sklearn.utils.testing",
    "scipy._lib._numpy_compat",
    "scipy.signal.spectral",
    "scipy.misc",
    "scipy.weave",
    "tensorflow.python",
    "tensorflow.contrib",
    "pydantic.datetime_parse",
    "numpy.testing.decorators",
    "matplotlib.tests",
)

# R3 — names removed from a *still-present* dotted module.
DEPRECATED_NAMES_BY_MODULE: dict[str, frozenset[str]] = {
    "scipy.spatial.distance": frozenset({
        "kulsinski", "wminkowski", "matching",
        "sokalmichener", "sokalsneath",
    }),
    "numpy": frozenset({
        "int_", "float_", "bool_", "object_", "str_",
        "long", "unicode", "typeDict",
    }),
}

# R6 — system binaries not present in the sandbox image. Test calls
# ``subprocess.run(["X", ...])`` where ``X`` is one of these will
# FileNotFoundError before any agent solution can run.
SYSTEM_BINARIES_REQUIRED: frozenset[str] = frozenset({
    "ray", "docker", "kubectl", "minio",
    "redis-server", "mongod", "nomad", "consul", "helm",
})

# R7 — pytest builtin/plugin fixtures that are always available.
KNOWN_FIXTURES: frozenset[str] = frozenset({
    # core pytest
    "request",
    "tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory",
    "capfd", "capsys", "capfdbinary", "capsysbinary", "capteesys",
    "caplog", "monkeypatch", "cache", "doctest_namespace", "recwarn",
    "record_property", "record_testsuite_property", "record_xml_attribute",
    "pytestconfig",
    # pytest-mock
    "mocker", "class_mocker", "module_mocker", "package_mocker",
    "session_mocker",
    # Faker / pytest-faker
    "faker", "_session_faker",
    # requests-mock
    "requests_mock",
    # anyio
    "anyio_backend", "anyio_backend_name", "anyio_backend_options",
    # ddtrace
    "ddspan", "ddtracer",
    # pytest-subtests
    "subtests",
    # pytest-asyncio side fixtures occasionally exposed by plugins
    "free_tcp_port", "free_tcp_port_factory",
    "free_udp_port", "free_udp_port_factory",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instruction_tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def _collect_imports(tree: ast.AST) -> tuple[set[str], set[str], list[tuple[str, list[str]]]]:
    """Return (top_level_names, dotted_module_paths, from_import_pairs)."""
    top: set[str] = set()
    dotted: set[str] = set()
    fromimports: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    top.add(alias.name.split(".")[0])
                    dotted.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                top.add(node.module.split(".")[0])
                dotted.add(node.module)
                fromimports.append((node.module, [a.name for a in node.names]))
    return top, dotted, fromimports


def _parametrize_argnames(tree: ast.AST) -> set[str]:
    """Return argnames declared by any ``@pytest.mark.parametrize(<arg1>, ...)``.

    Supports both the string form (``"a,b,c"``) and the tuple/list form
    (``("a","b","c")`` or ``["a","b","c"]``).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            fn = ast.unparse(node.func)
        except Exception:
            fn = ""
        if not fn.endswith("parametrize"):
            continue
        if not node.args:
            continue
        a = node.args[0]
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            for s in re.split(r"[,\s]+", a.value):
                if s:
                    names.add(s)
        elif isinstance(a, (ast.Tuple, ast.List)):
            for elt in a.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
    return names


def _defaultless_args(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return positional + kw-only args that have NO default value.

    Pytest fixtures cannot have default values, so an arg with a default
    can never trigger a 'fixture not found' error.
    """
    args = fn.args.args
    n_def = len(fn.args.defaults)
    pos = args[: len(args) - n_def] if n_def else args
    kwonly: list[str] = []
    for i, kw in enumerate(fn.args.kwonlyargs):
        # kw_defaults[i] is None iff no default
        kw_defs = fn.args.kw_defaults or []
        if i >= len(kw_defs) or kw_defs[i] is None:
            kwonly.append(kw.arg)
    return [a.arg for a in pos] + kwonly


def _defined_fixture_names(tree: ast.AST) -> set[str]:
    """Return names of every function decorated with @pytest.fixture / @fixture."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            try:
                src_d = ast.unparse(d)
            except Exception:
                src_d = ""
            if "fixture" in src_d:
                out.add(node.name)
                break
    return out


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_task(task_dir: Path) -> dict:
    """Return a per-task verdict with rich diagnostics.

    Keys:
      kept: bool
      reason: str        — one of {no_test_file, ast_syntax,
                            missing_import, deprecated_module,
                            deprecated_symbol, pytest_warns_none,
                            django_unconfigured, system_binary,
                            missing_fixture}
      details: list[str] — the offending tokens (imports, fixtures, ...)
    """
    test_file = task_dir / "tests" / "test_solution.py"
    instruction_file = task_dir / "instruction.md"

    if not test_file.exists():
        return {"kept": False, "reason": "no_test_file", "details": []}

    src = test_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"kept": False, "reason": "ast_syntax", "details": [exc.msg[:80]]}

    instr_tokens: set[str] = set()
    if instruction_file.exists():
        instr_tokens = _instruction_tokens(
            instruction_file.read_text(encoding="utf-8", errors="replace")
        )

    top_imports, dotted_imports, fromimport_pairs = _collect_imports(tree)

    # R1 — top-level allowlist + local-module heuristic
    bad_top: list[str] = []
    for name in sorted(top_imports):
        if name in STDLIB:
            continue
        if name in ALWAYS_INSTALLED:
            continue
        if name in PIP_WHITELIST:
            continue
        if name.lower() in instr_tokens:
            continue
        if name in LIKELY_LOCAL_MODS:
            bad_top.append(f"local:{name}")
            continue
        bad_top.append(name)
    if bad_top:
        return {"kept": False, "reason": "missing_import", "details": bad_top}

    # R2 — deprecated dotted submodule
    bad_dot: list[str] = []
    for dotted in sorted(dotted_imports):
        for pref in DEPRECATED_DOTTED_PREFIXES:
            if dotted == pref or dotted.startswith(pref + "."):
                bad_dot.append(dotted)
                break
    if bad_dot:
        return {"kept": False, "reason": "deprecated_module", "details": bad_dot}

    # R3 — deprecated symbol from a still-valid module
    bad_sym: list[str] = []
    for dotted, names in fromimport_pairs:
        depr = DEPRECATED_NAMES_BY_MODULE.get(dotted, frozenset())
        for n in names:
            if n in depr:
                bad_sym.append(f"{dotted}.{n}")
    if bad_sym:
        return {"kept": False, "reason": "deprecated_symbol", "details": bad_sym}

    # R4 — pytest.warns(None)
    if re.search(r"pytest\.warns\(\s*None\b", src):
        return {"kept": False, "reason": "pytest_warns_none", "details": ["pytest.warns(None)"]}

    # R5 — django.contrib.* without settings configuration
    if any(d == "django.contrib" or d.startswith("django.contrib.") for d in dotted_imports):
        if (
            "settings.configure" not in src
            and "DJANGO_SETTINGS_MODULE" not in src
            and "django.setup" not in src
        ):
            return {"kept": False, "reason": "django_unconfigured", "details": []}

    # R6 — system-binary subprocess call
    for binary in SYSTEM_BINARIES_REQUIRED:
        list_arg = re.search(rf"""\[\s*["']{re.escape(binary)}["']\s*[,\]]""", src) is not None
        positional_call = re.search(
            rf"""(?:check_output|check_call|run|Popen)\s*\(\s*["']{re.escape(binary)}\s""",
            src,
        ) is not None
        if (list_arg and "subprocess" in src) or positional_call:
            return {
                "kept": False,
                "reason": f"system_binary:{binary}",
                "details": [binary],
            }

    # R7 — fixtures referenced but not defined in this file
    defined_fixtures = _defined_fixture_names(tree)
    param_argnames = _parametrize_argnames(tree)
    used_fixtures: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_test = node.name.startswith("test_")
            is_fixture = node.name in defined_fixtures
            if not (is_test or is_fixture):
                continue
            for argname in _defaultless_args(node):
                used_fixtures.add(argname)

    used_fixtures -= {"self", "cls"}
    used_fixtures -= param_argnames
    missing_fixtures = used_fixtures - defined_fixtures - KNOWN_FIXTURES
    if missing_fixtures:
        return {
            "kept": False,
            "reason": "missing_fixture",
            "details": sorted(missing_fixtures),
        }

    return {"kept": True, "reason": "", "details": []}


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
                   help="Print the first N drop samples per reason.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"[patch_exp_rle_error_report_v3] inspecting {len(task_dirs)} tasks under {root}")

    by_reason: dict[str, int] = {}
    samples: dict[str, list[tuple[str, list[str]]]] = {}
    verdicts: dict[str, dict] = {}
    kept = 0

    for td in task_dirs:
        v = evaluate_task(td)
        verdicts[td.name] = v
        if v["kept"]:
            kept += 1
            continue
        reason = v["reason"]
        by_reason[reason] = by_reason.get(reason, 0) + 1
        bucket = samples.setdefault(reason, [])
        if len(bucket) < args.show_bad:
            bucket.append((td.name, v["details"]))

    total = len(task_dirs)
    dropped = total - kept

    print()
    print(f"[patch_exp_rle_error_report_v3] total:           {total}")
    print(f"[patch_exp_rle_error_report_v3] kept:            {kept}")
    print(f"[patch_exp_rle_error_report_v3] dropped:         {dropped}")
    print("[patch_exp_rle_error_report_v3] dropped breakdown:")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:30s} {n:4d}")

    for reason, bucket in samples.items():
        print()
        print(f"[patch_exp_rle_error_report_v3] sample drops [{reason}]:")
        for tid, det in bucket:
            print(f"  {tid}: {det}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_exp_rle_error_report_v3] wrote per-task verdicts: {args.report_json}")

    if args.dry_run:
        print("[patch_exp_rle_error_report_v3] dry-run: not deleting dropped tasks.")
        return

    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_exp_rle_error_report_v3] removed {removed} dropped task directories.")


if __name__ == "__main__":
    main()
