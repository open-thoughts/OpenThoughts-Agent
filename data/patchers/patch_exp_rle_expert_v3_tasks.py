#!/usr/bin/env python3
"""
exp_rle_expert **v3** patcher.

Background
----------
laion/exp_rle_expert-v2 was the v1-filtered output of the upstream
5,000-task pool (DCAgent/exp_rle_expert). v1 applied
``patch_exp_rle_expert_tasks.py`` (= the shared rle_flat25 filter):
drop tasks whose ``tests/test_solution.py`` imports an unknown top-level
module not in stdlib / container whitelist / ``instruction.md`` token
set, plus drop the obvious all-skip rubber-stamps (module-level
``pytestmark = pytest.mark.skip[if](...)`` or every test decorated with
``@pytest.mark.skip[if]``). v1 produced ``laion/exp_rle_expert-v2`` with
730 surviving tasks.

QC re-triage (2026-05-15) — 203 RL traces against v2:
  100% infra-ok, 8.4% solve (17/203). All 17 v2-solves are rubber
  stamps. 186 failures split as:

    Collection error (ModuleNotFoundError):   86  (46%)
       pandas.util.testing                    11
       tzdata                                  8
       tests (project-local)                   7
       tensorflow.python                       4
       pandas.tests.*                          3
       numpy.* (moved/removed)                 5
       keras.* (Keras 3 reshuffle)             4
       sklearn.externals / utils.seq_dataset   2
       pipelinewise / bokeh / app / src / ...  7
    Real test failures (FAILED)               33  (18%)
    "no tests collected" (empty/commented)     7   (4%)
    Other / fixture / setup errors            57  (31%)

  17 v2-solves all match one of three rubber-stamp signatures:

    A. ``trivial_assertions``  — every active test body reduces to
       ``assert True``, ``assert 1 == 1``, ``assert 1 is 1``, or
       ``@pytest.mark.xfail(strict=True)`` + ``assert False`` (xpass).
       7/17 solves (e.g. 5k-expert-1599 ``test_always_true``,
       5k-expert-2490 ``test_example: assert True == True``,
       5k-expert-4783 ``test_sanity: assert 1 is 1``).

    B. ``no_impl_referenced``  — test imports only stdlib / pytest /
       pip-whitelist packages AND every ``Load`` Name inside every
       test body resolves to a module-level import / module-level
       definition / local binding / Python builtin. The agent's
       ``solution.py`` is never imported and never referenced —
       nothing the agent does can affect the test outcome.
       9/17 solves (e.g. 5k-expert-0000 imports ``ddtrace.compat``
       which already ships ``to_unicode/reraise/...`` so the installed
       package's exports satisfy the test without any agent code;
       5k-expert-1028 is a stdlib-only Python-tuples tutorial;
       5k-expert-2733 defines an inline ``solution(text): ...`` in
       every test method).

    C. ``no_test_functions``  — ``test_solution.py`` has zero
       ``test_*`` functions (entire file commented out). pytest
       collects 0 items and the verifier exits with non-zero, so
       these never solve in v2 traces, but they're dead weight.

Filter logic (cumulative — v3 = v1 + v3 extensions; v3 extensions are
the **shared** error_report-v3 rules R1-R7 plus the two new
expert-specific rules R8 + R10 below)
----------------------------------------------------------------------
A task is **dropped** if any of the following are true (in order):

  R1  (v1)  Top-level import not stdlib/whitelist/instruction-mentioned
            (provided by ``patch_exp_rle_error_report_v3_tasks``'s
            ``LIKELY_LOCAL_MODS`` extension).
  R2  (v3 shared)  Imports a known-deprecated/removed dotted submodule
            (``pandas.util.testing``, ``scipy.signal.spectral``,
            ``tensorflow.python``, ``sklearn.utils.testing``,
            ``pydantic.datetime_parse``, ``matplotlib.tests``,
            ``numpy.testing.decorators``, ...).
  R3  (v3 shared)  ``from <module> import <name>`` where the symbol is
            known to have been removed
            (e.g. ``scipy.spatial.distance.kulsinski``).
  R4  (v3 shared)  ``pytest.warns(None)`` — pytest>=8 raises TypeError.
  R5  (v3 shared)  Imports ``django.contrib.*`` but never calls
            ``settings.configure()`` / sets ``DJANGO_SETTINGS_MODULE``
            / calls ``django.setup()``.
  R6  (v3 shared)  Test calls ``subprocess.run(["ray", ...])`` or
            similar for a binary not in the sandbox (``ray``, ``docker``,
            ``kubectl``, ``nomad``, ...).
  R7  (v3 shared)  Test function (or pytest-fixture function) requests
            a fixture by argument name where the fixture is neither
            defined in the test file itself nor a well-known pytest
            builtin/plugin fixture.

  R8  (v3 expert)  ``no_test_functions``  — no ``test_*`` function or
            method exists. The whole file is comments / imports /
            docstrings, so pytest collects 0 items and ``test.sh``
            exits non-zero regardless of agent action.
  R9  (v3 expert)  ``trivial_assertions``  — every active (non-skipped)
            test body is "outcome-independent of agent": only
            ``pass`` / docstring / ``del`` / ``assert True`` /
            ``assert <constant comparison that is statically True>``
            / ``@pytest.mark.xfail(strict=True)`` + ``assert False``.
            Every solve in this bucket is a 100% positive reward
            regardless of agent action.
  R10 (v3 expert)  ``no_impl_referenced``  — test imports only
            stdlib / always-installed / pip-whitelist packages
            (no impl-local import, no relative import) AND every
            ``Load`` Name inside every test body resolves to an
            imported name, a module-level definition, a local
            binding (incl. nested-function args, comprehension
            targets, ``for``/``with`` targets, ``except`` aliases),
            or a Python builtin. The agent's ``solution.py`` cannot
            be imported by the test → cannot affect the outcome →
            rubber-stamp.

Validation against 203-trial v2 sample
--------------------------------------
  Predicted drop: 191 / 203  (16/17 solves dropped as rubber-stamps,
                              175/186 fails dropped — 94% of fails
                              are statically-detectable junk)
  Solves kept:    1  / 17    (only legitimate solve survives —
                              5k-expert-0008 ``tools.ci.tc.decision``
                              implementation, all 18 tests pass)
  Predicted kept slice solve rate: 8.3%  (vs 8.4% v2 baseline; but
                                          the v2 8.4% was 100%
                                          rubber-stamps; v3 keeps
                                          the 1 real solve)
  Full v2 → v3 pool: keep 39 / 730  (5.3%)

Idempotency: this patcher only removes task directories; re-running
on an already-filtered tree is a no-op. A marker file
``.exp_rle_expert_v3_patched`` is written to ``--root`` on success.

Usage
-----
  python data/patchers/patch_exp_rle_expert_v3_tasks.py --root <tasks_dir>
  python data/patchers/patch_exp_rle_expert_v3_tasks.py --root <tasks_dir> --dry-run
  python data/patchers/patch_exp_rle_expert_v3_tasks.py --root <tasks_dir> --limit 200
  python data/patchers/patch_exp_rle_expert_v3_tasks.py --root <tasks_dir> --report-json out.json
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import shutil
import sys
from pathlib import Path

# Reuse the shared error_report-v3 R1-R7 evaluator (R1..R7 in this module's
# numbering) and the v1 helpers for active-test discovery.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from patch_exp_rle_error_report_v3_tasks import (  # noqa: E402
    evaluate_task as _eval_error_report_v3,
    STDLIB,
    ALWAYS_INSTALLED,
    PIP_WHITELIST,
)
from patch_exp_rle_expert_tasks import (  # noqa: E402
    _iter_test_functions,
    _decorator_skips,
)

MARKER_FILENAME = ".exp_rle_expert_v3_patched"


# ---------------------------------------------------------------------------
# R9: trivial-assertion rubber-stamp detector
# ---------------------------------------------------------------------------

def _is_tautological_assertion(expr: ast.AST) -> bool:
    """True if ``assert <expr>`` is statically true regardless of program
    state. Covers ``assert True``, ``assert 1``, ``assert 1 == 1``,
    ``assert 1 is 1``, ``assert 5 != 6``, etc."""
    if isinstance(expr, ast.Constant):
        return bool(expr.value)
    if (
        isinstance(expr, ast.Compare)
        and len(expr.comparators) == 1
        and isinstance(expr.left, ast.Constant)
        and isinstance(expr.comparators[0], ast.Constant)
    ):
        op = expr.ops[0]
        lv = expr.left.value
        rv = expr.comparators[0].value
        if isinstance(op, ast.Eq):
            return lv == rv
        if isinstance(op, ast.NotEq):
            return lv != rv
        if isinstance(op, ast.Is):
            return lv is rv or lv == rv
        if isinstance(op, ast.IsNot):
            return lv is not rv
    return False


def _is_assert_false_literal(expr: ast.AST) -> bool:
    """True iff ``assert <expr>`` is ``assert False`` (literal)."""
    return isinstance(expr, ast.Constant) and expr.value is False


def _is_xfail_strict(decorator_list: list[ast.expr]) -> bool:
    """True iff any decorator is ``@pytest.mark.xfail(..., strict=True, ...)``."""
    for d in decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if not (isinstance(target, ast.Attribute) and target.attr == "xfail"):
            continue
        # Walk the dotted chain to confirm ``mark`` appears (so this is
        # ``pytest.mark.xfail`` and not some unrelated ``foo.xfail``).
        cur: ast.AST = target.value
        chain_ok = False
        while isinstance(cur, ast.Attribute):
            if cur.attr == "mark":
                chain_ok = True
            cur = cur.value
        if not chain_ok:
            continue
        if isinstance(d, ast.Call):
            for kw in d.keywords:
                if (
                    kw.arg == "strict"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    return True
    return False


def _test_body_is_outcome_independent(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff every statement in ``func.body`` is a no-op or a
    statically-true assertion (or an xfail-strict ``assert False`` xpass)."""
    xfail_strict = _is_xfail_strict(func.decorator_list)
    for stmt in func.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr):
            # docstring or bare-constant statement
            if isinstance(stmt.value, ast.Constant):
                continue
            return False
        if isinstance(stmt, ast.Delete):
            continue
        if isinstance(stmt, ast.Assert):
            if _is_tautological_assertion(stmt.test):
                continue
            if xfail_strict and _is_assert_false_literal(stmt.test):
                continue
            return False
        return False
    return True


# ---------------------------------------------------------------------------
# R10: "no impl referenced" — test cannot import the agent's solution
# ---------------------------------------------------------------------------

def _has_local_or_relative_import(tree: ast.Module) -> bool:
    """True iff ``tree`` has any import that resolves to a project-local
    module (relative ``from . import X`` OR top-level package not in
    stdlib / always-installed / pip-whitelist)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in STDLIB or top in ALWAYS_INSTALLED or top in PIP_WHITELIST:
                    continue
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                return True
            if node.module:
                top = node.module.split(".")[0]
                if top in STDLIB or top in ALWAYS_INSTALLED or top in PIP_WHITELIST:
                    continue
                return True
    return False


def _module_level_known_names(tree: ast.Module) -> set[str]:
    """Names bound at module level: imports + ``def``/``class``/Assign
    targets. Builtin names are added on top."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                names.add(a.asname or a.name)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.add(node.name)
    names.update(dir(builtins))
    return names


class _NameAnalyzer(ast.NodeVisitor):
    """Walk a function body and collect every locally-bound name plus
    every ``Load`` Name reference. Used to detect tests that reference
    nothing outside the test file."""

    def __init__(self, root_known: set[str]) -> None:
        self.bound: set[str] = set()
        self.loaded: set[str] = set()
        self.root_known: set[str] = root_known

    def _collect_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
            self.bound.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_target(elt)
        elif isinstance(target, ast.Starred):
            self._collect_target(target.value)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(a.arg)
        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)
        for s in node.body:
            self.visit(s)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(a.arg)
        if node.args.vararg:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.bound.add(node.args.kwarg.arg)
        self.visit(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        for s in node.body:
            self.visit(s)

    def visit_For(self, node: ast.For) -> None:
        self._collect_target(node.target)
        self.visit(node.iter)
        for s in node.body + node.orelse:
            self.visit(s)

    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def visit_With(self, node: ast.With) -> None:
        for it in node.items:
            self.visit(it.context_expr)
            if it.optional_vars:
                self._collect_target(it.optional_vars)
        for s in node.body:
            self.visit(s)

    visit_AsyncWith = visit_With  # type: ignore[assignment]

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._collect_target(t)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._collect_target(node.target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._collect_target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._collect_target(node.target)
        self.visit(node.value)

    def _visit_comp_generators(self, generators: list[ast.comprehension]) -> None:
        for g in generators:
            self._collect_target(g.target)
            self.visit(g.iter)
            for c in g.ifs:
                self.visit(c)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comp_generators(node.generators)
        self.visit(node.elt)

    visit_SetComp = visit_ListComp  # type: ignore[assignment]
    visit_GeneratorExp = visit_ListComp  # type: ignore[assignment]

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comp_generators(node.generators)
        self.visit(node.key)
        self.visit(node.value)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        for s in node.body:
            self.visit(s)

    def visit_Global(self, node: ast.Global) -> None:
        for n in node.names:
            self.bound.add(n)

    visit_Nonlocal = visit_Global  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.bound.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            if a.name == "*":
                continue
            self.bound.add(a.asname or a.name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.bound.add(node.id)

    def has_unknown(self) -> bool:
        known = self.root_known | self.bound
        return any(n not in known for n in self.loaded)


def _no_impl_referenced(tree: ast.Module) -> bool:
    """True iff (a) no import in this file resolves to a project-local
    module, AND (b) every ``Load`` Name inside every test function/method
    body is either imported, defined at module level, locally bound, or a
    Python builtin. Implies the agent's solution.py can never influence
    the test outcome."""
    if _has_local_or_relative_import(tree):
        return False
    root_known = _module_level_known_names(tree)
    test_funcs = _iter_test_functions(tree)
    if not test_funcs:
        return False
    for f in test_funcs:
        an = _NameAnalyzer(root_known)
        for a in f.args.args + f.args.kwonlyargs + f.args.posonlyargs:
            an.bound.add(a.arg)
        if f.args.vararg:
            an.bound.add(f.args.vararg.arg)
        if f.args.kwarg:
            an.bound.add(f.args.kwarg.arg)
        for s in f.body:
            an.visit(s)
        if an.has_unknown():
            return False
    return True


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_task(task_dir: Path) -> dict:
    """Return a per-task verdict.

    Keys:
      kept: bool
      reason: str        — one of the v1/v3 rule names
      details: list[str] — offending tokens / fixture / func names
    """
    test_file = task_dir / "tests" / "test_solution.py"
    if not test_file.exists():
        return {"kept": False, "reason": "no_test_file", "details": []}

    # R1-R7 via the shared evaluator (covers missing_import,
    # deprecated_module, deprecated_symbol, pytest_warns_none,
    # django_unconfigured, system_binary, missing_fixture).
    v = _eval_error_report_v3(task_dir)
    if not v["kept"]:
        return v

    src = test_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"kept": False, "reason": "ast_syntax", "details": [exc.msg[:80]]}

    test_funcs = _iter_test_functions(tree)

    # R8 — file has no test functions at all (comments-out / empty).
    if not test_funcs:
        return {"kept": False, "reason": "no_test_functions", "details": []}

    # R9 — every active test body is outcome-independent of the agent.
    active = [f for f in test_funcs if not _decorator_skips(f)]
    if active and all(_test_body_is_outcome_independent(f) for f in active):
        return {
            "kept": False,
            "reason": "trivial_assertions",
            "details": [f.name for f in active][:5],
        }

    # R10 — no impl import + every test name resolves locally.
    if _no_impl_referenced(tree):
        return {"kept": False, "reason": "no_impl_referenced", "details": []}

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
    p.add_argument("--force", action="store_true",
                   help="Re-run even if the idempotency marker is present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    marker = root / MARKER_FILENAME
    if marker.exists() and not args.force and not args.dry_run:
        print(f"[patch_exp_rle_expert_v3] marker present at {marker}; skipping (use --force to re-run).")
        return

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    print(f"[patch_exp_rle_expert_v3] inspecting {len(task_dirs)} tasks under {root}")

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
            bucket.append((td.name, v.get("details", [])))

    total = len(task_dirs)
    dropped = total - kept

    print()
    print(f"[patch_exp_rle_expert_v3] total:           {total}")
    print(f"[patch_exp_rle_expert_v3] kept:            {kept}")
    print(f"[patch_exp_rle_expert_v3] dropped:         {dropped}")
    print("[patch_exp_rle_expert_v3] dropped breakdown:")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:30s} {n:4d}")

    for reason, bucket in samples.items():
        print()
        print(f"[patch_exp_rle_expert_v3] sample drops [{reason}]:")
        for tid, det in bucket:
            print(f"  {tid}: {det}")

    if args.report_json:
        args.report_json.write_text(json.dumps(verdicts, indent=2))
        print(f"[patch_exp_rle_expert_v3] wrote per-task verdicts: {args.report_json}")

    if args.dry_run:
        print("[patch_exp_rle_expert_v3] dry-run: not deleting dropped tasks.")
        return

    removed = 0
    for tid, v in verdicts.items():
        if v["kept"]:
            continue
        target = root / tid
        if target.exists():
            shutil.rmtree(target)
            removed += 1
    print(f"[patch_exp_rle_expert_v3] removed {removed} dropped task directories.")

    marker.write_text(
        f"total={total}\n"
        f"kept={kept}\n"
        f"removed={removed}\n"
    )
    print(f"[patch_exp_rle_expert_v3] wrote idempotency marker: {marker}")


if __name__ == "__main__":
    main()
