#!/usr/bin/env python3
"""
Patch DCAgent/exp_rpt_softwareheritage-large with an *auto-install* strategy
(supersedes the old filter-only patcher).

Rationale
---------

The previous filter-only strategy dropped ~95% of tasks (229 / 4998 = 4.6%
kept) — well below the 30% bar. The dropped majority were tasks whose tests
import third-party packages that ARE pip-installable but the prior strategy
didn't try to install them.

The Dockerfile (`python:3.10-slim` + `pip install pytest`) only pre-installs
pytest, but `tests/test.sh` runs at verifier time and CAN do `pip install`.
So this patcher INJECTS `pip install <pkg> || true` lines into test.sh for
each non-stdlib import that resolves on PyPI, and only drops the task when:

  1. `test_solution.py` fails py_compile (unfixable syntax error), OR
  2. an import is project-internal (e.g. `myapp.foo`, `tests.helpers`, or
     a name beginning with `_`), OR
  3. an import name doesn't resolve on PyPI (HEAD `https://pypi.org/pypi/<n>/json`).

Already-installed names (pytest, the agent's `solution` module, stdlib) are
skipped (no install line generated).

The injected lines are wrapped in `|| true` so a single failed install does
not abort the verifier — pytest will then fail on the missing import, which
is the correct semantic (a real test failure rather than a fixture crash).

A `# --- laion v2 patch: auto-install test deps ---` marker makes the patch
idempotent.

PyPI HEAD checks are cached at /tmp/pypi_cache.json, keyed by import name.
Use `--no-pypi-check` to skip the network calls (useful for offline reruns:
treats every unknown name as known → installs are attempted at runtime,
which is the same fail-soft behavior).

Usage:
    python patch_softwareheritage_tasks.py --root /path/to/tasks
    python patch_softwareheritage_tasks.py --root /path/to/tasks --dry-run
    python patch_softwareheritage_tasks.py --root /path/to/tasks --limit 50
    python patch_softwareheritage_tasks.py --root /path/to/tasks --no-pypi-check
"""

from __future__ import annotations

import argparse
import ast
import json
import py_compile
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist (additions to sys.stdlib_module_names)
# ---------------------------------------------------------------------------

# Modules pre-installed in the task's daytona container OR provided by /app.
# Anything here is treated as "no install needed".
EXTRA_ALLOWED = frozenset({
    "pytest",
    "_pytest",   # part of the pytest install
    "solution",  # /app/solution.py — agent's canonical output
})

# Common import-name → PyPI-package-name mismatches.
# Only entries where `import X` is satisfied by `pip install Y` with X != Y.
PIP_NAME_MAP = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "tensorflow_datasets": "tensorflow-datasets",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyopenssl",
    "magic": "python-magic",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "MySQLdb": "mysqlclient",
    "psycopg2": "psycopg2-binary",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxPython",
    "OpenGL": "PyOpenGL",
    "Image": "pillow",
    "ldap": "python-ldap",
    "memcache": "python-memcached",
    "atom": "atom",
    "git": "GitPython",
    "github": "PyGithub",
    "gitlab": "python-gitlab",
    "ldap3": "ldap3",
    "_yaml": "pyyaml",
    "win32com": "pywin32",
    "win32api": "pywin32",
    "pywintypes": "pywin32",
    "snappy": "python-snappy",
    "kafka": "kafka-python",
    "google.cloud": "google-cloud-storage",
    "google.protobuf": "protobuf",
    "twisted": "Twisted",
    "Levenshtein": "python-Levenshtein",
    "dns": "dnspython",
    "fastapi_users": "fastapi-users",
    "pydantic_settings": "pydantic-settings",
    "airflow": "apache-airflow",
    "homeassistant": "homeassistant",  # exists; identity (no rename)
    "oslo_config": "oslo.config",
    "oslo_log": "oslo.log",
    "oslo_utils": "oslo.utils",
    "oslo_serialization": "oslo.serialization",
    "oslo_messaging": "oslo.messaging",
    "oslo_concurrency": "oslo.concurrency",
    "oslo_context": "oslo.context",
    "oslo_db": "oslo.db",
    "oslo_i18n": "oslo.i18n",
    "oslo_middleware": "oslo.middleware",
    "oslo_policy": "oslo.policy",
    "oslo_rootwrap": "oslo.rootwrap",
    "oslo_service": "oslo.service",
    "oslo_versionedobjects": "oslo.versionedobjects",
    "google.cloud": "google-cloud-storage",
    "requests_mock": "requests-mock",
    "all_repos": "all-repos",
    "pre_commit": "pre-commit",
}

# Names that look like project-internal packages (almost certainly NOT on
# PyPI even if a HEAD check accidentally returns 200 — there's a long tail
# of squatted/generic package names). These map to "drop the task".
PROJECT_INTERNAL_HEADS = frozenset({
    # Plainly project-internal
    "tests", "test", "_tests", "test_framework", "testing",
    "conftest",
    "app", "myapp", "myproject", "mypackage", "mymodule",
    # Generic names — exist on PyPI as squatted/abandoned packages but
    # almost never the real intent (a test asking to `import helpers`
    # almost certainly means the project's local helpers module).
    "src", "lib", "core", "utils", "util",
    "model", "models", "helpers", "helper", "common",
    "config", "configs", "settings",
    "main", "module", "package",
    "scripts", "tools",
})


# ---------------------------------------------------------------------------
# PyPI cache
# ---------------------------------------------------------------------------

_PYPI_CACHE_PATH = Path("/tmp/pypi_cache.json")
_PYPI_CACHE: dict[str, bool] = {}


def _load_pypi_cache() -> None:
    global _PYPI_CACHE
    if _PYPI_CACHE_PATH.exists():
        try:
            _PYPI_CACHE = json.loads(_PYPI_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            _PYPI_CACHE = {}
    else:
        _PYPI_CACHE = {}


def _save_pypi_cache() -> None:
    try:
        _PYPI_CACHE_PATH.write_text(json.dumps(_PYPI_CACHE, sort_keys=True))
    except OSError:
        pass


def pypi_exists(pkg_name: str, no_check: bool = False, retries: int = 2) -> bool:
    """Return True iff `https://pypi.org/pypi/<pkg_name>/json` returns 200.

    Cached in /tmp/pypi_cache.json. With `no_check=True`, returns True for
    every name (skip network).
    """
    if no_check:
        return True
    if pkg_name in _PYPI_CACHE:
        return _PYPI_CACHE[pkg_name]

    url = f"https://pypi.org/pypi/{pkg_name}/json"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok = resp.status == 200
                _PYPI_CACHE[pkg_name] = ok
                return ok
        except urllib.error.HTTPError as e:
            if e.code == 404:
                _PYPI_CACHE[pkg_name] = False
                return False
            # 429 / 5xx → backoff & retry
            last_err = e
            time.sleep(1.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    # Network died on us — assume the package exists (fail-open).
    # If it really doesn't exist, the runtime `pip install` will fail and
    # pytest will then fail on the missing import (correct semantic).
    print(f"  warning: PyPI check for {pkg_name!r} failed after retries: {last_err}",
          file=sys.stderr)
    _PYPI_CACHE[pkg_name] = True
    return True


# ---------------------------------------------------------------------------
# Stdlib + import parsing
# ---------------------------------------------------------------------------

def stdlib_names() -> frozenset[str]:
    if hasattr(sys, "stdlib_module_names"):
        return frozenset(sys.stdlib_module_names)
    raise RuntimeError("Need Python 3.10+ for sys.stdlib_module_names")


def py_compile_check(test_path: Path) -> bool:
    try:
        py_compile.compile(str(test_path), doraise=True, quiet=1)
        return True
    except py_compile.PyCompileError:
        return False
    except SyntaxError:
        return False


def parse_top_level_imports(test_path: Path) -> set[str]:
    """Return the set of TOP-LEVEL package names imported by the file.

    For `import a.b.c` and `from a.b.c import d`, returns `{"a"}` (only
    the leftmost component, which determines the install).

    Skips relative imports.
    """
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                names.add(head)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative
            if node.module:
                head = node.module.split(".")[0]
                names.add(head)
    return names


# ---------------------------------------------------------------------------
# Per-import classification
# ---------------------------------------------------------------------------

def _looks_project_internal(name: str) -> bool:
    """Heuristic: an import head that's clearly NOT a published PyPI package.

    Drops on:
      - leading underscore (private-by-convention)
      - matches PROJECT_INTERNAL_HEADS exactly
    """
    if name.startswith("_") and name not in EXTRA_ALLOWED:
        return True
    if name in PROJECT_INTERNAL_HEADS:
        return True
    return False


def classify_import(
    name: str,
    stdlib: frozenset[str],
    no_pypi_check: bool,
) -> tuple[str, str | None]:
    """Classify a single top-level import name.

    Returns (verdict, pip_name) where verdict is one of:
        "skip"              — stdlib / allowlist (no install needed)
        "install"           — pip-installable; pip_name is the package
        "drop_internal"     — looks project-internal
        "drop_not_on_pypi"  — PyPI HEAD returned 404
    """
    if name in stdlib:
        return ("skip", None)
    if name in EXTRA_ALLOWED:
        return ("skip", None)
    if _looks_project_internal(name):
        return ("drop_internal", None)

    pip_name = PIP_NAME_MAP.get(name, name)
    if pypi_exists(pip_name, no_check=no_pypi_check):
        return ("install", pip_name)
    return ("drop_not_on_pypi", None)


# ---------------------------------------------------------------------------
# test.sh patching
# ---------------------------------------------------------------------------

PATCH_MARKER = "# --- laion v2 patch: auto-install test deps ---"
PATCH_END = "# --- end laion v2 patch ---"


def patch_test_sh(test_sh_path: Path, pip_packages: list[str]) -> bool:
    """Inject `pip install <pkg> || true` lines just before the pytest call.

    Returns True if the file was modified (or already had the marker for the
    same set of packages — idempotent).
    """
    text = test_sh_path.read_text(encoding="utf-8")

    # Idempotency: if the marker is present, strip the old block first so
    # we always re-emit a fresh block.
    if PATCH_MARKER in text:
        start = text.index(PATCH_MARKER)
        end_marker_idx = text.find(PATCH_END, start)
        if end_marker_idx == -1:
            # malformed prior patch — don't risk it
            return False
        end = end_marker_idx + len(PATCH_END)
        # also eat trailing newline
        if end < len(text) and text[end] == "\n":
            end += 1
        text = text[:start] + text[end:]

    if not pip_packages:
        # No installs needed; still strip any old patch block above and
        # rewrite the file to keep things clean.
        test_sh_path.write_text(text, encoding="utf-8")
        return True

    # Build the install block.
    install_lines = "\n".join(
        f'pip3 install --quiet "{pkg}" 2>/dev/null || true' for pkg in pip_packages
    )
    block = f"{PATCH_MARKER}\n{install_lines}\n{PATCH_END}\n"

    # Insert right BEFORE the line containing the first pytest invocation.
    lines = text.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("pytest ") or stripped.startswith("pytest\t") \
                or stripped.startswith("python -m pytest") \
                or stripped.startswith("python3 -m pytest"):
            insert_at = i
            break

    if insert_at is None:
        # No pytest line found — append before final exit, or at end.
        # Find the last non-empty line and insert before it.
        insert_at = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("exit"):
                insert_at = i
                break

    new_text = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
    test_sh_path.write_text(new_text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task_dir: Path,
    stdlib: frozenset[str],
    no_pypi_check: bool,
) -> tuple[str, list[str], list[str]]:
    """Return (verdict, pip_packages, drop_reasons).

    verdict:
        "keep"     — task is patchable; pip_packages is the dedup'd install set
        "syntax"   — py_compile failed
        "internal" — at least one import is project-internal
        "no_pypi"  — at least one import isn't on PyPI
        "error"    — missing files
    """
    test_path = task_dir / "tests" / "test_solution.py"
    test_sh_path = task_dir / "tests" / "test.sh"

    if not test_path.exists():
        return ("error", [], ["missing tests/test_solution.py"])
    if not test_sh_path.exists():
        return ("error", [], ["missing tests/test.sh"])

    if not py_compile_check(test_path):
        return ("syntax", [], ["py_compile failed"])

    imports = parse_top_level_imports(test_path)

    pip_packages: list[str] = []
    drop_internal: list[str] = []
    drop_no_pypi: list[str] = []

    for name in sorted(imports):
        verdict, pip_name = classify_import(name, stdlib, no_pypi_check)
        if verdict == "skip":
            continue
        if verdict == "install":
            assert pip_name is not None
            if pip_name not in pip_packages:
                pip_packages.append(pip_name)
        elif verdict == "drop_internal":
            drop_internal.append(name)
        elif verdict == "drop_not_on_pypi":
            drop_no_pypi.append(name)

    if drop_internal:
        return ("internal", [], drop_internal)
    if drop_no_pypi:
        return ("no_pypi", [], drop_no_pypi)

    return ("keep", pip_packages, [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-install patcher for softwareheritage-large tasks.",
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing swh-XXXX task folders")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen without modifying anything")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N tasks (sanity)")
    parser.add_argument("--no-pypi-check", action="store_true",
                        help="Skip the PyPI HEAD check (treat all unmapped names as known)")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of example tasks to print per bucket")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Not a directory: {args.root}")

    _load_pypi_cache()
    stdlib = stdlib_names()
    task_dirs = sorted(d for d in args.root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[:args.limit]

    counts = {"keep": 0, "syntax": 0, "internal": 0, "no_pypi": 0, "error": 0}
    examples = {k: [] for k in counts}
    install_freq: dict[str, int] = {}

    for i, td in enumerate(task_dirs):
        verdict, pip_packages, drop_reasons = evaluate_task(td, stdlib, args.no_pypi_check)
        counts[verdict] += 1

        if len(examples[verdict]) < args.examples:
            examples[verdict].append((td.name, pip_packages, drop_reasons))

        if verdict == "keep":
            for pkg in pip_packages:
                install_freq[pkg] = install_freq.get(pkg, 0) + 1
            if not args.dry_run:
                test_sh_path = td / "tests" / "test.sh"
                patch_test_sh(test_sh_path, pip_packages)
        else:
            if not args.dry_run:
                shutil.rmtree(td)

        # Save cache periodically so we don't lose state on Ctrl-C
        if (i + 1) % 200 == 0:
            _save_pypi_cache()

    _save_pypi_cache()

    total = sum(counts.values())
    action = "Would keep / drop" if args.dry_run else "Kept / dropped"
    print(f"\n=== {action} (total={total}) ===")
    print(f"  keep                  : {counts['keep']:>5} ({counts['keep']/total*100:.1f}%)")
    print(f"  drop (syntax)         : {counts['syntax']:>5} ({counts['syntax']/total*100:.1f}%)")
    print(f"  drop (internal import): {counts['internal']:>5} ({counts['internal']/total*100:.1f}%)")
    print(f"  drop (not on PyPI)    : {counts['no_pypi']:>5} ({counts['no_pypi']/total*100:.1f}%)")
    print(f"  drop (error)          : {counts['error']:>5} ({counts['error']/total*100:.1f}%)")

    print("\nTop 20 most-common installed packages:")
    for pkg, n in sorted(install_freq.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {n:>4}  {pkg}")

    print("\nExample KEPT tasks (id, packages):")
    for name, pkgs, _ in examples["keep"]:
        print(f"  {name:<10} {pkgs}")

    print("\nExample DROPPED — syntax:")
    for name, _, reasons in examples["syntax"]:
        print(f"  {name:<10} {reasons}")

    print("\nExample DROPPED — project-internal:")
    for name, _, reasons in examples["internal"]:
        print(f"  {name:<10} {reasons}")

    print("\nExample DROPPED — not on PyPI:")
    for name, _, reasons in examples["no_pypi"]:
        print(f"  {name:<10} {reasons}")


if __name__ == "__main__":
    main()
