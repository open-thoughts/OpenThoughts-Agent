#!/usr/bin/env python3
"""
Patch DCAgent/exp_rpt_stack-selfdoc-gpt5mini with the same auto-install
strategy that yielded ~47% on softwareheritage-large.

Reuses the classification + injection logic from
``patch_softwareheritage_tasks`` verbatim — same task layout
(``tests/test_solution.py`` + ``tests/test.sh``), same Dockerfile bones,
same fix strategy:

  1. ``py_compile`` the test file → drop on syntax fail (catches things
     like Python 2 ``except Exception, e:`` and removed stdlib modules).
  2. AST-parse the test file's top-level imports.
  3. Drop project-internal imports (relative imports, ``tests``/``app``/...,
     ``_``-prefixed).
  4. For each remaining non-stdlib import: classify against the PyPI-name
     mapping table; HEAD-check unknown names against PyPI (cached at
     ``/tmp/pypi_cache.json``); 404 → DROP task.
  5. For ``keep`` tasks, inject ``pip3 install --quiet "<pkg>" || true``
     lines into ``tests/test.sh`` immediately before the ``pytest`` line,
     bracketed by an idempotency marker.

Usage:
    python -m data.patchers.patch_stack_selfdoc_tasks \\
        --root /path/to/exp_rpt_stack-selfdoc-gpt5mini \\
        [--dry-run] [--limit N] [--no-pypi-check]

This is functionally identical to ``patch_softwareheritage_tasks.py`` —
the file exists separately so the dataset/pipeline mapping stays explicit
in the repo.
"""

from __future__ import annotations

# Re-export everything from the softwareheritage patcher and run its main.
# That patcher's logic is dataset-agnostic: it walks the per-task layout
# (`tests/test_solution.py` + `tests/test.sh`), which is identical here.

from data.patchers.patch_softwareheritage_tasks import (  # noqa: F401
    EXTRA_ALLOWED,
    PIP_NAME_MAP,
    PROJECT_INTERNAL_HEADS,
    PATCH_MARKER,
    PATCH_END,
    classify_import,
    evaluate_task,
    main,
    parse_top_level_imports,
    patch_test_sh,
    py_compile_check,
    pypi_exists,
    stdlib_names,
)


if __name__ == "__main__":
    main()
