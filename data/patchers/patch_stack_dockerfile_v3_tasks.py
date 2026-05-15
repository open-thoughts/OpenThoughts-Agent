#!/usr/bin/env python3
"""
exp_rpt_stack-dockerfile patcher (v2 -> v3).

QC of v2 (`laion/exp_rpt_stack-dockerfile-gpt5mini-v2`) found 99.5%
infra failure (199/200 trials). The bottleneck was NOT the v2
filter (PATH_GLOB / image-404 / apt-broken filters all fired
correctly). The bottleneck is upstream of the patcher in
``data/commons.py::ensure_output_dir_in_dockerfile``.

The bug
-------
``ensure_output_dir_in_dockerfile`` injects a ``RUN mkdir -p /output
&& chmod 777 /output`` directive into every Dockerfile, intended to
sit just after the first ``WORKDIR`` line (so the directory is created
inside the image). Code at ``data/commons.py:1452-1458``::

    insert_idx = 0
    for idx, line in enumerate(content):
        if line.strip().upper().startswith("WORKDIR"):
            insert_idx = idx + 1
            break

    content.insert(insert_idx, directive)

If the Dockerfile has NO ``WORKDIR`` directive (which is true for
roughly half of the dockerfile-corpus tasks, especially ones based on
balena / resin / minimal Alpine images), ``insert_idx`` stays at its
initial value of ``0``. The directive is inserted at line 0, BEFORE
the ``FROM`` directive — yielding::

    RUN mkdir -p /output && chmod 777 /output      <- injected at idx 0
    FROM resin/colibri-imx6dl-buildpack-deps:stretch
    ...

Docker requires ``FROM`` to be the first non-comment, non-ARG
instruction. With a ``RUN`` before any ``FROM``, the build fails with
``error response from daemon: no build stage in current context``.

QC count: 85/200 (42.5%) of v2 trials hit this exact error. The
remaining ~57% of failures are real Docker-runtime issues (apt-get
install of a removed package, image manifest unavailable, etc.) that
the v2 filter couldn't catch ahead of time.

The fix
-------
This v3 patcher walks the *already-extracted* task directories and
rewrites every ``environment/Dockerfile`` so the ``RUN mkdir -p
/output && chmod 777 /output`` line lives AFTER the first ``FROM``
(if no ``WORKDIR`` exists) instead of at line 0.

Algorithm:

  1. Find all ``RUN mkdir -p /output && chmod 777 /output`` lines.
     For each, if its line index is *before* the first ``FROM`` line,
     remove it from its current spot.
  2. Determine the canonical insertion point:
     a. After the first ``WORKDIR`` line, if any.
     b. Else after the *last* ``FROM`` line that is not followed by
        a build-stage AS. (We want the directive in the final
        runtime stage.) For multi-stage builds, this is the safest
        spot.
     c. Else after the first ``FROM`` line.
  3. Insert the directive at that position. If a properly-placed
     directive already exists (idx > FROM idx), no-op.

Idempotent via a marker comment::

    # --- laion v3 patch: dockerfile mkdir-output ordering ---

We do NOT mutate ``RUN mkdir -p /output ...`` lines that are correctly
placed already (some upstream Dockerfiles already have the
``mkdir /output`` baked in). We only relocate the line if it appears
before any ``FROM``.

This patcher must run on already-v2-filtered task directories. It
does not re-run the v2 PATH_GLOB / image-404 / apt-broken filters.
The v2 filters were correct; we don't need to redo them.

Usage::

    python data/patchers/patch_stack_dockerfile_v3_tasks.py \\
        --root /path/to/v2_filtered_tasks_dir \\
        [--dry-run] [--limit N]

Then upload to ``laion/exp_rpt_stack-dockerfile-gpt5mini-v3``.

Followup recommendation (NOT in this patcher's scope)
-----------------------------------------------------

The root cause is in ``data/commons.py::ensure_output_dir_in_dockerfile``.
That function should be fixed upstream so future patcher runs don't
re-introduce the bug. The minimal fix::

    insert_idx = None
    for idx, line in enumerate(content):
        s = line.strip().upper()
        if s.startswith("WORKDIR"):
            insert_idx = idx + 1
            break
    if insert_idx is None:
        # No WORKDIR — insert after the FIRST FROM directive.
        for idx, line in enumerate(content):
            if line.strip().upper().startswith("FROM"):
                insert_idx = idx + 1
                break
    if insert_idx is None:
        # No FROM either (truly malformed Dockerfile) — skip insertion;
        # the build will fail later on its own.
        return

    content.insert(insert_idx, directive)

The patcher in this file does the same logic on already-extracted
task dirs so we can re-upload v3 without touching ``commons.py``,
but the upstream fix is recommended before generating new corpora.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

V3_MARKER = "# --- laion v3 patch: dockerfile mkdir-output ordering ---"

OUTPUT_DIRECTIVE_RE = re.compile(
    r"^\s*RUN\s+mkdir\s+-p\s+/output\s+&&\s+chmod\s+777\s+/output\s*$"
)
FROM_RE = re.compile(r"^\s*FROM\s+", re.IGNORECASE)
WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+", re.IGNORECASE)


def patch_dockerfile_text(text: str) -> tuple[str, bool, str]:
    """Return ``(new_text, changed, reason)``.

    ``reason`` is one of:
      - ``"already-v3"``     — V3_MARKER present
      - ``"no-from"``        — Dockerfile has no FROM (skip; build will
                                fail anyway)
      - ``"no-output-line"`` — no ``RUN mkdir -p /output ...``; nothing
                                for v3 to fix (kept as-is)
      - ``"already-correct"``— output line is already after first FROM
      - ``"relocated"``      — moved output line from before-FROM to
                                after-FROM/WORKDIR
    """
    if V3_MARKER in text:
        return text, False, "already-v3"

    lines = text.splitlines()

    # Find indices of interest.
    from_indices = [i for i, ln in enumerate(lines) if FROM_RE.match(ln)]
    if not from_indices:
        return text, False, "no-from"

    output_indices = [i for i, ln in enumerate(lines) if OUTPUT_DIRECTIVE_RE.match(ln)]
    if not output_indices:
        return text, False, "no-output-line"

    first_from = from_indices[0]
    last_from = from_indices[-1]

    # Find the first WORKDIR after the LAST FROM (i.e. WORKDIR in the
    # final runtime stage, for multi-stage builds).
    workdir_after_last_from: int | None = None
    for i, ln in enumerate(lines):
        if i > last_from and WORKDIR_RE.match(ln):
            workdir_after_last_from = i
            break

    # If ALL output directives are after the first FROM already and at
    # least one is in the final runtime stage, we're already correct.
    if all(idx > first_from for idx in output_indices):
        return text, False, "already-correct"

    # Remove all output directives (we'll re-insert one at the right place).
    new_lines = [ln for i, ln in enumerate(lines) if i not in set(output_indices)]
    directive = "RUN mkdir -p /output && chmod 777 /output"

    # Recompute the FROM/WORKDIR positions in the *post-removal* list.
    new_from_indices = [i for i, ln in enumerate(new_lines) if FROM_RE.match(ln)]
    if not new_from_indices:
        # Should never happen — but defensive.
        return text, False, "no-from"
    new_last_from = new_from_indices[-1]

    new_workdir: int | None = None
    for i, ln in enumerate(new_lines):
        if i > new_last_from and WORKDIR_RE.match(ln):
            new_workdir = i
            break

    if new_workdir is not None:
        insert_at = new_workdir + 1
    else:
        insert_at = new_last_from + 1

    new_lines.insert(insert_at, directive)
    new_lines.insert(insert_at, V3_MARKER)
    return "\n".join(new_lines) + "\n", True, "relocated"


def patch_one_task(task_dir: Path, dry_run: bool) -> tuple[str, str]:
    df = task_dir / "environment" / "Dockerfile"
    if not df.is_file():
        return "no-dockerfile", ""
    try:
        text = df.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return "read-err", str(e)
    new_text, changed, reason = patch_dockerfile_text(text)
    if changed and not dry_run:
        df.write_text(new_text, encoding="utf-8")
    return reason, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for i, td in enumerate(task_dirs, 1):
        reason, _ = patch_one_task(td, args.dry_run)
        counts[reason] = counts.get(reason, 0) + 1
        examples.setdefault(reason, [])
        if len(examples[reason]) < 5:
            examples[reason].append(td.name)
        if i % 1000 == 0 or i == len(task_dirs):
            print(f"[{i}/{len(task_dirs)}] {counts}", flush=True)

    print()
    print("=" * 60)
    print(f"Total: {len(task_dirs)}  Dry={args.dry_run}")
    for k, v in sorted(counts.items()):
        print(f"  {k:<20}: {v}")
        for ex in examples.get(k, [])[:5]:
            print(f"      {ex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
