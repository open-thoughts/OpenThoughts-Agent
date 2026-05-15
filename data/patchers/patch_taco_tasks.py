#!/usr/bin/env python3
"""
Patch every instruction.md under a Harbor `exp_rpt_taco` task tree so the
agent knows the verifier expects the implementation at ``/app/solution.py``.

Same bug and fix as ``patch_codenet_python_tasks.py``: the test script in
``tests/test.sh`` runs ``python3 /app/solution.py < input.txt`` but the
prompt only says "implement in /app/" with no filename. Without the
explicit path, models write to arbitrary names and reward is always 0.

The fix
-------
Append a single line to `instruction.md`:

    Important: place your implementation at `/app/solution.py` so the
    verifier can run it.

Idempotent — re-running on already patched dirs is a no-op.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCH_MARKER = "Important: place your implementation at"
SUFFIX = (
    "\n\n"
    "Important: place your implementation at `/app/solution.py` so the "
    "verifier can run it.\n"
)


def patch_instruction(text: str) -> tuple[str, bool]:
    """Return (patched_text, changed). Idempotent."""
    if PATCH_MARKER in text:
        return text, False
    base = text.rstrip("\n")
    return base + SUFFIX, True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Tasks dir (extracted parquet)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Patch at most N tasks (0 = all)")
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

    for i, td in enumerate(task_dirs, 1):
        instr_path = td / "instruction.md"
        if not instr_path.is_file():
            n_skipped += 1
            continue

        instr_text = instr_path.read_text()
        patched, changed = patch_instruction(instr_text)
        if not changed:
            n_already += 1
            continue

        n_changed += 1
        if not args.dry_run:
            instr_path.write_text(patched)

        if i % 1000 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} already={n_already} "
                f"skipped={n_skipped}",
                flush=True,
            )

    print(
        f"Done. changed={n_changed}/{n_total}, already_patched={n_already}, "
        f"skipped={n_skipped}, dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
