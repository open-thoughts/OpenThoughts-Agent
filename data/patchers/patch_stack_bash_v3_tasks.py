#!/usr/bin/env python3
"""
exp_rpt_stack-bash v3 patcher (cumulative on top of v2).

QC found that ~9/10 sampled v2 traces still failed because per-task
`tests/test.sh` files have bash-level bugs that the v2 patch did not
address:

  - Hard syntax errors (heredoc delimited by EOF, unbalanced quotes)
  - `cd: ./tests: No such file or directory` (wrong cwd)
  - `hsh: command not found` even though the agent built `hsh` at /app/hsh
    (the test ran `hsh` without `./` and `/app` was not on PATH)
  - `realpath: '': No such file`, `$1: ambiguous redirect`, etc.

Two fixes applied per task:

  1. **bash -n (syntax) filter** -- if `bash -n tests/test.sh` fails on
     the v2-patched script, the task is *unfixable mechanically*. We
     remove the entire task directory so it does not appear in v3's
     parquet.

  2. **cwd + PATH shim** injected immediately after the v2 preamble
     (after the `# --- end laion v2 patch ---` marker). This catches
     the `hsh: command not found` family by adding `/app` to PATH and
     making `/app` the cwd before any task-specific verifier code runs.

The v2 marker block is preserved verbatim -- v3 is strictly cumulative.

CLI shape mirrors the existing v2 patcher (`--root`, `--dry-run`,
`--limit`).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Marker line emitted by v2; we inject the v3 shim immediately after this.
V2_END_MARKER = "# --- end laion v2 patch ---"

# Marker line emitted by v3; used to detect double-patch.
V3_BEGIN_MARKER = "# --- laion v3 patch: cwd + PATH shim ---"
V3_END_MARKER = "# --- end laion v3 patch ---"

V3_SHIM = f"""{V3_BEGIN_MARKER}
# QC on v2 found per-task verifiers that fail with `cd: ./tests: No such
# file or directory` or `hsh: command not found` (agent built /app/hsh
# correctly but test invoked `hsh` without ./ and /app was off PATH).
# Force cwd to /app and prepend /app to PATH before any task code runs.
cd /app 2>/dev/null || true
export PATH="/app:${{PATH}}"
{V3_END_MARKER}
"""


def patch_test_sh(text: str) -> tuple[str, bool]:
    """Inject the v3 shim after the v2 end marker. Returns (new_text, changed)."""
    if V3_BEGIN_MARKER in text:
        # Already patched.
        return text, False

    if V2_END_MARKER not in text:
        # No v2 preamble -- this script never went through v2. Don't touch
        # it; the v2 patcher should have handled it. We treat this as
        # unchanged so the caller can decide what to do.
        return text, False

    # Insert the v3 shim on the line after the v2 end marker.
    marker_idx = text.index(V2_END_MARKER)
    line_end = text.find("\n", marker_idx)
    if line_end == -1:
        # File ends right after the marker -- still patchable, just
        # append.
        new_text = text + "\n" + V3_SHIM
    else:
        new_text = text[: line_end + 1] + V3_SHIM + text[line_end + 1 :]

    return new_text, True


def syntax_check(path: Path) -> tuple[bool, str]:
    """Return (ok, stderr). ok=True iff `bash -n path` exits 0."""
    try:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "bash -n timed out"
    except FileNotFoundError:
        return False, "bash not found"
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr or result.stdout).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N task dirs (0 = all)",
    )
    p.add_argument(
        "--drop-log",
        type=str,
        default=None,
        help="Optional path to write dropped-task report (TSV: task_id\\tstderr)",
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    # Sort by task dir for deterministic ordering.
    test_paths = sorted(root.glob("*/tests/test.sh"))
    if not test_paths:
        print(f"No tests/test.sh files under {root}", file=sys.stderr)
        return 2

    if args.limit:
        test_paths = test_paths[: args.limit]

    n_total = len(test_paths)
    n_patched = 0
    n_already_patched = 0
    n_no_v2 = 0
    n_dropped = 0
    n_kept = 0
    dropped_examples: list[tuple[str, str]] = []
    drop_log_lines: list[str] = []

    for i, p in enumerate(test_paths, 1):
        task_dir = p.parent.parent  # <root>/<task_id>/tests/test.sh -> <root>/<task_id>
        task_id = task_dir.name
        original = p.read_text()

        # Step 1: syntax check the v2-patched script. If it fails, drop the task.
        ok, stderr = syntax_check(p)
        if not ok:
            # Trim stderr to one-line summary.
            first_line = stderr.splitlines()[0] if stderr else "(no stderr)"
            drop_log_lines.append(f"{task_id}\t{first_line}")
            if len(dropped_examples) < 5:
                dropped_examples.append((task_id, first_line))
            n_dropped += 1
            if not args.dry_run:
                shutil.rmtree(task_dir, ignore_errors=True)
            if i % 500 == 0 or i == n_total:
                print(
                    f"[{i}/{n_total}] kept={n_kept} dropped={n_dropped} "
                    f"patched={n_patched} no_v2={n_no_v2}",
                    flush=True,
                )
            continue

        # Step 2: inject the v3 shim.
        patched, changed = patch_test_sh(original)
        if V2_END_MARKER not in original:
            n_no_v2 += 1
        if changed:
            n_patched += 1
            if not args.dry_run:
                p.write_text(patched)
        elif V3_BEGIN_MARKER in original:
            n_already_patched += 1
        n_kept += 1

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] kept={n_kept} dropped={n_dropped} "
                f"patched={n_patched} no_v2={n_no_v2}",
                flush=True,
            )

    pct_dropped = (100.0 * n_dropped / n_total) if n_total else 0.0
    print()
    print("=" * 60)
    print(f"Total tasks scanned:     {n_total}")
    print(f"Kept (bash -n passed):   {n_kept}")
    print(f"Dropped (syntax error):  {n_dropped} ({pct_dropped:.1f}%)")
    print(f"  v3 shim injected:      {n_patched}")
    print(f"  already had v3 shim:   {n_already_patched}")
    print(f"  no v2 preamble found:  {n_no_v2}")
    print(f"Dry run:                 {args.dry_run}")
    print("=" * 60)

    if dropped_examples:
        print("\nFirst dropped tasks (id, bash -n stderr):")
        for tid, msg in dropped_examples:
            print(f"  {tid}: {msg}")

    if args.drop_log and drop_log_lines:
        Path(args.drop_log).write_text("\n".join(drop_log_lines) + "\n")
        print(f"\nFull drop log written to: {args.drop_log}")

    if pct_dropped > 50.0:
        print(
            f"\nWARNING: dropped {pct_dropped:.1f}% of tasks -- "
            "syntax check may be too aggressive. Review before uploading.",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
