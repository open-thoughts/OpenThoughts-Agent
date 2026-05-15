#!/usr/bin/env python3
"""
exp_rpt_stack-pytest-synthetic-gpt5nano patcher (v1 -> v2).

QC found that 100% of v1 traces fail before any test runs because the
verifier's `pip3 install --quiet pytest` line crashes on PEP 668
("error: externally-managed-environment") on Debian/Ubuntu containers
that mark the system Python as externally-managed.

Sample of the offending block (identical across all 10000 tasks):

    if ! command -v pytest &> /dev/null; then
        pip3 install --quiet pytest
    fi

Fix: replace the `pip [3] install ...` invocation with one that:

  1. Tries `pip install --break-system-packages X` first (works on
     PEP 668-enforced pip >= 23.0; ignores stderr).
  2. Falls back to a plain `pip install X` (older pips that don't
     understand --break-system-packages).

Idempotent via marker `# --- laion v2 patch: PEP 668 unblock ---`.
A `bash -n` syntax check is run on every patched file; tasks that
fail are dropped (their entire task dir removed) so the parquet
re-upload doesn't ship broken scripts.

CLI shape mirrors the existing v3 patcher (`--root`, `--dry-run`,
`--limit`, `--drop-log`).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Idempotency marker. If present, we skip patching (already v2).
V2_MARKER = "# --- laion v2 patch: PEP 668 unblock ---"

# Match `pip install ...`, `pip3 install ...`, or `python[3] -m pip install ...`.
# Captures:
#   indent   : leading whitespace
#   pip_cmd  : the full pip invocation up to "install"
#   args     : the rest of the line (packages + flags)
# The regex deliberately requires "install" so that lines like `pip --version`
# are not touched. We also require that --break-system-packages NOT already be
# present in args, so re-running the patcher is a no-op (belt+suspenders on top
# of the V2_MARKER guard).
PIP_INSTALL_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<pip_cmd>(?:pip3?|python3?[ \t]+-m[ \t]+pip))"
    r"[ \t]+install"
    r"(?P<args>(?:[ \t]+(?!--break-system-packages\b)[^\n]*)?)"
    r"$",
    flags=re.MULTILINE,
)


def _replace_pip_install(match: re.Match[str]) -> str:
    indent = match.group("indent")
    pip_cmd = match.group("pip_cmd")
    args = match.group("args") or ""
    # Strip a leading single space so we can format consistently.
    args_stripped = args.lstrip()

    # Compose the two-step install: try with --break-system-packages first
    # (suppress stderr so the older-pip "unrecognized option" message doesn't
    # confuse logs), then fall back to plain install.
    flagged = f"{pip_cmd} install --break-system-packages {args_stripped}".rstrip()
    fallback = f"{pip_cmd} install {args_stripped}".rstrip()
    return f"{indent}{V2_MARKER}\n{indent}{flagged} 2>/dev/null || {fallback}"


def patch_test_sh(text: str) -> tuple[str, bool, int]:
    """Patch pip-install lines. Returns (new_text, changed, n_pip_lines_replaced)."""
    if V2_MARKER in text:
        return text, False, 0

    new_text, n = PIP_INSTALL_RE.subn(_replace_pip_install, text)
    return new_text, n > 0, n


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

    test_paths = sorted(root.glob("*/tests/test.sh"))
    if not test_paths:
        print(f"No tests/test.sh files under {root}", file=sys.stderr)
        return 2

    if args.limit:
        test_paths = test_paths[: args.limit]

    n_total = len(test_paths)
    n_patched = 0
    n_already_patched = 0
    n_no_pip = 0
    n_dropped = 0
    n_kept = 0
    dropped_examples: list[tuple[str, str]] = []
    drop_log_lines: list[str] = []

    for i, p in enumerate(test_paths, 1):
        task_dir = p.parent.parent  # <root>/<task_id>/tests/test.sh -> <root>/<task_id>
        task_id = task_dir.name
        original = p.read_text()

        patched, changed, n_pip = patch_test_sh(original)
        if V2_MARKER in original:
            n_already_patched += 1
        elif n_pip == 0:
            n_no_pip += 1

        if changed and not args.dry_run:
            p.write_text(patched)

        # Syntax-check the (possibly) patched file. If broken, drop the task.
        # On dry-run we can only check the in-memory text by writing to a tmp
        # path; cheaper to skip the dry-run drop and just count.
        if args.dry_run:
            if changed:
                n_patched += 1
            n_kept += 1
        else:
            ok, stderr = syntax_check(p)
            if not ok:
                first_line = stderr.splitlines()[0] if stderr else "(no stderr)"
                drop_log_lines.append(f"{task_id}\t{first_line}")
                if len(dropped_examples) < 5:
                    dropped_examples.append((task_id, first_line))
                n_dropped += 1
                shutil.rmtree(task_dir, ignore_errors=True)
            else:
                if changed:
                    n_patched += 1
                n_kept += 1

        if i % 1000 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] kept={n_kept} dropped={n_dropped} "
                f"patched={n_patched} already_v2={n_already_patched} no_pip={n_no_pip}",
                flush=True,
            )

    pct_dropped = (100.0 * n_dropped / n_total) if n_total else 0.0
    print()
    print("=" * 60)
    print(f"Total tasks scanned:     {n_total}")
    print(f"Kept (bash -n passed):   {n_kept}")
    print(f"Dropped (syntax error):  {n_dropped} ({pct_dropped:.1f}%)")
    print(f"  v2 patch applied:      {n_patched}")
    print(f"  already had v2 patch:  {n_already_patched}")
    print(f"  no pip-install line:   {n_no_pip}")
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
