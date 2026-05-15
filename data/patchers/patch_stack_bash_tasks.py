#!/usr/bin/env python3
"""
exp_rpt_stack-bash patcher.

Bug: every task wraps the real test in a shell function `run_verifier()` and
then writes reward based on its exit code:

    if run_verifier; then
        echo "1" > /logs/verifier/reward.txt
        echo "Tests passed!"
        ...

The function body has no `set -e`, so missing tools (sudo: command not found,
dart: command not found, ...) only print errors -- the function still returns
the exit code of its LAST command, which is often 0. Result: false-positive
rewards on broken environments.

Even adding `set -e` inside the function is not enough on its own: bash
suppresses `set -e` whenever a function is invoked from a tested context like
`if run_verifier; then ...` (POSIX errexit propagation rule). So we must do
TWO things:

Fix:
  1. Inject `set -eo pipefail` as the first statement inside `run_verifier() {`
     so any failed command kills the function.
  2. Rewrite the `if run_verifier; then ... else ... fi` dispatch into
     standalone calls so that `set -e` is NOT suppressed:

         run_verifier
         _rv_rc=$?
         if [ "$_rv_rc" -eq 0 ]; then
             ...
         else
             ...
         fi

  3. Add the standard env-shim + reward-floor preamble so reward.txt always
     exists and a missing /testbed/conda doesn't crash the script before the
     trap can write reward=0.

We intentionally do NOT stub sudo or other missing tools: a task that needs
sudo to verify the agent's solution was never actually verifiable, so failing
loudly is the correct behavior (no new false positives, no new false negatives).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PREAMBLE = r"""# --- laion v2 verifier patch: env shims + reward floor ---
mkdir -p /logs/verifier 2>/dev/null || true
echo 0 > /logs/verifier/reward.txt 2>/dev/null || true

# Stub /testbed -> /app for any task that references the legacy path
if [ ! -e /testbed ]; then
  ln -s /app /testbed 2>/dev/null || mkdir -p /testbed
fi

# python alias for scripts that hard-code `python`
if [ ! -e /usr/local/bin/python ] && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python 2>/dev/null || true
fi

# Always leave reward.txt populated even if the script aborts mid-way
trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT
# --- end laion v2 patch ---
"""

SHEBANG_RE = re.compile(r"^(#![^\n]*\n)")

# Match the opening of `run_verifier()` and capture surrounding indent so we
# can inject a `set -eo pipefail` line as the first statement of the body.
RUN_VERIFIER_OPEN_RE = re.compile(
    r"^(?P<indent>\s*)run_verifier\s*\(\)\s*\{\s*$",
    flags=re.MULTILINE,
)

# Marker emitted by the injected line; used to detect double-patch.
INJECT_MARKER = "set -eo pipefail  # laion v2: fail-fast inside run_verifier"

# Match the dispatch block:
#   if run_verifier; then
#       echo "1" > /logs/verifier/reward.txt
#       echo "Tests passed!"
#       exit 0
#   else
#       echo "0" > /logs/verifier/reward.txt
#       echo "Tests failed!"
#       exit 1
#   fi
#
# We replace the leading `if run_verifier; then` with explicit invocation so
# bash's POSIX rule that suppresses `set -e` inside a function called from an
# `if` test no longer applies.
DISPATCH_RE = re.compile(
    r"^(?P<indent>[ \t]*)if[ \t]+run_verifier[ \t]*;[ \t]*then[ \t]*$",
    flags=re.MULTILINE,
)

DISPATCH_REPLACEMENT = (
    "{indent}# laion v2: bash suppresses `set -e` inside a function called\n"
    "{indent}# from a tested context (`if foo;`, `foo ||`, `! foo`, etc.).\n"
    "{indent}# Calling in a bare subshell breaks that propagation, so the\n"
    "{indent}# `set -eo pipefail` injected at the top of run_verifier\n"
    "{indent}# actually fires on missing commands / failed pipelines.\n"
    "{indent}( run_verifier )\n"
    "{indent}_rv_rc=$?\n"
    "{indent}# Additionally require the agent to have written something to\n"
    "{indent}# /app: many tasks short-circuit run_verifier with `exit 0`\n"
    "{indent}# when invoked with no args, so a 0 exit with empty /app is a\n"
    "{indent}# false positive.\n"
    "{indent}if [ -d /app ] && [ -z \"$(ls -A /app 2>/dev/null)\" ]; then\n"
    "{indent}    _rv_rc=1\n"
    "{indent}fi\n"
    "{indent}if [ \"$_rv_rc\" -eq 0 ]; then"
)


def _inject_set_e(match: re.Match[str]) -> str:
    indent = match.group("indent")
    body_indent = indent + "    "
    return f"{match.group(0)}\n{body_indent}{INJECT_MARKER}"


def _rewrite_dispatch(match: re.Match[str]) -> str:
    return DISPATCH_REPLACEMENT.format(indent=match.group("indent"))


def patch_test_sh(text: str) -> tuple[str, bool]:
    original = text

    if INJECT_MARKER not in text:
        text = RUN_VERIFIER_OPEN_RE.sub(_inject_set_e, text, count=1)

    if "_rv_rc=" not in text:
        text = DISPATCH_RE.sub(_rewrite_dispatch, text, count=1)

    m = SHEBANG_RE.match(text)
    if m:
        text = m.group(1) + PREAMBLE + text[m.end():]
    else:
        text = "#!/bin/bash\n" + PREAMBLE + text

    return text, text != original


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

    test_paths = sorted(root.glob("*/tests/test.sh"))
    if not test_paths:
        print(f"No tests/test.sh files under {root}", file=sys.stderr)
        return 2

    if args.limit:
        test_paths = test_paths[: args.limit]

    n_changed = 0
    n_no_runv = 0
    n_total = len(test_paths)
    for i, p in enumerate(test_paths, 1):
        original = p.read_text()
        if "run_verifier()" not in original:
            n_no_runv += 1
        patched, changed = patch_test_sh(original)
        if changed:
            n_changed += 1
            if not args.dry_run:
                p.write_text(patched)
        if i % 1000 == 0 or i == n_total:
            print(f"[{i}/{n_total}] patched={n_changed} no_runv={n_no_runv}", flush=True)

    print(f"Done. {n_changed}/{n_total} files modified (dry_run={args.dry_run}, "
          f"tasks_without_run_verifier={n_no_runv}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
