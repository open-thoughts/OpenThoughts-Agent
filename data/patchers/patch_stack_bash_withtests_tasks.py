#!/usr/bin/env python3
"""
exp_rpt_stack-bash-withtests v2 patcher.

QC: 10/10 sampled v1 traces failed with `VerifierRuntimeError: No reward
file found` because every per-task `tests/test.sh` aborts before writing
`reward.txt`. The dominant failure mode is that the script wraps the real
verifier in a `run_verifier` shell function which reads `$1`/`$2`/... but
the dispatch (`if run_verifier; then ...`) calls it with no args. The
function then prints a `Usage:` line and exits non-zero -- but
`/logs/verifier/reward.txt` was never written, so the harness can't even
register a 0 reward.

Examples (real Usage lines pulled from the corpus):

  Usage: /tests/test.sh {start|stop|restart|status}
  Usage: ./install-game.sh GAME_DIR_NAME [GAME_READABLE_NAME]
  USAGE: $0 <solve base parameters file> <inference options file>
  Usage: \\$0 [start|restart|graceful|graceful-stop|stop]
  usage: ciphertest <server:port> <server|pds>
  Usage: test.sh <pkg_ident>

Two unconditional fixes per task, plus one conditional fix:

  1. Reward.txt floor + EXIT trap. Inserted at the very top of the file
     (right after the shebang, before any `set -e`). Guarantees that even
     if every later step fails, `/logs/verifier/reward.txt` ends up as "0"
     instead of missing.

  2. Default positional args. If we can detect a `Usage:` / `usage:` /
     `USAGE:` line documenting the script's expected arguments, we extract
     placeholder values for each positional and inject `set -- "..." "..."`
     near the top of the script. This makes `$1`, `$2`, ... resolve to
     non-empty strings inside `run_verifier` so the script gets past its
     argument check. (The script may then fail on missing files or
     binaries -- that's fine, the EXIT trap will still leave reward.txt
     populated.)

  3. Idempotent via marker comment.

We do NOT attempt to fix:

  - Missing binaries (`module`, elasticsearch, hsh, ...): no mechanical fix
  - Missing source files (`functions.sh`, `parse_options.sh`): same
  - Wrong CWD: not applicable here (different bug pattern from v3 stack-bash)

The reward floor + trap handle these by leaving reward=0 on exit.

CLI: --root <dir> [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Idempotency marker.
PATCH_MARKER = "# --- laion v2 patch: stack-bash-withtests args + reward floor ---"
PATCH_END_MARKER = "# --- end laion v2 patch ---"

# Reward floor + EXIT trap. Unconditional, always inserted right after the
# shebang.
REWARD_FLOOR = f"""{PATCH_MARKER}
mkdir -p /logs/verifier 2>/dev/null || true
echo 0 > /logs/verifier/reward.txt 2>/dev/null || true
trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT
{PATCH_END_MARKER}
"""

# Match a usage-style help line. Anchored to "usage" (case-insensitive)
# followed by ":" (or "is" -- some scripts say "Usage is foo.sh ..."). We
# capture the rest of the line so we can extract positional argument
# tokens. The capture stops at the first newline or closing quote so we
# don't pick up trailing redirections / dollar-vars.
USAGE_LINE_RE = re.compile(
    r"""(?ix)
    (?:^|[\s"'#])                          # boundary (start, space, quote, or comment)
    (?:Usage|usage|USAGE)
    \s*(?:[:]|\s+is\s+)\s*                 # `:` separator or ` is `
    (?P<rest>[^\n"]*?)                     # doc text, no newlines, no quotes
    (?:["]|$)                              # stops at closing dquote or EOL
    """,
    flags=re.VERBOSE,
)

SHEBANG_RE = re.compile(r"^(#![^\n]*\n)")

# Match the `if run_verifier; then` dispatch. We rewrite it to pass
# positional args.
DISPATCH_RE = re.compile(
    r"^(?P<indent>[ \t]*)if[ \t]+run_verifier[ \t]*;[ \t]*then[ \t]*$",
    flags=re.MULTILINE,
)

# Tokens we treat as "noise" inside a usage line and skip when extracting
# positional defaults. The script name itself is always the first token.
USAGE_NOISE = {
    "$0", "\\$0", "$@", "\\$@",
    "[options]", "[OPTIONS]",
    "[flags]", "[FLAGS]",
    "[args]", "[ARGS]",
    "[--help]", "[-h]", "-h", "--help",
    "[options...]", "[arguments...]",
}


def _looks_like_script_name(token: str) -> bool:
    """First token in a usage line is usually the script name itself."""
    t = token.strip().strip("`").strip("'").strip('"')
    if not t:
        return True
    if t.endswith(".sh") or t.endswith(".bash"):
        return True
    if t.startswith("./") or t.startswith("/"):
        return True
    if t in ("test.sh", "$0", "\\$0"):
        return True
    # Common harness names.
    if re.fullmatch(r"[a-zA-Z_][\w.+-]*", t) and len(t) <= 30:
        # Could be the script name (e.g. `ciphertest`, `count`). We treat
        # any single bare identifier as the script name and skip it -- the
        # real positionals come after.
        return True
    return False


def _extract_default_for_token(token: str) -> str | None:
    """Given an arg-doc token like `<filename>` or `start|stop|restart`,
    return a sensible placeholder default, or None if we can't tell.

    We return None for "skip this token but keep going" cases (option
    flags like `-h`, `[--help]`, dash-prefixed tokens). All bracketed
    placeholders -- including `[optional]` -- yield a default value;
    in practice the scripts in this corpus read `$1` and crash on empty,
    so providing a default for an "optional" slot can only help.

    Strategy:
      - `start|stop|restart|status`         -> "start"
      - `{start|stop|restart}`              -> "start"
      - `[start|restart|stop]`              -> "start"   (first choice)
      - `<filename>` / `<srcroot>` / `name` -> "default"
      - `-r`, `-b n`, `[-r]`, `[--help]`    -> None (skip, opt flag)
    """
    t = token.strip()
    if not t:
        return None
    if t in USAGE_NOISE:
        return None

    # Strip surrounding bracket pair (any of <>, {}, []).
    inner = t
    if (
        (inner.startswith("<") and inner.endswith(">"))
        or (inner.startswith("{") and inner.endswith("}"))
        or (inner.startswith("[") and inner.endswith("]"))
    ):
        inner = inner[1:-1].strip()

    if not inner:
        return None

    # Skip option flags: `-r`, `--help`, `-b n` (the `n` is the value
    # placeholder for the flag, but we have no way to know it's required).
    # We treat any token whose stripped inner content starts with `-` as
    # an option flag and skip it.
    if inner.startswith("-"):
        return None

    # Choice list: pick the first option.
    if "|" in inner:
        first = inner.split("|", 1)[0].strip()
        first = re.sub(r"[<>{}\[\]]", "", first).strip()
        if first and not first.startswith("-"):
            return first
        return None

    # Plain placeholder. Use "default" as a safe non-empty string.
    cleaned = re.sub(r"[<>{}\[\]]", "", inner).strip()
    if not cleaned:
        return None
    return "default"


def parse_usage_args(usage_rest: str) -> list[str] | None:
    """Parse the doc-text portion of a usage line and return a list of
    positional defaults, or None if we can't make sense of it.

    Returns at least 1 default if successful; returns None if the line
    has no extractable positional placeholders (e.g. only `[options]`).
    """
    # Strip trailing redirection that survived the usage regex. We cannot
    # naively split on `|` because that's a legitimate delimiter inside
    # `{a|b|c}` choice lists; we only strip `&`, `;`, `>`, and `|` when
    # they appear OUTSIDE any bracket group.
    text = usage_rest.strip()

    def _strip_trailing_redir(s: str) -> str:
        depth_a = depth_c = depth_s = 0
        for i, ch in enumerate(s):
            if ch == "<": depth_a += 1
            elif ch == ">":
                if depth_a > 0:
                    depth_a -= 1
                else:
                    return s[:i].rstrip()
            elif ch == "{": depth_c += 1
            elif ch == "}": depth_c -= 1
            elif ch == "[": depth_s += 1
            elif ch == "]": depth_s -= 1
            elif ch in "&;" and depth_a == 0 and depth_c == 0 and depth_s == 0:
                return s[:i].rstrip()
        return s

    text = _strip_trailing_redir(text)
    text = text.strip().strip(":").strip()
    if not text:
        return None

    # Tokenize on whitespace. We also handle tokens like `{a|b|c}` and
    # `<foo bar>` (where the angle-bracket group should be a single
    # token); a simple whitespace split breaks those, so we do a small
    # bracket-aware split.
    tokens: list[str] = []
    buf = ""
    depth_angle = depth_curly = depth_square = 0
    for ch in text:
        if ch in "<{[":
            depth_angle += ch == "<"
            depth_curly += ch == "{"
            depth_square += ch == "["
            buf += ch
            continue
        if ch in ">}]":
            depth_angle -= ch == ">"
            depth_curly -= ch == "}"
            depth_square -= ch == "]"
            buf += ch
            continue
        if ch.isspace() and depth_angle == 0 and depth_curly == 0 and depth_square == 0:
            if buf:
                tokens.append(buf)
                buf = ""
            continue
        buf += ch
    if buf:
        tokens.append(buf)

    if not tokens:
        return None

    # Drop the first token if it looks like the script name.
    if _looks_like_script_name(tokens[0]):
        tokens = tokens[1:]

    if not tokens:
        return None

    defaults: list[str] = []
    for tok in tokens:
        d = _extract_default_for_token(tok)
        if d is None:
            # Skip this token (option flag, noise) and keep scanning --
            # later positionals may still parse. Bound the scan so an
            # adversarial usage string doesn't blow up the result.
            if len(defaults) >= 8:
                break
            continue
        defaults.append(d)
        if len(defaults) >= 8:
            break

    if not defaults:
        return None
    return defaults


def find_first_usage(text: str) -> tuple[str, list[str]] | None:
    """Search the script for the first parseable usage line. Returns
    (raw_match, parsed_defaults) or None.
    """
    for m in USAGE_LINE_RE.finditer(text):
        rest = m.group("rest")
        defaults = parse_usage_args(rest)
        if defaults:
            return rest, defaults
    return None


def patch_test_sh(text: str) -> tuple[str, bool, bool]:
    """Apply the reward floor (always) + arg injection (if usage line
    parseable). Returns (new_text, changed, args_injected).
    """
    if PATCH_MARKER in text:
        # Already patched.
        return text, False, False

    args_injected = False
    new_text = text

    # Step 1: rewrite the dispatch to pass positional args, if we can
    # detect a usage line.
    detected = find_first_usage(text)
    if detected is not None:
        _raw, defaults = detected
        # Build the `set -- "a" "b" ...` snippet. We use bash quoting.
        quoted = " ".join(f'"{d}"' for d in defaults)
        # Rewrite the dispatch line: pass the args to run_verifier.
        def _rewrite(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}# laion v2: inject default positional args parsed from\n"
                f"{indent}# the script's Usage:/usage: line so run_verifier doesn't\n"
                f"{indent}# abort on `[ -z \"$1\" ]` before reward.txt is written.\n"
                f"{indent}if run_verifier {quoted}; then"
            )
        new_text2, n_subs = DISPATCH_RE.subn(_rewrite, new_text, count=1)
        if n_subs > 0:
            new_text = new_text2
            args_injected = True

    # Step 2: prepend the reward floor + trap. Inserted right after the
    # shebang (or at the very top if no shebang).
    m = SHEBANG_RE.match(new_text)
    if m:
        new_text = m.group(1) + REWARD_FLOOR + new_text[m.end():]
    else:
        new_text = "#!/bin/bash\n" + REWARD_FLOOR + new_text

    return new_text, new_text != text, args_injected


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
    n_with_args = 0
    n_floor_only = 0
    n_already = 0
    n_dropped = 0
    drop_log_lines: list[str] = []
    dropped_examples: list[tuple[str, str]] = []

    import shutil

    for i, p in enumerate(test_paths, 1):
        task_dir = p.parent.parent
        task_id = task_dir.name
        original = p.read_text()

        if PATCH_MARKER in original:
            n_already += 1
            continue

        patched, changed, args_injected = patch_test_sh(original)
        if not changed:
            # Nothing applied (shouldn't happen since reward floor is
            # unconditional, but be defensive).
            continue

        # Write provisionally so we can syntax-check.
        if args.dry_run:
            # In dry-run mode, do bash -n on a temp file instead.
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tf:
                tf.write(patched)
                tmp_path = Path(tf.name)
            try:
                ok, stderr = syntax_check(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            p.write_text(patched)
            ok, stderr = syntax_check(p)

        if not ok:
            first_line = stderr.splitlines()[0] if stderr else "(no stderr)"
            drop_log_lines.append(f"{task_id}\t{first_line}")
            if len(dropped_examples) < 5:
                dropped_examples.append((task_id, first_line))
            n_dropped += 1
            if not args.dry_run:
                shutil.rmtree(task_dir, ignore_errors=True)
            if i % 1000 == 0 or i == n_total:
                print(
                    f"[{i}/{n_total}] with_args={n_with_args} "
                    f"floor_only={n_floor_only} dropped={n_dropped} "
                    f"already={n_already}",
                    flush=True,
                )
            continue

        if args_injected:
            n_with_args += 1
        else:
            n_floor_only += 1

        if i % 1000 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] with_args={n_with_args} "
                f"floor_only={n_floor_only} dropped={n_dropped} "
                f"already={n_already}",
                flush=True,
            )

    pct_dropped = (100.0 * n_dropped / n_total) if n_total else 0.0
    pct_with_args = (100.0 * n_with_args / n_total) if n_total else 0.0
    print()
    print("=" * 60)
    print(f"Total tasks scanned:           {n_total}")
    print(f"Patched with args injection:   {n_with_args} ({pct_with_args:.1f}%)")
    print(f"Patched floor+trap only:       {n_floor_only}")
    print(f"Already patched (skipped):     {n_already}")
    print(f"Dropped (bash -n failed):      {n_dropped} ({pct_dropped:.1f}%)")
    print(f"Dry run:                       {args.dry_run}")
    print("=" * 60)

    if dropped_examples:
        print("\nFirst dropped tasks (id, bash -n stderr):")
        for tid, msg in dropped_examples:
            print(f"  {tid}: {msg}")

    if args.drop_log and drop_log_lines:
        Path(args.drop_log).write_text("\n".join(drop_log_lines) + "\n")
        print(f"\nFull drop log written to: {args.drop_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
