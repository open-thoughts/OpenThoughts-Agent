#!/usr/bin/env python3
"""
Patch SWE-Gym Harbor task tree so:

  (1) The repo is cloned + checked out to base_commit during Docker build.
      Currently the Dockerfile sets WORKDIR /testbed but never clones the repo,
      and test.sh hard-codes `cd "$REPO_DIR"` (/testbed/repo). 100% of trials
      fail with `cd: /testbed/repo: No such file or directory`.

  (2) test.sh has a defense-in-depth clone-if-missing block (mirrors solve.sh).

  (3) The pytest exit-code-4-means-"skip" false-positive is closed: every
      test must actually be collected and pass. swegym-0941 case: ModuleNotFoundError
      for tlz caused all tests to fail collection; the original script swallowed
      that as "Test not found, skipping" → 0 failures → reward=1. The fix is a
      final accounting check that compares the number of "Test passed:" log lines
      against the expected count.

  (4) reward.txt is initialized to 0 at the very start so a script that aborts
      before reaching its on_error trap (e.g. SIGKILL, syntax error, OOM) still
      surfaces as fail rather than as "VerifierRuntimeError: No reward file found".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PATCH_MARKER = "# --- laion v2 patch (swegym): repo clone + reward floor + strict pass check ---"


def patch_dockerfile(text: str, repo: str, base_commit: str) -> tuple[str, bool]:
    """Append a `git clone + git checkout` RUN to the Dockerfile."""
    if PATCH_MARKER in text:
        return text, False

    clone_block = (
        f"\n{PATCH_MARKER}\n"
        f"RUN git clone https://github.com/{repo}.git /testbed/repo \\\n"
        f"    && cd /testbed/repo \\\n"
        f"    && git checkout {base_commit}\n"
    )
    return text.rstrip() + clone_block, True


# --- test.sh transforms ---

# Match the original "cd $REPO_DIR\n    ensure_dependencies" so we can inject
# a defense-in-depth clone block right before the cd. Original line uses the
# unusual 4-space indent that's preserved throughout swegym's test.sh files.
TEST_CD_RE = re.compile(
    r'(?P<indent>[ \t]*)cd "\$REPO_DIR"\n(?P<after>\s*ensure_dependencies)',
)

# Match the lenient "exit 4 = skip" branch so we can replace it with a fail.
# Original block (with 16-space outer indent):
#                 if [ $ec -eq 4 ]; then
#                     log "WARNING: Test not found ..."
#                 else
#                     log "Test failed with exit code $ec: $target"
#                     return 1
#                 fi
# We capture the indent of the `if` and re-emit `log "Test failed..."` +
# `return 1` at that same indent (replacing the entire if/else/fi).
EXIT4_RE = re.compile(
    r'^(?P<base>[ \t]*)if \[ \$ec -eq 4 \]; then[^\n]*\n'
    r'[ \t]*log "WARNING:[^"]*?"[^\n]*\n'
    r'[ \t]*else[^\n]*\n'
    r'[ \t]*(?P<fail_log>log "Test failed[^"]*?")[^\n]*\n'
    r'[ \t]*(?P<fail_return>return 1)[^\n]*\n'
    r'[ \t]*fi',
    flags=re.MULTILINE,
)


def _flatten_exit4(m: re.Match[str]) -> str:
    base = m.group("base")
    return f"{base}{m.group('fail_log')}\n{base}{m.group('fail_return')}"

# Match the final reward=1 / "All configured tests succeeded" tail.
FINAL_REWARD_RE = re.compile(
    r'(?P<indent>[ \t]*)echo 1 > "\$REWARD_FILE"\n'
    r'\s*log "All configured tests succeeded"\s*$',
    flags=re.MULTILINE | re.DOTALL,
)


def patch_test_sh(text: str, repo: str, base_commit: str) -> tuple[str, bool]:
    if PATCH_MARKER in text:
        return text, False

    original = text

    # 1. Floor reward.txt to 0 the moment the script starts, before any
    #    command can fail and bypass the existing on_error trap.
    floor_block = (
        f'\n{PATCH_MARKER}\n'
        f'mkdir -p /logs/verifier 2>/dev/null || true\n'
        f'[ -f /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt 2>/dev/null || true\n'
        f'trap \'[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt\' EXIT\n'
    )

    # Insert floor right after the first shebang line (preserve it).
    shebang_re = re.compile(r"^([ \t]*)(#![^\n]*\n)")
    m = shebang_re.match(text)
    if m:
        text = m.group(0) + floor_block + text[m.end():]
    else:
        text = "#!/usr/bin/env bash\n" + floor_block + text

    # 2. Inject defense-in-depth clone block before `cd "$REPO_DIR"`.
    indent = "    "  # swegym test.sh uses 4-space indent throughout the body

    def _clone_block_for(repo: str, commit: str) -> str:
        return (
            f'{indent}# defense-in-depth: ensure repo is cloned (Dockerfile also clones at build)\n'
            f'{indent}if [ ! -d "$REPO_DIR" ]; then\n'
            f'{indent}    cd /testbed\n'
            f'{indent}    git clone "https://github.com/{repo}.git" repo\n'
            f'{indent}fi\n'
            f'{indent}cd "$REPO_DIR"\n'
            f'{indent}git checkout "{commit}" 2>/dev/null || true\n'
        )

    text, n = TEST_CD_RE.subn(
        lambda m: _clone_block_for(repo, base_commit) + m.group("after"),
        text,
        count=1,
    )
    if n != 1:
        # Fall back to whatever indent the file actually used.
        m2 = re.search(r'^(?P<indent>[ \t]*)cd "\$REPO_DIR"', text, flags=re.MULTILINE)
        if m2:
            actual_indent = m2.group("indent")
            new_text = re.sub(
                r'^(?P<indent>[ \t]*)cd "\$REPO_DIR"\n',
                lambda m: _clone_block_for(repo, base_commit).replace(indent, actual_indent),
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if new_text != text:
                text = new_text
                n = 1

    if n != 1:
        # Couldn't find the cd; bail without claiming we patched.
        return original, False

    # 3. Replace the lenient exit-4-skip block with the fail body.
    text = EXIT4_RE.sub(_flatten_exit4, text)

    # 4. Replace the final reward=1 / "All configured tests succeeded" tail
    #    with an accounting check.
    accounting = (
        '{indent}# laion v2: count actual passes; require >= expected before reward=1\n'
        '{indent}_expected_count=$(( ${{#PASS_TESTS[@]}} + ${{#FAIL_TESTS[@]}} ))\n'
        '{indent}_actual_passes=$(grep -c "^\\[swegym\\] Test passed:" "$LOG_FILE" 2>/dev/null || echo 0)\n'
        '{indent}if [ "$_actual_passes" -ge "$_expected_count" ] && [ "$_expected_count" -gt 0 ]; then\n'
        '{indent}    echo 1 > "$REWARD_FILE"\n'
        '{indent}    log "All $_expected_count configured tests passed"\n'
        '{indent}else\n'
        '{indent}    echo 0 > "$REWARD_FILE"\n'
        '{indent}    log "Only $_actual_passes/$_expected_count configured tests passed; marking failure"\n'
        '{indent}    exit 1\n'
        '{indent}fi'
    )
    indent_for_accounting = "    "
    text = FINAL_REWARD_RE.sub(
        accounting.format(indent=indent_for_accounting),
        text,
    )

    return text, text != original


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_test_changed = 0
    n_dockerfile_changed = 0
    n_skipped = 0

    for i, td in enumerate(task_dirs, 1):
        meta_path = td / "metadata.json"
        if not meta_path.is_file():
            n_skipped += 1
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            n_skipped += 1
            continue

        repo = meta.get("repo")
        base_commit = meta.get("base_commit")
        if not repo or not base_commit:
            n_skipped += 1
            continue

        # Dockerfile
        df_path = td / "environment" / "Dockerfile"
        if df_path.is_file():
            df = df_path.read_text()
            new_df, df_changed = patch_dockerfile(df, repo, base_commit)
            if df_changed:
                n_dockerfile_changed += 1
                if not args.dry_run:
                    df_path.write_text(new_df)

        # test.sh
        ts_path = td / "tests" / "test.sh"
        if ts_path.is_file():
            ts = ts_path.read_text()
            new_ts, ts_changed = patch_test_sh(ts, repo, base_commit)
            if ts_changed:
                n_test_changed += 1
                if not args.dry_run:
                    ts_path.write_text(new_ts)

        if i % 200 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] dockerfile_patched={n_dockerfile_changed} "
                f"test_sh_patched={n_test_changed} skipped={n_skipped}",
                flush=True,
            )

    print(
        f"Done. dockerfile={n_dockerfile_changed}/{n_total}, "
        f"test_sh={n_test_changed}/{n_total}, skipped={n_skipped}, "
        f"dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
