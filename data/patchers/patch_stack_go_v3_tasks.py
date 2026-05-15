#!/usr/bin/env python3
"""
exp_rpt_stack-go v3 patcher (cumulative on top of v2).

QC on a 10/10 sample of v2 traces found two failure modes:

  (A) **Missing Go modules** (6/10): The agent writes `*.go` files that import
      packages like `k8s.io/apimachinery`, `github.com/docker/docker`,
      `github.com/alicebob/miniredis`, `gopkg.in/square/go-jose.v2`, etc.
      The verifier's `go mod init solution` creates an empty go.mod that does
      NOT pull those imports, so `go test` fails with
      `package <X> is not in std`. Fix: inject `go mod tidy` (with `go get
      ./...` as a belt-and-suspenders fallback) AFTER the agent's solution and
      the test file are both in /app, but BEFORE `go test`.

  (B) **Package-name mismatch** (3/10): `tests/solution_test.go` declares e.g.
      `package operation`, but the agent's `solution.go` uses `package app` (a
      reasonable default given no spec). `go test` then refuses with
      `found packages operation (test) and app (impl) in /app`. The agent has
      no way to know the required name from the prompt, so we extract it from
      the test file's `package <X>` directive and append a hint to
      `instruction.md`. If the test file uses an external test package
      (`package foo_test`), the implementation must be `package foo` -- we
      strip the `_test` suffix in the hint.

This patcher is **cumulative on v2's Dockerfile shim** (util-linux-misc for
tmux/script): we ONLY touch `tests/test.sh` and `instruction.md`; the v2
Dockerfile patch is left in place untouched.

Both transformations are idempotent (marker-guarded). If the test file has no
parseable `package` directive (rare -- 39/10000 in v2 sample), we still apply
the test.sh patch but skip the instruction.md hint.

CLI shape mirrors other v3 patchers in this directory (`--root`, `--dry-run`,
`--limit`).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# (A) test.sh injection
# --------------------------------------------------------------------------

TESTSH_MARKER = "# --- laion v3 patch: go mod tidy injection ---"

# Block injected just before the `go test ...` invocation. We attempt
# `go mod tidy` first (the right tool); if for any reason that fails, we fall
# back to `go get ./...` which is more forgiving (downloads anything imported
# without complaining about unused entries). Both are wrapped in `|| true`
# defensively so a network blip doesn't abort the verifier before `go test`
# runs (the test will still fail meaningfully if deps weren't fetched).
TESTSH_INJECTION = f"""{TESTSH_MARKER}
# QC on v2 found ~60% of failures were `package <X> is not in std` because the
# agent's solution imports third-party packages that `go mod init solution`
# does not fetch. Run `go mod tidy` to resolve+download all imports
# transitively (test file + agent solution), with `go get ./...` as a
# fallback for files that confuse tidy's import graph.
go mod tidy 2>/dev/null || go get ./... 2>/dev/null || true
# --- end laion v3 patch ---
"""

# We anchor on the `go test` line. Single canonical form across all 10000 v2
# tasks (md5sum of test.sh: 62e70a5f766d02f0241dcc767191563c):
#   go test -v ./... 2>&1 | tee /logs/verifier/test_output.txt
# We tolerate variation by matching any line whose first non-whitespace token
# is `go` followed by `test`.
GO_TEST_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)go[ \t]+test(?P<rest>[^\n]*)$",
    flags=re.MULTILINE,
)


def patch_test_sh(text: str) -> tuple[str, bool, str]:
    """
    Inject `go mod tidy` before the `go test` invocation.
    Returns (new_text, changed, reason).
    """
    if TESTSH_MARKER in text:
        return text, False, "already-patched"

    m = GO_TEST_LINE_RE.search(text)
    if not m:
        return text, False, "no-go-test-line"

    indent = m.group("indent")
    # Indent the injection block to match the go test line so syntax stays
    # consistent inside any if-blocks. Each inner line gets `indent` prepended.
    indented_block = "".join(
        f"{indent}{line}\n" if line else "\n"
        for line in TESTSH_INJECTION.split("\n")
    )
    # Strip a trailing extra newline from the join (TESTSH_INJECTION ends with \n).
    if indented_block.endswith("\n\n"):
        indented_block = indented_block[:-1]

    new_text = text[: m.start()] + indented_block + text[m.start():]
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, True, "injected"


# --------------------------------------------------------------------------
# (B) instruction.md package-name hint
# --------------------------------------------------------------------------

INSTRUCTION_MARKER_PREFIX = "Important: declare your solution in `package "

PKG_LINE_RE = re.compile(r"^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")


def parse_package_name(test_file_text: str) -> str | None:
    """
    Extract the `package <X>` directive from a Go test file, skipping
    leading blank lines, line comments (`//`), block comments (`/* ... */`),
    and `// +build ...` / `//go:build` constraint lines. The first non-comment
    non-blank line in a Go file MUST be the package directive (per Go spec),
    so this is robust.
    """
    in_block_comment = False
    for line in test_file_text.split("\n"):
        s = line.strip()
        if in_block_comment:
            if "*/" in s:
                in_block_comment = False
            continue
        if s.startswith("/*"):
            # Single-line /* ... */ stays open if no closing on same line.
            if "*/" not in s[2:]:
                in_block_comment = True
            continue
        if not s or s.startswith("//"):
            continue
        m = PKG_LINE_RE.match(s)
        if m:
            return m.group(1)
        # First non-comment line is not a package directive -> something odd.
        return None
    return None


def implementation_pkg_for(test_pkg: str) -> str:
    """
    If the test file uses an external test package (`package foo_test`),
    the implementation file must be `package foo` (Go spec). For internal
    test packages (`package foo`), the implementation must also be `foo`.
    """
    if test_pkg.endswith("_test") and len(test_pkg) > len("_test"):
        return test_pkg[: -len("_test")]
    return test_pkg


def patch_instruction_md(text: str, test_pkg: str) -> tuple[str, bool]:
    """
    Append a package-name hint to instruction.md. Idempotent via marker
    prefix ``Important: declare your solution in `package``.
    """
    if INSTRUCTION_MARKER_PREFIX in text:
        return text, False

    impl_pkg = implementation_pkg_for(test_pkg)
    hint = (
        f"\n\nImportant: declare your solution in `package {impl_pkg}` "
        f"(the test file uses `package {test_pkg}`, so the implementation "
        f"must use the matching package name).\n"
    )
    # Keep file ending with a single trailing newline.
    if text.endswith("\n"):
        new_text = text + hint.lstrip("\n")
    else:
        new_text = text + hint
    return new_text, True


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

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
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_testsh_patched = 0
    n_testsh_already = 0
    n_testsh_skipped = 0
    n_instr_patched = 0
    n_instr_already = 0
    n_instr_skipped_no_pkg = 0
    n_instr_missing_file = 0
    n_both = 0
    pkg_dist: Counter[str] = Counter()

    # Anomaly tracking
    n_no_test_file = 0
    n_no_testsh = 0
    n_no_instruction = 0

    for i, td in enumerate(task_dirs, 1):
        testsh_path = td / "tests" / "test.sh"
        instr_path = td / "instruction.md"
        test_go_path = td / "tests" / "solution_test.go"

        # ---- (A) test.sh injection
        testsh_changed = False
        if not testsh_path.exists():
            n_no_testsh += 1
            n_testsh_skipped += 1
        else:
            original = testsh_path.read_text()
            new_text, changed, reason = patch_test_sh(original)
            if reason == "already-patched":
                n_testsh_already += 1
            elif changed:
                n_testsh_patched += 1
                testsh_changed = True
                if not args.dry_run:
                    testsh_path.write_text(new_text)
            else:
                n_testsh_skipped += 1

        # ---- (B) instruction.md package-name hint
        instr_changed = False
        if not test_go_path.exists():
            n_no_test_file += 1
            n_instr_skipped_no_pkg += 1
        elif not instr_path.exists():
            n_no_instruction += 1
            n_instr_missing_file += 1
        else:
            test_text = test_go_path.read_text(errors="replace")
            pkg = parse_package_name(test_text)
            if pkg is None:
                n_instr_skipped_no_pkg += 1
            else:
                pkg_dist[pkg] += 1
                instr_text = instr_path.read_text()
                new_instr, changed = patch_instruction_md(instr_text, pkg)
                if INSTRUCTION_MARKER_PREFIX in instr_text:
                    n_instr_already += 1
                elif changed:
                    n_instr_patched += 1
                    instr_changed = True
                    if not args.dry_run:
                        instr_path.write_text(new_instr)

        if testsh_changed and instr_changed:
            n_both += 1

        if i % 1000 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] testsh={n_testsh_patched} "
                f"instr={n_instr_patched} both={n_both}",
                flush=True,
            )

    print()
    print("=" * 60)
    print(f"Total task dirs scanned:        {n_total}")
    print(f"Dry run:                        {args.dry_run}")
    print()
    print("(A) tests/test.sh patches")
    print(f"  injected (this run):          {n_testsh_patched}")
    print(f"  already patched:              {n_testsh_already}")
    print(f"  skipped (no go test line):    {n_testsh_skipped}")
    print(f"    of which: missing test.sh:  {n_no_testsh}")
    print()
    print("(B) instruction.md package-name hints")
    print(f"  appended (this run):          {n_instr_patched}")
    print(f"  already had hint:             {n_instr_already}")
    print(f"  skipped (no pkg directive):   {n_instr_skipped_no_pkg}")
    print(f"    of which: missing test.go:  {n_no_test_file}")
    print(f"  missing instruction.md:       {n_instr_missing_file}")
    print()
    print(f"Both patches applied:           {n_both}")
    print()
    print("Top 10 detected package names (test-file directive):")
    for name, n in pkg_dist.most_common(10):
        print(f"  {name:32s} {n}")
    print(f"Total unique packages:          {len(pkg_dist)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
