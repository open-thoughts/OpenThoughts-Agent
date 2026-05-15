#!/usr/bin/env python3
"""
exp_rpt_stack-csharp v5 patcher.

Bug (v4 -> v5): v4 added 91 NuGet `namespace -> package` mappings and emits
`dotnet add package <pkg>` lines into test.sh. The bare `dotnet add package
<pkg>` (no `--version`) resolves to the LATEST version on NuGet. For some
package families (Microsoft.EntityFrameworkCore.*, Microsoft.Extensions.*,
Microsoft.AspNetCore.*, Microsoft.Identity.Client) the latest version targets
`net10.0` or `net9.0` and is INCOMPATIBLE with the test container's `net8.0`
(`dotnet new xunit` on the .NET 8 SDK).

Concrete failure mode observed in v4 validation (~5% of trials):

    error: NU1202: Package Microsoft.EntityFrameworkCore 10.0.7 is not
           compatible with net8.0 (.NETCoreApp,Version=v8.0). Package
           Microsoft.EntityFrameworkCore 10.0.7 supports: net10.0

This caused csharp solve rate to drop from v3 3.0% -> v4 1.0% -- a regression.

Fix (v5): a single targeted, idempotent rewrite. We:

  1. Read each tasks/test.sh.
  2. Find every `dotnet add package <pkg> ...` line emitted by v2 OR v4.
  3. If <pkg> is in our `_PINNED_VERSIONS` map, REWRITE the line in place to
     pin to the net8-compatible version range (e.g. `[8.0,9.0)`). All other
     packages (Akka, AutoMapper, FluentAssertions, Moq, Polly, Serilog, etc.)
     are LEFT UNPINNED -- those are stable across major versions on net8.0.
  4. Wrap the changes in a single v5 idempotency marker so re-runs no-op.

We do NOT re-run v4's namespace-mapping logic from scratch. v4 is
non-destructive (it appends a block; doesn't rewrite v2's block), so by the
time v5 runs we can trust that all `dotnet add package` lines that need fixing
already exist in test.sh; the only thing wrong with them is the missing
`--version` flag. v5 is the smallest possible diff that resolves the
regression.

Conservative pin policy: a package is pinned ONLY if (a) its package id
belongs to a Microsoft package family known to have major versions >= 9 with
net8.0-incompatible latests, AND (b) an 8.x version exists on NuGet that
targets net8.0. When in doubt, leave unpinned -- a wrong pin is worse than no
pin (it can mask a real-version-resolution failure).

Constraints:
  - DO NOT modify tests/TestSolution.cs.
  - DO NOT mutate or remove v2/v4 markers in test.sh.
  - DO NOT drop tasks. v5 mutates lines in place; survivor counts stay flat.
  - Idempotent on re-runs (v5 marker check).

Usage:
  python data/patchers/patch_stack_csharp_v5_tasks.py \
      --root /path/to/exp_rpt_stack-csharp-v5 \
      [--dry-run] [--limit N] [--drop-log path.tsv]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Markers (idempotency)
# --------------------------------------------------------------------------- #

_V5_TESTSH_MARKER = "# --- laion v5 patch: pin net8.0-compatible NuGet versions ---"
_V5_TESTSH_END_MARKER = "# --- end laion v5 patch ---"

# --------------------------------------------------------------------------- #
# Pinned-version map: package id -> dotnet add package --version argument.
#
# We use the range form `[8.0,9.0)` (inclusive lower, exclusive upper) because
# it is unambiguous: NuGet will pick the highest 8.x version that has a
# net8.0-compatible TFM. Floating syntax (`8.*`) is equivalent in practice but
# the range form is more portable across nuget client versions.
#
# Inclusion criteria (all three must hold):
#   (a) Package family has shipped 9.x and/or 10.x. Latest targets net9.0+
#       (and therefore breaks net8.0 restore).
#   (b) An 8.x version exists on NuGet that supports net8.0.
#   (c) The package id is one v4 actually emits (otherwise pin is dead code).
#
# Out of scope (deliberately UNPINNED): Akka.* (1.x/2.x track), AutoMapper
# (12.x but supports net8.0), FluentAssertions (6.x/7.x), Moq (4.x), Polly
# (8.x supports net8.0), Serilog (4.x), NodaTime (3.x), Newtonsoft.Json (13.x
# multi-targets net8.0), MathNet.Numerics (5.x), NUnit (3.x/4.x), xunit
# (2.x). All of these have net8.0-compatible LATEST versions.
# --------------------------------------------------------------------------- #

_PIN_NET8 = "[8.0,9.0)"

_PINNED_VERSIONS: dict[str, str] = {
    # ----- Microsoft.EntityFrameworkCore family -----
    "Microsoft.EntityFrameworkCore": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.SqlServer": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.Sqlite": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.Tools": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.InMemory": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.Design": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.Relational": _PIN_NET8,
    "Microsoft.EntityFrameworkCore.Cosmos": _PIN_NET8,
    # ----- Microsoft.Extensions.* family -----
    # 8.x track explicitly targets net8.0; 9.x targets net9.0.
    "Microsoft.Extensions.DependencyInjection": _PIN_NET8,
    "Microsoft.Extensions.DependencyInjection.Abstractions": _PIN_NET8,
    "Microsoft.Extensions.Logging": _PIN_NET8,
    "Microsoft.Extensions.Logging.Abstractions": _PIN_NET8,
    "Microsoft.Extensions.Logging.Console": _PIN_NET8,
    "Microsoft.Extensions.Logging.Debug": _PIN_NET8,
    "Microsoft.Extensions.Configuration": _PIN_NET8,
    "Microsoft.Extensions.Configuration.Json": _PIN_NET8,
    "Microsoft.Extensions.Configuration.Abstractions": _PIN_NET8,
    "Microsoft.Extensions.Configuration.Binder": _PIN_NET8,
    "Microsoft.Extensions.Configuration.EnvironmentVariables": _PIN_NET8,
    "Microsoft.Extensions.Configuration.UserSecrets": _PIN_NET8,
    "Microsoft.Extensions.Configuration.CommandLine": _PIN_NET8,
    "Microsoft.Extensions.Options": _PIN_NET8,
    "Microsoft.Extensions.Options.ConfigurationExtensions": _PIN_NET8,
    "Microsoft.Extensions.Hosting": _PIN_NET8,
    "Microsoft.Extensions.Hosting.Abstractions": _PIN_NET8,
    "Microsoft.Extensions.Caching.Memory": _PIN_NET8,
    "Microsoft.Extensions.Caching.Abstractions": _PIN_NET8,
    "Microsoft.Extensions.Http": _PIN_NET8,
    "Microsoft.Extensions.FileProviders": _PIN_NET8,
    "Microsoft.Extensions.Primitives": _PIN_NET8,
    # ----- Microsoft.AspNetCore.* family -----
    # 8.x exists on NuGet for all the sub-packages v4 references.
    "Microsoft.AspNetCore.Mvc.Core": _PIN_NET8,
    "Microsoft.AspNetCore.Mvc.Testing": _PIN_NET8,
    "Microsoft.AspNetCore.Mvc.ViewFeatures": _PIN_NET8,
    "Microsoft.AspNetCore.Mvc.RazorPages": _PIN_NET8,
    "Microsoft.AspNetCore.Mvc.Abstractions": _PIN_NET8,
    "Microsoft.AspNetCore.TestHost": _PIN_NET8,
    "Microsoft.AspNetCore.Hosting": _PIN_NET8,
    "Microsoft.AspNetCore.Http.Abstractions": _PIN_NET8,
    "Microsoft.AspNetCore.Http.Features": _PIN_NET8,
    "Microsoft.AspNetCore.Routing": _PIN_NET8,
    "Microsoft.AspNetCore.Components": _PIN_NET8,
    "Microsoft.AspNetCore.Components.Web": _PIN_NET8,
    "Microsoft.AspNetCore.SignalR": _PIN_NET8,
    "Microsoft.AspNetCore.SignalR.Client": _PIN_NET8,
    "Microsoft.AspNetCore.Identity": _PIN_NET8,
    "Microsoft.AspNetCore.Authentication": _PIN_NET8,
    "Microsoft.AspNetCore.Authorization": _PIN_NET8,
    "Microsoft.AspNetCore.Cors": _PIN_NET8,
    "Microsoft.AspNetCore.WebUtilities": _PIN_NET8,
    "Microsoft.AspNetCore.Server.Kestrel.Core": _PIN_NET8,
    "Microsoft.AspNetCore.DataProtection": _PIN_NET8,
    "Microsoft.AspNetCore.Diagnostics.Abstractions": _PIN_NET8,
    # Note: Microsoft.AspNetCore.App.Ref is a framework-reference meta package;
    # it doesn't follow the same versioning trap. We DO NOT pin it.
    # ----- Microsoft.Identity.Client (and adjacent enterprise libs) -----
    # Aggressive major-version bumps; latest 4.x supports net8.0 but the next
    # major (5.x) is anticipated. Pin to <5.0 conservatively.
    "Microsoft.Identity.Client": "[4.0,5.0)",
}

# How many distinct pin rules are configured (for the report).
N_PINNED_PACKAGES: int = len(_PINNED_VERSIONS)


# --------------------------------------------------------------------------- #
# Line rewriter
# --------------------------------------------------------------------------- #

# Match a `dotnet add package <pkg>` invocation, optionally followed by extra
# args (e.g. v4's `>/dev/null 2>&1 || dotnet add package <pkg>` retry tail).
# We capture the leading whitespace, the package id, and everything after
# (which we re-emit verbatim AFTER the injected --version, so retry/redirect
# semantics are preserved).
#
# NB: package ids on NuGet may contain `.` `-` `_` and digits, plus letters.
_DOTNET_ADD_LINE_RE = re.compile(
    r"^(?P<indent>\s*)dotnet\s+add\s+package\s+(?P<pkg>[A-Za-z0-9_][\w\.\-]*)(?P<rest>.*)$"
)


def _rewrite_dotnet_add_line(line: str) -> tuple[str, bool]:
    """If `line` is `dotnet add package <pinned-pkg> ...`, inject --version.

    Returns (new_line, changed). `changed` is True iff a substitution occurred.
    Lines that already contain `--version` are left untouched (caller may have
    already pinned this package; v5 must be idempotent and conservative).
    """
    m = _DOTNET_ADD_LINE_RE.match(line)
    if not m:
        return line, False
    pkg = m.group("pkg")
    rest = m.group("rest")
    if pkg not in _PINNED_VERSIONS:
        return line, False
    if "--version" in line:
        # Already pinned (by us on a prior run, or by hand). Leave it.
        return line, False
    version_arg = _PINNED_VERSIONS[pkg]

    # Be careful with the v4 retry idiom:
    #   dotnet add package X >/dev/null 2>&1 || dotnet add package X
    # We need to inject --version into BOTH halves. Splitting on `||` and
    # re-rewriting is the cleanest way; the regex above only matched the
    # first half. We re-process `rest` to find a tail `dotnet add package X`.
    indent = m.group("indent")
    head = f'dotnet add package {pkg} --version "{version_arg}"'

    # Detect the retry tail: ` >/dev/null 2>&1 || dotnet add package <pkg>`
    # (we only retry-pin if the retry-package matches the head package id).
    tail_match = re.search(
        r"(\|\|\s*dotnet\s+add\s+package\s+)(" + re.escape(pkg) + r")(\b)",
        rest,
    )
    if tail_match:
        new_rest = (
            rest[: tail_match.start()]
            + tail_match.group(1)
            + tail_match.group(2)
            + f' --version "{version_arg}"'
            + rest[tail_match.end():]
        )
    else:
        new_rest = rest

    return f"{indent}{head}{new_rest}", True


def patch_test_sh(original: str) -> tuple[str | None, dict]:
    """Rewrite `dotnet add package <pinned-pkg>` lines to include --version.

    Returns (new_contents, info). `new_contents` is None if no change was
    made (already-patched OR no pinned packages present).
    info has keys:
      - 'changed': bool
      - 'lines_rewritten': int
      - 'pkgs_pinned': Counter[str]  (package id -> count of lines rewritten)
      - 'reason': str | None
    """
    info = {
        "changed": False,
        "lines_rewritten": 0,
        "pkgs_pinned": Counter(),
        "reason": None,
    }

    if _V5_TESTSH_MARKER in original:
        info["reason"] = "already_patched"
        return None, info

    lines = original.splitlines(keepends=True)
    out_lines: list[str] = []
    rewrites = 0
    pkg_counter: Counter[str] = Counter()

    for line in lines:
        # splitlines(keepends=True) preserves the trailing newline; strip it
        # before regex match so `rest` doesn't accidentally swallow the LF.
        if line.endswith("\r\n"):
            body, eol = line[:-2], "\r\n"
        elif line.endswith("\n"):
            body, eol = line[:-1], "\n"
        else:
            body, eol = line, ""

        new_body, changed = _rewrite_dotnet_add_line(body)
        if changed:
            rewrites += 1
            m = _DOTNET_ADD_LINE_RE.match(body)
            if m:
                pkg_counter[m.group("pkg")] += 1
            out_lines.append(new_body + eol)
        else:
            out_lines.append(line)

    if rewrites == 0:
        info["reason"] = "no_pinned_packages_present"
        return None, info

    # Inject the v5 marker block at the top of the script body (after the
    # shebang line if present) so re-runs short-circuit. Anchor: first
    # blank-or-comment line after the shebang, fall back to position 0.
    new_text = "".join(out_lines)
    marker_block = (
        f"{_V5_TESTSH_MARKER}\n"
        f"# Pinned NuGet versions to keep restore compatible with net8.0.\n"
        f"# (See data/patchers/patch_stack_csharp_v5_tasks.py for the rule list.)\n"
        f"{_V5_TESTSH_END_MARKER}\n"
    )

    if new_text.startswith("#!"):
        # Insert right after the shebang line.
        first_nl = new_text.find("\n")
        if first_nl == -1:
            new_text = new_text + "\n" + marker_block
        else:
            new_text = (
                new_text[: first_nl + 1] + marker_block + new_text[first_nl + 1 :]
            )
    else:
        new_text = marker_block + new_text

    info["changed"] = True
    info["lines_rewritten"] = rewrites
    info["pkgs_pinned"] = pkg_counter
    return new_text, info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--drop-log",
        type=str,
        default=None,
        help=(
            "Optional path to write a TSV of (task_dir<TAB>reason) for tasks "
            "that v5 could NOT mutate (e.g. no test.sh, unparseable). v5 is "
            "non-destructive; this is informational, not a delete-list."
        ),
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "instruction.md").exists()
    )
    if not task_dirs:
        print(f"No task dirs (with instruction.md) under {root}", file=sys.stderr)
        return 2

    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_changed = 0
    n_already_patched = 0
    n_skipped_no_test = 0
    n_no_pinned_pkgs = 0

    pkg_counter: Counter[str] = Counter()
    drop_log_lines: list[str] = []

    for i, d in enumerate(task_dirs, 1):
        testsh_path = d / "tests" / "test.sh"
        if not testsh_path.is_file():
            n_skipped_no_test += 1
            drop_log_lines.append(f"{d.name}\tno_test_sh")
            continue
        try:
            testsh_src = testsh_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_skipped_no_test += 1
            drop_log_lines.append(f"{d.name}\tread_failed")
            continue

        new_testsh, info = patch_test_sh(testsh_src)

        if info["reason"] == "already_patched":
            n_already_patched += 1
            continue
        if info["reason"] == "no_pinned_packages_present":
            n_no_pinned_pkgs += 1
            continue

        if new_testsh is None:
            # Defensive: shouldn't reach here, but treat as "no change needed".
            n_no_pinned_pkgs += 1
            continue

        n_changed += 1
        for pkg, c in info["pkgs_pinned"].items():
            pkg_counter[pkg] += c

        if not args.dry_run:
            testsh_path.write_text(new_testsh, encoding="utf-8")

        if i % 200 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} "
                f"already_patched={n_already_patched} "
                f"no_pinned_pkgs={n_no_pinned_pkgs} "
                f"skipped_no_test={n_skipped_no_test}",
                flush=True,
            )

    print(
        f"\nDone. {n_changed}/{n_total} task dirs modified "
        f"(dry_run={args.dry_run}).\n"
        f"  already_patched_skip     = {n_already_patched}\n"
        f"  no_pinned_pkgs           = {n_no_pinned_pkgs}\n"
        f"  skipped_no_test          = {n_skipped_no_test}\n"
        f"  v5_pin_rules_configured  = {N_PINNED_PACKAGES}\n"
    )
    print("Top 20 most-frequent NuGet packages pinned by v5:")
    for pkg, c in pkg_counter.most_common(20):
        print(f"  {c:>5}  {pkg}")

    if args.drop_log and drop_log_lines:
        Path(args.drop_log).write_text("\n".join(drop_log_lines) + "\n")
        print(f"\nDrop log: {args.drop_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
