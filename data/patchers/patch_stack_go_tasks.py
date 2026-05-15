#!/usr/bin/env python3
"""
exp_rpt_stack-go v2 patcher.

Bug: 100% of v1 traces fail with

    RuntimeError: Failed to start tmux session.
    Error: bash: line 1: script: command not found

The Terminus-2 agent uses the `script` binary (typescript-recorder, not the
generic word "script") to capture tmux output. The stack-go base image is
`golang:1.21-alpine`, which does NOT ship `script` by default.

The user-facing instructions name two Debian packages (`bsdmainutils` and
`util-linux`) -- those are correct on Debian/Ubuntu, but the stack-go corpus
is Alpine-based (alpine 3.x, apk package manager). On Alpine:

  - `bsdmainutils` does not exist (Debian-only).
  - `script` lives in `util-linux-misc` (Alpine 3.18+) -- a subpackage that
    Alpine carved out of `util-linux` to keep the base small.

We therefore install `util-linux-misc` via `apk add --no-cache`, extending
the existing `apk add` line that already installs `bash git`. We keep the
patcher Alpine-only because every Dockerfile in the corpus is byte-identical
(md5sum verified across 10000 tasks: single hash `c602f8158d92802289431aa2fa103af8`).

If we ever encounter a Debian/Ubuntu variant in this corpus, the patcher
falls back to injecting an `apt-get install -y --no-install-recommends
bsdmainutils util-linux` RUN step.

Layout per task: <root>/<task_id>/environment/Dockerfile

The patch is idempotent via the marker
    `# --- laion v2 patch: install script (util-linux) ---`

CLI shape mirrors other patchers in this directory (`--root`, `--dry-run`,
`--limit`).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import Counter

# Marker emitted into every patched Dockerfile so we can detect prior runs
# (idempotency) and verify post-upload that >=99% of Dockerfiles got the patch.
MARKER = "# --- laion v2 patch: install script (util-linux) ---"

# The single Alpine `apk add ...` line that every v1 Dockerfile already has.
# We extend it in-place rather than adding a new RUN layer to keep the image
# layer count unchanged.
APK_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)RUN[ \t]+apk[ \t]+add[ \t]+--no-cache[ \t]+(?P<pkgs>[^\n]+?)[ \t]*$",
    flags=re.MULTILINE,
)

# Detect existing apt-get install lines for the Debian fallback.
APT_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)RUN[ \t]+(?:[^\n]*&&[ \t]*)?apt-get[ \t]+install[ \t]+(?P<flags>[^\n]*?)(?P<pkgs>[A-Za-z0-9._+\- ]+?)[ \t]*$",
    flags=re.MULTILINE,
)

# Alpine package that actually provides /usr/bin/script (verified in
# golang:1.21-alpine docker run). util-linux on Alpine alone does NOT contain
# script -- you need util-linux-misc.
ALPINE_PKG = "util-linux-misc"

# Debian fallback packages (only used if we ever see a Debian-based variant).
DEBIAN_PKGS = "bsdmainutils util-linux"


def patch_dockerfile(text: str) -> tuple[str, bool, str]:
    """
    Inject the script-providing package into the Dockerfile.

    Returns (new_text, changed, variant) where variant is one of
    "alpine-extended", "debian-extended", "debian-new-run", "alpine-new-run",
    "already-patched", or "unknown-base".
    """
    if MARKER in text:
        return text, False, "already-patched"

    # --- Alpine path: extend existing `apk add --no-cache <pkgs>` line.
    apk_match = APK_LINE_RE.search(text)
    if apk_match:
        existing = apk_match.group("pkgs").strip()
        # Don't double-add if util-linux-misc somehow already on the line.
        if ALPINE_PKG in existing.split():
            new_line = apk_match.group(0)
        else:
            new_pkgs = f"{existing} {ALPINE_PKG}"
            new_line = (
                f"{apk_match.group('indent')}"
                f"RUN apk add --no-cache {new_pkgs}"
            )
        # Insert marker comment immediately above the apk line for visibility.
        replacement = f"{apk_match.group('indent')}{MARKER}\n{new_line}"
        new_text = text[: apk_match.start()] + replacement + text[apk_match.end():]
        # Ensure trailing newline preserved
        if not new_text.endswith("\n"):
            new_text += "\n"
        return new_text, True, "alpine-extended"

    # --- Debian path: extend existing `apt-get install` if any.
    apt_match = APT_LINE_RE.search(text)
    if apt_match:
        existing = apt_match.group("pkgs").strip()
        flags = apt_match.group("flags") or ""
        already_have = set(existing.split()) | set(DEBIAN_PKGS.split())
        new_pkgs = " ".join(sorted(already_have))
        new_line = (
            f"{apt_match.group('indent')}"
            f"RUN apt-get install {flags}{new_pkgs}"
        )
        replacement = f"{apt_match.group('indent')}{MARKER}\n{new_line}"
        new_text = text[: apt_match.start()] + replacement + text[apt_match.end():]
        if not new_text.endswith("\n"):
            new_text += "\n"
        return new_text, True, "debian-extended"

    # --- No package-manager line found. Decide which package manager to use
    # by sniffing the `FROM` line. Alpine images contain "alpine" in the tag.
    from_line_re = re.compile(r"^FROM[ \t]+([^\s]+)", flags=re.MULTILINE)
    m = from_line_re.search(text)
    if not m:
        return text, False, "unknown-base"

    base = m.group(1).lower()
    insertion_pt = text.find("\n", m.start()) + 1
    if "alpine" in base:
        block = (
            f"\n{MARKER}\n"
            f"RUN apk add --no-cache {ALPINE_PKG}\n"
        )
        variant = "alpine-new-run"
    else:
        # Best-effort Debian/Ubuntu install. Use update + install in one RUN.
        block = (
            f"\n{MARKER}\n"
            f"RUN apt-get update && apt-get install -y --no-install-recommends "
            f"{DEBIAN_PKGS} && rm -rf /var/lib/apt/lists/*\n"
        )
        variant = "debian-new-run"

    new_text = text[:insertion_pt] + block + text[insertion_pt:]
    return new_text, True, variant


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

    dockerfiles = sorted(root.glob("*/environment/Dockerfile"))
    if not dockerfiles:
        print(f"No environment/Dockerfile files under {root}", file=sys.stderr)
        return 2

    if args.limit:
        dockerfiles = dockerfiles[: args.limit]

    n_total = len(dockerfiles)
    variant_counts: Counter[str] = Counter()
    n_patched = 0
    n_already = 0
    n_unknown = 0

    for i, df in enumerate(dockerfiles, 1):
        original = df.read_text()
        new_text, changed, variant = patch_dockerfile(original)
        variant_counts[variant] += 1

        if variant == "already-patched":
            n_already += 1
        elif variant == "unknown-base":
            n_unknown += 1
        elif changed:
            n_patched += 1
            if not args.dry_run:
                df.write_text(new_text)

        if i % 1000 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] patched={n_patched} already={n_already} "
                f"unknown={n_unknown}",
                flush=True,
            )

    print()
    print("=" * 60)
    print(f"Total Dockerfiles scanned: {n_total}")
    print(f"Patched (this run):        {n_patched}")
    print(f"Already patched:           {n_already}")
    print(f"Unknown base (skipped):    {n_unknown}")
    print(f"Dry run:                   {args.dry_run}")
    print()
    print("Variants encountered:")
    for v, c in variant_counts.most_common():
        print(f"  {v:24s} {c}")
    print("=" * 60)

    if n_unknown > 0:
        print(
            f"\nWARNING: {n_unknown} Dockerfile(s) had no recognized FROM/apk/apt "
            "line and were not patched. Review before uploading.",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
