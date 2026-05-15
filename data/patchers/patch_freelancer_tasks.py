#!/usr/bin/env python3
"""
Patch every tests/test.sh under a Harbor task tree so that:
  - The container env shortcomings (missing /testbed, miniconda, python alias,
    pytest) are compensated for at runtime
  - The unconditional `echo "VERIFIER: PASS"` false-positive (set -uo pipefail
    style scripts) is closed by forcing set -e
  - /logs/verifier/reward.txt is always written, even when the script crashes
    before reaching parser.py (no more VerifierRuntimeError: No reward file)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PREAMBLE = r"""# --- laion v2 verifier patch: env shims + reward floor ---
mkdir -p /logs/verifier 2>/dev/null || true
echo 0 > /logs/verifier/reward.txt 2>/dev/null || true

# /testbed → /app (Dockerfile sets WORKDIR /app; tasks reference /testbed)
if [ ! -e /testbed ]; then
  ln -s /app /testbed 2>/dev/null || mkdir -p /testbed
fi

# python alias for scripts that hard-code `python`
if [ ! -e /usr/local/bin/python ] && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python 2>/dev/null || true
fi

# Stub miniconda so `source /opt/miniconda3/bin/activate` and `conda activate`
# are no-ops instead of fatal errors
mkdir -p /opt/miniconda3/bin 2>/dev/null || true
if [ ! -f /opt/miniconda3/bin/activate ]; then
  printf '%s\n' '#!/bin/bash' ':' > /opt/miniconda3/bin/activate
  chmod +x /opt/miniconda3/bin/activate 2>/dev/null || true
fi
if ! command -v conda >/dev/null 2>&1; then
  printf '%s\n' '#!/bin/bash' 'exit 0' > /opt/miniconda3/bin/conda
  chmod +x /opt/miniconda3/bin/conda 2>/dev/null || true
fi
export PATH="/opt/miniconda3/bin:$PATH"

# Install pytest if missing; PEP 668 requires --break-system-packages on Ubuntu 24.04
if ! command -v pytest >/dev/null 2>&1; then
  python3 -m pip install --quiet --break-system-packages pytest 2>/dev/null \
    || python3 -m pip install --quiet pytest 2>/dev/null \
    || true
fi

# Always leave reward.txt populated even if the script aborts mid-way
trap '[ -s /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt' EXIT
# --- end laion v2 patch ---
"""


SHEBANG_RE = re.compile(r"^(#![^\n]*\n)")
SET_LINE_RE = re.compile(
    r"^(\s*)set\s+-([uo][^\n]*?pipefail[^\n]*)$",
    flags=re.MULTILINE,
)
# `python(3)? -m pip install ...` lines without an existing `||` tail.
# We make pip installs non-fatal so set -e doesn't kill the verifier when the
# agent's /app lacks setup.py/pyproject.toml or PEP 668 blocks the install.
PIP_INSTALL_RE = re.compile(
    r"^(?P<indent>\s*)(?P<cmd>python3?\s+-m\s+pip\s+install\b[^\n|]*)$",
    flags=re.MULTILINE,
)


def _ensure_pip_tolerant(match: re.Match[str]) -> str:
    return f"{match.group('indent')}{match.group('cmd').rstrip()} || true"


def patch_test_sh(text: str) -> tuple[str, bool]:
    """Return (patched_text, changed)."""
    original = text

    # 1. Force `-e` into any `set -uo pipefail [...]` line that lacks it.
    def _ensure_e(match: re.Match[str]) -> str:
        indent, rest = match.group(1), match.group(2)
        if "e" in rest.split()[0]:
            return match.group(0)
        # rest starts with e.g. "uo pipefail -x"; insert e after first flag char
        first = rest[:rest.index("o")] + "e" + rest[rest.index("o"):]
        return f"{indent}set -{first}"

    text = SET_LINE_RE.sub(_ensure_e, text)

    # 2. Make `pip install` lines non-fatal.
    text = PIP_INSTALL_RE.sub(_ensure_pip_tolerant, text)

    # 3. Prepend the preamble right after the shebang line (or at top if none).
    m = SHEBANG_RE.match(text)
    if m:
        text = m.group(1) + PREAMBLE + text[m.end():]
    else:
        text = "#!/bin/bash\n" + PREAMBLE + text

    return text, text != original


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

    test_paths = sorted(root.glob("*/tests/test.sh"))
    if not test_paths:
        print(f"No tests/test.sh files under {root}", file=sys.stderr)
        return 2

    if args.limit:
        test_paths = test_paths[: args.limit]

    n_changed = 0
    n_total = len(test_paths)
    for i, p in enumerate(test_paths, 1):
        original = p.read_text()
        patched, changed = patch_test_sh(original)
        if changed:
            n_changed += 1
            if not args.dry_run:
                p.write_text(patched)
        if i % 1000 == 0 or i == n_total:
            print(f"[{i}/{n_total}] patched={n_changed}", flush=True)

    print(f"Done. {n_changed}/{n_total} files modified (dry_run={args.dry_run}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
