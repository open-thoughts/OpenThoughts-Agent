"""One-off: complete the BASE-model agentic passes without redoing instruct.

The full re-run was interrupted during base/function_calling. The shim log
already holds the good (eos-fixed) INSTRUCT captures (both parsers). This script:
  1. Keeps only the instruct records in the shim log (drops the partial base ones).
  2. Starts the BASE shim and drives swe-agent over BOTH parser variants x 5 repos
     (appending fresh base captures).
  3. Re-renders the full log (instruct + base) -> all agentic HTML.
Run with the tokviz-rt interpreter.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_agentic as ra
from config import SHIM_LOG


def keep_only_instruct() -> int:
    if not SHIM_LOG.exists():
        return 0
    recs = [json.loads(l) for l in SHIM_LOG.read_text().splitlines() if l.strip()]  # noqa: E741
    instruct = [r for r in recs if not r.get("model_id", "").endswith("Base")]
    with open(SHIM_LOG, "w", encoding="utf-8") as f:
        for r in instruct:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(instruct)


def main() -> None:
    kept = keep_only_instruct()
    print(f"[finish-base] preserved {kept} instruct records in shim log", flush=True)

    if not ra.docker_available() or not Path(ra.SWEAGENT_BIN).exists():
        print("[finish-base] ERROR: docker or sweagent missing; aborting", flush=True)
        sys.exit(2)

    # Base shim up; drive BOTH parser variants over all repos (appends to log).
    shim = ra.start_shim("base")
    try:
        for parser_key, bundled_config, _suffix in ra.VARIANTS:
            print(f"\n[finish-base] base / {parser_key} (config={bundled_config.name})",
                  flush=True)
            for repo in sorted(p for p in ra.REPOS_DIR.iterdir() if p.is_dir()):
                ra.run_sweagent_on_repo("base", repo, bundled_config)
    finally:
        shim.terminate()
        try:
            shim.wait(timeout=10)
        except Exception:
            shim.kill()
        print("[finish-base] base shim stopped", flush=True)

    # Render the full (instruct + base) log.
    from render_agentic import render_first_calls_per_which
    from transformers import AutoTokenizer

    @functools.lru_cache(maxsize=4)
    def loader(mid: str):
        return AutoTokenizer.from_pretrained(mid)

    produced = render_first_calls_per_which(loader)
    print(f"\n[finish-base] RENDERED {len(produced)} agentic HTML files:", flush=True)
    for p in produced:
        print("  ", p, flush=True)
    print("[finish-base] DONE", flush=True)


if __name__ == "__main__":
    main()
