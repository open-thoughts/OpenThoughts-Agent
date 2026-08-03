#!/usr/bin/env python
"""Generate eval/configs/model_configs.yaml from model_config/ (the single editable source).

Reverse of scripts/migrate_eval_registry.py: reads every per-model file under
``model_config/<org>/<slug>.yaml`` + ``model_config/_patterns.yaml`` and emits the
flat mono-file registry the eval listener consumes
(``eval/unified_eval_listener.load_model_registry``).

The mono-file is a GENERATED BUILD ARTIFACT — do not hand-edit it. Edit the per-model
files under ``model_config/`` and re-run this script (or the ``--check`` mode in CI to
fail if the committed mono-file is stale).

Run from the repo root:
    python scripts/generate_eval_registry.py             # regenerate
    python scripts/generate_eval_registry.py --check     # CI gate: nonzero if stale
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "model_config"
DST = REPO / "eval" / "configs" / "model_configs.yaml"

INTRINSIC = {
    "trust_remote_code", "hf_overrides", "limit_mm_per_prompt",
    "max_model_len", "tool_call_parser", "reasoning_parser",
}


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def generate() -> dict:
    """Build the mono-file registry dict from the per-model files."""
    models: dict[str, dict] = {}

    for org_dir in sorted(SRC_DIR.iterdir()):
        if not org_dir.is_dir() or org_dir.name.startswith(("_", ".")):
            continue
        for f in sorted(org_dir.glob("*.yaml")):
            data = _load(f)
            model_id = data.get("model")
            if not model_id:
                continue

            # Layer 1: base intrinsics (top-level fields, minus structural keys).
            entry: dict = {}
            for k, v in data.items():
                if k not in ("model", "subsystems", "variants", "notes"):
                    entry[k] = v

            # Layer 2: the eval subsystem's flat fields (merged on top of base).
            eval_sub = (data.get("subsystems") or {}).get("eval") or {}
            for k, v in eval_sub.items():
                if k == "variants":
                    continue
                entry[k] = v

            if entry:
                models[model_id] = entry

            # Layer 3: @profile standalones from the eval subsystem's variants.
            # These are WHOLESALE REPLACES in load_model_registry — emitted
            # VERBATIM (the variant fields exactly as stored, NOT base+variant).
            variants = eval_sub.get("variants") or {}
            for profile, variant_fields in variants.items():
                if variant_fields:
                    models[f"{model_id}@{profile}"] = dict(variant_fields)

    # Patterns: pass through verbatim from _patterns.yaml (profiles preserved).
    patterns_out = []
    pat_path = SRC_DIR / "_patterns.yaml"
    if pat_path.is_file():
        pat_data = _load(pat_path)
        for pat in pat_data.get("patterns", []):
            # match goes first (matches original ordering); everything else verbatim.
            out = {"match": pat.get("match", "")}
            out.update({k: v for k, v in pat.items() if k != "match"})
            patterns_out.append(out)

    return {"models": models, "patterns": patterns_out}


def _dump(data: dict) -> str:
    header = (
        "# AUTO-GENERATED from model_config/ — do NOT hand-edit.\n"
        "# Edit the per-model files under model_config/<org>/<slug>.yaml and run:\n"
        "#     python scripts/generate_eval_registry.py\n"
        "# (or scripts/generate_eval_registry.py --check in CI to detect drift).\n"
        "# Source of truth: model_config/ (file-per-model, layered schema).\n"
        "\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero if the committed mono-file is stale vs model_config/")
    args = ap.parse_args()

    data = generate()

    if args.check:
        existing = yaml.safe_load(DST.read_text()) if DST.is_file() else {}
        if existing != data:
            print(f"STALE: {DST.relative_to(REPO)} does not match model_config/. "
                  f"Run: python scripts/generate_eval_registry.py", file=sys.stderr)
            return 1
        print(f"OK: {DST.relative_to(REPO)} is up to date vs model_config/.")
        return 0

    DST.write_text(_dump(data))
    print(f"WROTE {DST.relative_to(REPO)}: "
          f"{len(data['models'])} models, {len(data['patterns'])} patterns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
