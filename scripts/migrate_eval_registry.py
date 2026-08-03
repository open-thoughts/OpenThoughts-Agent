#!/usr/bin/env python
"""One-off migration: eval/configs/model_configs.yaml -> model_config/<org>/<slug>.yaml.

Splits the mono-file eval registry into per-model files with the layered schema:
  - MODEL-INTRINSIC base fields (trust_remote_code, max_model_len, tool_call_parser,
    reasoning_parser, hf_overrides, limit_mm_per_prompt) -> top-level of each file.
  - Everything else (TP, DP, swap, extra_args, conda_env, agent_kwargs, ...) ->
    ``subsystems.eval:`` (since the mono-file IS the eval registry).
  - ``name@profile`` standalones -> merged into the model's file under
    ``subsystems.eval.variants.<profile>``.
  - ``patterns:`` -> model_config/_patterns.yaml (with ``subsystems: [eval]`` defaults).

Run from the repo root:
    python scripts/migrate_eval_registry.py --dry-run   # preview
    python scripts/migrate_eval_registry.py             # write
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "eval" / "configs" / "model_configs.yaml"
DST = REPO / "model_config"

INTRINSIC = {
    "trust_remote_code", "hf_overrides", "limit_mm_per_prompt",
    "max_model_len", "tool_call_parser", "reasoning_parser",
}


def _slugify(model: str) -> tuple[str, str]:
    parts = model.split("/", 1)
    if len(parts) == 2:
        org, rest = parts
    else:
        org, rest = "_unaffiliated", model
    slug = re.sub(r"[^\w.\-]", "_", rest)
    return org, slug


def _split_entry(model_id: str, entry: dict) -> tuple[dict, dict]:
    """Split a mono-file entry into (base_intrinsic, eval_subsystem_overlay)."""
    base, sub = {}, {}
    for k, v in entry.items():
        if k in INTRINSIC:
            base[k] = v
        else:
            sub[k] = v
    base["model"] = model_id
    return base, sub


def _dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True))


def migrate(dry_run: bool = False) -> None:
    data = yaml.safe_load(SRC.read_text()) or {}
    models = data.get("models", {}) or {}
    patterns = data.get("patterns", []) or []

    # Group @profile standalones by their base model name.
    base_entries: dict[str, dict] = {}      # model_id -> entry
    profile_entries: dict[str, dict[str, dict]] = defaultdict(dict)  # model_id -> {profile -> entry}

    for key, entry in models.items():
        if "@" in key:
            base_name, profile = key.rsplit("@", 1)
            profile_entries[base_name][profile] = entry
        else:
            base_entries[key] = entry

    # Merge profile standalones into their base model's file.
    written = 0
    for model_id, entry in base_entries.items():
        base, eval_sub = _split_entry(model_id, entry)

        # Attach hardware variants from @profile standalones.
        # @profile entries in the mono-file are WHOLESALE REPLACES in
        # load_model_registry (not a merge over the bare entry) — they are
        # self-contained. Store the FULL @profile entry verbatim as the
        # variant; do NOT split it into intrinsic/non-intrinsic (that would
        # lose per-profile independence and break resolution parity).
        profiles = profile_entries.get(model_id, {})
        if profiles:
            variants = {}
            for prof, prof_entry in profiles.items():
                # Store the full entry minus the 'model' key (structural).
                variant_full = {k: v for k, v in prof_entry.items() if k != "model"}
                if variant_full:
                    variants[prof] = variant_full
            if variants:
                eval_sub["variants"] = variants

        file_data = base
        if eval_sub:
            file_data["subsystems"] = {"eval": {k: v for k, v in eval_sub.items() if k != "model"}}

        org, slug = _slugify(model_id)
        out_path = DST / org / f"{slug}.yaml"
        if dry_run:
            print(f"WOULD WRITE {out_path.relative_to(REPO)}: {list(file_data.keys())}"
                  + (f" + variants {list(eval_sub.get('variants', {}).keys())}" if eval_sub.get("variants") else ""))
        else:
            _dump_yaml(file_data, out_path)
            print(f"wrote {out_path.relative_to(REPO)}")
        written += 1

    # @profile standalones whose base model has NO bare entry -> standalone file with
    # ONLY variants (no base/eval fields), so the generator emits @profile entries but
    # NOT a spurious bare entry.
    for base_name, profiles in profile_entries.items():
        if base_name in base_entries:
            continue  # already merged above
        # No bare entry existed — store the full @profile entries verbatim as variants
        # under a minimal file (model id + variants only).
        variants = {}
        for prof, prof_entry in profiles.items():
            variant_full = {k: v for k, v in prof_entry.items() if k != "model"}
            if variant_full:
                variants[prof] = variant_full
        file_data = {"model": base_name}
        if variants:
            file_data["subsystems"] = {"eval": {"variants": variants}}
        org, slug = _slugify(base_name)
        out_path = DST / org / f"{slug}.yaml"
        if dry_run:
            print(f"WOULD WRITE (profile-only) {out_path.relative_to(REPO)}")
        else:
            _dump_yaml(file_data, out_path)
            print(f"wrote (profile-only) {out_path.relative_to(REPO)}")
        written += 1

    # Patterns -> _patterns.yaml (preserve `profiles` verbatim — the legacy scoping
    # mechanism; the resolver checks it directly). Strip only `match` (re-added first).
    pat_out = []
    for pat in patterns:
        regex = pat.get("match", "")
        entry = {"match": regex}
        for k, v in pat.items():
            if k == "match":
                continue
            entry[k] = v  # keep `profiles`, `subsystems`, and all config fields verbatim
        pat_out.append(entry)
    pat_path = DST / "_patterns.yaml"
    if dry_run:
        print(f"WOULD WRITE {pat_path.relative_to(REPO)}: {len(pat_out)} patterns")
    else:
        _dump_yaml({"patterns": pat_out}, pat_path)
        print(f"wrote {pat_path.relative_to(REPO)}: {len(pat_out)} patterns")

    print(f"\n{'DRY RUN: ' if dry_run else ''}{written} model files + {len(pat_out)} patterns.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    migrate(dry_run=args.dry_run)
