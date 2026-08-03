"""Resolver for the unified per-model config (model_config/).

The single resolution function every entrypoint (eval listener, Iris, datagen)
calls to answer "what vLLM params does model X need in context Y?".

Resolution merge order (later wins, most-specific):
    base intrinsics  ->  subsystem(s)[0]  ->  ...  ->  subsystem(s)[-1]  ->  hardware variant

Falls back to regex patterns (model_config/_patterns.yaml) when no per-model
file exists, preserving the eval registry's size-inference defaults.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

MODEL_CONFIG_DIR = Path(__file__).resolve().parent

# Fields that are MODEL-INTRINSIC — the same regardless of subsystem. A field
# goes to the base ONLY if the model's entry actually carries it (so Pattern-D
# models that deliberately omit tool_call_parser/reasoning_parser stay omitted).
INTRINSIC_FIELDS = frozenset({
    "trust_remote_code",
    "hf_overrides",
    "limit_mm_per_prompt",
    "max_model_len",
    "tool_call_parser",
    "reasoning_parser",
})


def _slugify(model: str) -> tuple[str, str]:
    """Return (org, slug) for a model HF id, for the model_config/<org>/<slug>.yaml path.

    ``Qwen/Qwen3-32B`` -> ("Qwen", "Qwen3-32B"). Org is the first path segment
    (case-preserved); slug is the rest with path-unsafe chars replaced.
    """
    parts = model.split("/", 1)
    if len(parts) == 2:
        org, rest = parts
    else:
        org, rest = "_unaffiliated", model
    # Make the slug filesystem-safe (keep alnum, dash, underscore, dot; replace the rest).
    slug = re.sub(r"[^\w.\-]", "_", rest)
    return org, slug


def find_model_file(model: str) -> Optional[Path]:
    """Locate the per-model YAML for an HF id, or None if no exact file exists."""
    org, slug = _slugify(model)
    candidate = MODEL_CONFIG_DIR / org / f"{slug}.yaml"
    return candidate if candidate.is_file() else None


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Shallow-aware merge: dict values merge recursively; scalars replace."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _strip_internal_keys(d: dict) -> dict:
    """Remove keys that are schema/structural (model, subsystems, variants) — not vLLM params."""
    return {k: v for k, v in d.items() if k not in ("model", "subsystems", "variants", "notes")}


def _resolve_patterns(model: str, subsystem: str, hardware: Optional[str]) -> dict:
    """Regex-fallback resolution from model_config/_patterns.yaml (size-inference defaults).

    Patterns are evaluated in order; first match wins (mirrors the eval registry's
    precedence). A pattern applies when its ``profiles`` list (default ``["default"]``)
    contains the hardware profile name (``"default"`` when hardware is None) OR the
    subsystem. The hardware-variant cascade is NOT applied here (patterns are coarse).
    """
    patterns_path = MODEL_CONFIG_DIR / "_patterns.yaml"
    if not patterns_path.is_file():
        return {}
    data = _load_yaml(patterns_path)
    # The "active profile" for matching: hardware name if set, else "default".
    active_profile = hardware or "default"
    for pat in data.get("patterns", []):
        regex = pat.get("match")
        if not regex:
            continue
        profiles = pat.get("profiles") or ["default"]
        if active_profile not in profiles:
            continue
        if re.search(regex, model):
            return _strip_internal_keys(
                {k: v for k, v in pat.items() if k not in ("match", "profiles", "subsystems")}
            )
    return {}


def resolve_model_config(
    model: str,
    subsystem: str = "eval",
    hardware: Optional[str] = None,
    subsystems: Optional[Sequence[str]] = None,
) -> dict:
    """Resolve the vLLM serve config for ``model`` in a given context.

    Args:
        model: HF model id (e.g. ``"Qwen/Qwen3-32B"``).
        subsystem: The primary consumer profile (``"eval"`` / ``"datagen"`` / ``"iris"``).
        hardware: Optional hardware variant name (``"multi_gpu"`` / ``"gh200"`` / ...).
            Merged LAST (wins over subsystem + base).
        subsystems: Optional ordered list of subsystem profiles to stack (earlier
            is less specific). When given, ``subsystem`` is prepended. Use this for
            named overrides (e.g. ``["eval", "pattern_d"]``).

    Returns:
        A flat dict of vLLM params (the merge of base + subsystem(s) + variant),
        with structural keys (``model``, ``subsystems``, ``variants``) stripped.
        Empty dict if the model has no file and no pattern matches.
    """
    chain: List[str] = []
    if subsystems:
        chain = list(subsystems)
    if subsystem not in chain:
        chain.insert(0, subsystem)

    model_file = find_model_file(model)
    if model_file is None:
        # Fallback: regex patterns (size-inference defaults).
        return _resolve_patterns(model, subsystem, hardware)

    data = _load_yaml(model_file)

    # NOTE: the old mono-file registry excluded bare entries on non-default hardware
    # profiles (forcing pattern fallback). That was a quirk of its profile-splitting
    # mechanics, not desired behavior. The new resolver does the sensible thing: an
    # unmatched hardware variant just means no variant overlay — the model's base
    # intrinsics + subsystem config still apply. The listener (which still reads the
    # mono-file directly) is unaffected; migrating it to this resolver with the old
    # exclusion semantics is a careful follow-up.

    # Layer 1: base intrinsics.
    merged = _strip_internal_keys(data)

    # Layer 2+: subsystem overlays (in chain order; later wins).
    subs_block = data.get("subsystems", {})
    for sub in chain:
        sub_overlay = subs_block.get(sub)
        if sub_overlay:
            # Separate the variant from the subsystem-level fields.
            variant_block = sub_overlay.get("variants", {})
            sub_fields = {k: v for k, v in sub_overlay.items() if k != "variants"}
            merged = _deep_merge(merged, sub_fields)
            # Layer 3: hardware variant (last, wins).
            if hardware and hardware in variant_block:
                merged = _deep_merge(merged, variant_block[hardware])

    return merged


def load_all_model_configs(subsystem: str = "eval", hardware: Optional[str] = None) -> Dict[str, dict]:
    """Load + resolve EVERY model file into a flat {HF-name -> resolved config} dict.

    This is the back-compat bridge for the eval listener's ``load_model_registry`` /
    ``load_baseline_model_configs`` (which return a flat dict keyed by HF name).
    Models with no file are NOT included (callers fall back to patterns separately).
    """
    out: Dict[str, dict] = {}
    for org_dir in sorted(MODEL_CONFIG_DIR.iterdir()):
        if not org_dir.is_dir() or org_dir.name.startswith("_") or org_dir.name.startswith("."):
            continue
        for f in sorted(org_dir.glob("*.yaml")):
            data = _load_yaml(f)
            model_id = data.get("model")
            if not model_id:
                continue
            out[model_id] = resolve_model_config(model_id, subsystem=subsystem, hardware=hardware)
    return out
