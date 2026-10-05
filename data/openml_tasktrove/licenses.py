"""Repository-owned automatic dataset license policy."""

import tomllib
from pathlib import Path

POLICY_PATH = Path(__file__).with_name("allowed_licenses.toml")


def _normalized(value):
    return " ".join(str(value or "").strip().lower().replace("_", "-").split())


def load_policy(path=POLICY_PATH):
    entries = tomllib.loads(Path(path).read_text(encoding="utf-8"))["licenses"]
    aliases = {}
    for entry in entries:
        for value in [entry["id"], *entry.get("aliases", [])]:
            key = _normalized(value)
            if key in aliases:
                raise ValueError(f"Duplicate license alias in policy: {value}")
            aliases[key] = {
                "license_id": entry["id"],
                "license_url": entry["url"],
            }
    return aliases


def resolve_license(value, policy=None):
    return (policy or load_policy()).get(_normalized(value))
