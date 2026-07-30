"""Redact credential-shaped text before publishing models or trace datasets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    """One replaced credential shape and its source location."""

    shape: str
    location: str
    line: int | None = None


REDACTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pem_private_key", re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.DOTALL)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
)


def redact(text: str, *, location: str = "text") -> tuple[str, list[Finding]]:
    """Replace known credential shapes with stable markers and report every replacement."""
    findings: list[Finding] = []
    for shape, pattern in REDACTIONS:
        def replace(match: re.Match[str]) -> str:
            findings.append(Finding(shape, location, text.count("\n", 0, match.start()) + 1))
            return f"<redacted:{shape}>"
        text = pattern.sub(replace, text)
    return text, findings


def redact_record(record: dict[str, Any], *, _location: str = "") -> tuple[dict[str, Any], list[Finding]]:
    """Recursively redact a JSON-compatible trace record without mutating it."""
    findings: list[Finding] = []
    def visit(value: Any, location: str) -> Any:
        if isinstance(value, str):
            cleaned, found = redact(value, location=location)
            findings.extend(found)
            return cleaned
        if isinstance(value, list):
            return [visit(item, f"{location}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: visit(item, f"{location}.{key}" if location else key) for key, item in value.items()}
        return value
    return visit(record, _location), findings


def redact_tree(root: Path) -> list[Finding]:
    """Redact UTF-8 text files in a staging tree in place and return findings."""
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        cleaned, found = redact(text, location=str(path.relative_to(root)))
        if found:
            path.write_text(cleaned, encoding="utf-8")
            findings.extend(found)
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Redact credential-shaped text in a staging tree.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    findings = redact_tree(args.root)
    print(json.dumps({"findings": [asdict(finding) for finding in findings]}, indent=2))


if __name__ == "__main__":
    main()
