#!/usr/bin/env python3
"""
Shared Nemotron loading utility for all nemotron-mine generators.

Provides streaming access to two NVIDIA datasets:

1. nvidia/Nemotron-Pretraining-Code-v2 (synthetic configs):
   - Synthetic-Rewriting: Python only, actual rewritten code
   - Synthetic-Transpilation: C++ only, transpiled from Python
   Both have a 'content' column with code and a 'language' column.

2. nvidia/Nemotron-CC-Code-v1 (Common Crawl code pages):
   - 216M web pages containing code (cleaned by Phi-4)
   - Multi-language: bash, java, python, ruby, c#, c++, rust, js, etc.
   - Code is embedded in markdown code blocks (```lang ... ```)
   - We extract code blocks matching the target language

Strategy:
   - Python: Synthetic-Rewriting (direct code content)
   - C++: Synthetic-Transpilation (direct code content)
   - All other languages: CC-Code-v1 (extract from web page code blocks)
"""

import re
from collections import Counter
from datasets import load_dataset

NEMOTRON_V2_REPO = "nvidia/Nemotron-Pretraining-Code-v2"
CC_CODE_REPO = "nvidia/Nemotron-CC-Code-v1"

# Source selection: v2 synthetic for python/c++, CC-Code for everything else
V2_LANGUAGE_CONFIG = {
    "python": "Synthetic-Rewriting",
    "c++": "Synthetic-Transpilation",
}

# Mapping from our language keys to markdown code fence labels in CC-Code-v1
CC_CODE_FENCE_LABELS = {
    "shell":   ["bash", "sh", "shell"],
    "python":  ["python", "py", "python3"],
    "c++":     ["cpp", "c++", "cxx"],
    "c-sharp": ["csharp", "cs", "c#"],
    "java":    ["java"],
    "ruby":    ["ruby", "rb"],
    "rust":    ["rust", "rs"],
    "javascript": ["javascript", "js", "typescript", "ts"],
}

# Mapping for v2 language column values
V2_LANGUAGE_MAP = {
    "python": ["Python"],
    "c++":    ["C++"],
}

# All supported languages
ALL_LANGUAGES = set(CC_CODE_FENCE_LABELS.keys())


def _extract_code_blocks(text: str, fence_labels: list[str]) -> list[str]:
    """
    Extract code blocks from markdown text that match the given fence labels.

    Looks for patterns like:
        ```python
        code here
        ```

    Returns list of extracted code strings.
    """
    blocks = []
    # Build pattern: ```label\n ... ```
    labels_pattern = "|".join(re.escape(label) for label in fence_labels)
    pattern = rf"```(?:{labels_pattern})\s*\n(.*?)```"
    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if len(code) > 50:  # Skip trivially short snippets
            blocks.append(code)
    return blocks


def load_nemotron_stream(language: str):
    """
    Load Nemotron streaming dataset filtered to a specific language.

    Strategy:
    - Python: Synthetic-Rewriting from v2 (direct code, 100M samples)
    - C++: Synthetic-Transpilation from v2 (direct code, 29M samples)
    - All others: CC-Code-v1 (extract code blocks from web pages)

    Args:
        language: Language key (e.g. "shell", "python", "c++", "java", etc.)

    Returns:
        An iterable streaming dataset. For v2 configs, samples have 'content'
        and 'language' columns. For CC-Code, samples have 'text' column
        (web page with embedded code blocks).
    """
    if language not in ALL_LANGUAGES:
        raise ValueError(
            f"Unknown language key '{language}'. "
            f"Valid keys: {sorted(ALL_LANGUAGES)}"
        )

    if language in V2_LANGUAGE_CONFIG:
        # Use v2 synthetic configs (direct code content)
        config = V2_LANGUAGE_CONFIG[language]
        accepted_languages = set(V2_LANGUAGE_MAP[language])

        print(f"  Loading {NEMOTRON_V2_REPO} config='{config}' for language={language}")

        ds = load_dataset(
            NEMOTRON_V2_REPO,
            name=config,
            split="train",
            streaming=True,
        )
        ds = ds.filter(lambda sample: sample.get("language", "") in accepted_languages)
        return ds
    else:
        # Use CC-Code-v1 (web pages with code blocks)
        print(f"  Loading {CC_CODE_REPO} for language={language} (code block extraction)")

        ds = load_dataset(
            CC_CODE_REPO,
            split="train",
            streaming=True,
        )
        return ds


def normalize_sample(sample: dict, language: str = None) -> dict:
    """
    Map Nemotron columns to The Stack column names.

    For v2 synthetic configs:
        Has 'content' column directly.

    For CC-Code-v1:
        Has 'text' column with web page content.
        If language is provided, extracts the first matching code block.
        Otherwise returns the full text.

    Returns a dict with keys: content, path, repo, size
    """
    # v2 synthetic configs have 'content' directly
    if "content" in sample:
        content = sample["content"]
        return {
            "content": content,
            "path": sample.get("rel_path", sample.get("nemotron_pretraining_id", "")),
            "repo": sample.get("commit_id", sample.get("seed_source", "")),
            "size": len(content),
        }

    # CC-Code-v1 has 'text' (web page with code blocks)
    text = sample.get("text", "")

    if language and language in CC_CODE_FENCE_LABELS:
        # Extract code blocks matching the language
        blocks = _extract_code_blocks(text, CC_CODE_FENCE_LABELS[language])
        if blocks:
            # Use the longest code block as the content
            content = max(blocks, key=len)
        else:
            content = ""
    else:
        content = text

    return {
        "content": content,
        "path": sample.get("uuid", ""),
        "repo": "nemotron-cc-code-v1",
        "size": len(content),
    }


def extract_code_from_sample(sample: dict, language: str) -> str:
    """
    Extract the best code block for a given language from a CC-Code-v1 sample.
    Returns empty string if no matching code block found.
    """
    text = sample.get("text", sample.get("content", ""))
    if language not in CC_CODE_FENCE_LABELS:
        return ""
    blocks = _extract_code_blocks(text, CC_CODE_FENCE_LABELS[language])
    if blocks:
        return max(blocks, key=len)
    return ""


def discover_languages(n: int = 2_000) -> dict:
    """
    Utility: scan n samples from each Nemotron config and report languages.
    """
    import itertools

    print("=== Nemotron-Pretraining-Code-v2 configs ===")
    for config in ["Nemotron-Code-Metadata", "Synthetic-Rewriting", "Synthetic-Transpilation",
                   "Synthetic-Code-Review", "Synthetic-Question-Answering", "Synthetic-Student-Teacher"]:
        print(f"\n--- {config} (first {n:,} samples) ---")
        try:
            ds = load_dataset(NEMOTRON_V2_REPO, name=config, split="train", streaming=True)
            counter = Counter()
            has_content = False
            for sample in itertools.islice(ds, n):
                counter[sample.get("language", "<MISSING>")] += 1
                if not has_content and sample.get("content", ""):
                    has_content = True
            print(f"  Has content column: {has_content}")
            for lang, count in counter.most_common(10):
                print(f"  {lang:20s} {count:>6,}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n\n=== Nemotron-CC-Code-v1 (first {n:,} samples) ===")
    try:
        ds = load_dataset(CC_CODE_REPO, split="train", streaming=True)
        has_codeblock = 0
        lang_hits = Counter()
        for i, sample in enumerate(itertools.islice(ds, n)):
            text = sample.get("text", "")
            if "```" in text:
                has_codeblock += 1
            for lang_key, labels in CC_CODE_FENCE_LABELS.items():
                blocks = _extract_code_blocks(text, labels)
                if blocks:
                    lang_hits[lang_key] += 1
        print(f"  Has code blocks: {has_codeblock}/{n} ({has_codeblock/n*100:.1f}%)")
        for lang, count in lang_hits.most_common():
            print(f"  {lang:20s} {count:>6,}")
    except Exception as e:
        print(f"  Error: {e}")

    return {}


if __name__ == "__main__":
    discover_languages()
