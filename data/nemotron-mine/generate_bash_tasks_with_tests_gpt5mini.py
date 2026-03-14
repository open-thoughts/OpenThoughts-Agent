#!/usr/bin/env python3
"""
Generate tasks from Bash verifier scripts in Nemotron using GPT-5-mini - WITH tests in instruction.
Filters for bash scripts that ARE test/verifier scripts, uses them as test.sh,
generates instructions synthetically, and INCLUDES the test code in instruction.md.

This is the GPT-5-mini version (higher quality than gpt-4o-mini).
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from nemotron_loader import load_nemotron_stream, normalize_sample, extract_code_from_sample
from data.completions import run_completions
from data.commons import create_standard_task_toml, create_standard_dockerfile, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-mini-2025-08-07"  # Upgraded from gpt-4o-mini for higher quality generation

# =============================================================================
# LLM Prompts
# =============================================================================

VERIFIER_TO_TASK_PROMPT = """You are an expert at creating coding tasks from test/verifier scripts.

Given the following bash test script (a verifier), create a task description for what needs to be implemented to make this verifier pass. The task should:
1. Describe what functionality needs to be built
2. Specify expected inputs and outputs
3. Be clear about file locations and naming (solution should be in /app/)
4. NOT reveal the exact test logic, but give enough info to implement correctly

The verifier script tests:
{{text}}

Create a task description (just the task, no preamble). The solution should be placed in /app/:"""

# =============================================================================
# Verifier Detection Patterns
# =============================================================================

VERIFIER_PATTERNS = [
    r'assert\s+', r'test\s+-[efdrsxw]', r'\[\s+-[efdrsxw]\s+',
    r'diff\s+', r'cmp\s+', r'exit\s+[01]',
    r'echo\s+["\']?(PASS|FAIL|OK|ERROR)',
    r'\$\?\s*-?(eq|ne|gt|lt)\s*[01]', r'if\s+.*;\s*then.*exit',
    r'expected.*actual|actual.*expected',
    r'assertEqual|assertEquals', r'run_test|test_.*\(\)',
]

NON_VERIFIER_PATTERNS = [
    r'apt-get\s+install', r'curl.*\|\s*bash',
    r'systemctl\s+(start|stop|enable)', r'docker\s+(build|run|push)',
    r'git\s+(clone|pull|push)', r'ssh\s+', r'scp\s+',
    r'backup|restore', r'deploy|release',
]

DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/',
    r'rm\s+-rf\s+\*',
    r':\s*\(\s*\)\s*\{',
    r'>\s*/dev/sd[a-z]',
    r'dd\s+if=.*of=/dev/',
    r'mkfs\.',
]


# =============================================================================
# Filtering Functions
# =============================================================================


def has_shebang(content: str) -> bool:
    """Check if content starts with a bash shebang."""
    first_line = content.strip().split('\n')[0] if content.strip() else ''
    return bool(re.match(r'^#!\s*/(?:usr/)?bin/(?:ba)?sh', first_line))


def count_code_lines(content: str) -> int:
    """Count non-comment, non-empty lines."""
    return sum(1 for l in content.split('\n') if l.strip() and not l.strip().startswith('#'))


def is_dangerous(content: str) -> bool:
    """Check if script contains dangerous patterns."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def is_verifier_script(content: str) -> bool:
    """Check if a bash script looks like a test/verifier script."""
    if not has_shebang(content):
        return False
    code_lines = count_code_lines(content)
    if code_lines < 5 or code_lines > 150:
        return False
    if is_dangerous(content):
        return False
    for pattern in NON_VERIFIER_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return False
    verifier_score = 0
    content_lower = content.lower()
    for pattern in VERIFIER_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            verifier_score += 1
    for keyword in ['test', 'check', 'verify', 'assert', 'expect', 'pass', 'fail']:
        if keyword in content_lower:
            verifier_score += 0.5
    return verifier_score >= 2


def adapt_verifier_for_harbor(content: str) -> str:
    """Adapt a verifier script to work with harbor format."""
    if '/logs/verifier/reward.txt' in content:
        return content

    lines = content.split('\n')
    start_idx = 1 if lines[0].startswith('#!') else 0
    body = '\n'.join('    ' + l for l in lines[start_idx:])

    return f'''#!/bin/bash
set -e
mkdir -p /logs/verifier

# Ensure reward.txt is always written, even on unexpected errors
trap 'echo "0" > /logs/verifier/reward.txt' ERR

run_verifier() {{
{body}
}}

if run_verifier; then
    echo "1" > /logs/verifier/reward.txt
    echo "Tests passed!"
    exit 0
else
    echo "0" > /logs/verifier/reward.txt
    echo "Tests failed!"
    exit 1
fi
'''


def filter_verifiers_from_nemotron(limit: int, max_scan: int = 50_000_000) -> List[Dict]:
    """Filter Nemotron for bash verifier/test scripts."""
    print(f"Loading Nemotron (Shell subset, streaming)...")
    ds = load_nemotron_stream("shell")

    verifier_samples = []
    scanned = 0

    print(f"Scanning for verifier scripts (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding verifiers")

    for raw_sample in itertools.islice(ds, max_scan):
        scanned += 1

        # CC-Code-v1: extract code blocks from web pages
        if "text" in raw_sample and "content" not in raw_sample:
            content = extract_code_from_sample(raw_sample, "shell")
            if not content:
                continue
            sample = {"content": content, "path": raw_sample.get("uuid", ""), "repo": "cc-code-v1", "size": len(content)}
        else:
            sample = normalize_sample(raw_sample)
            content = sample['content']

        if is_verifier_script(content):
            adapted_content = adapt_verifier_for_harbor(content)

            verifier_samples.append({
                'text': content,
                'adapted_text': adapted_content,
                'path': sample['path'],
                'repo': sample['repo'],
                'size': sample['size'],
                'code_lines': count_code_lines(content),
            })
            pbar.update(1)

            if len(verifier_samples) >= limit:
                break

        if scanned % 10000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(verifier_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(verifier_samples)} verifier scripts")
    return verifier_samples


# =============================================================================
# Task Generation Functions
# =============================================================================


def create_instruction_with_tests(task_description: str, test_code: str) -> str:
    """
    Create instruction.md content that includes both the task description
    and the test/verifier code.
    """
    return f"""{task_description}

## Tests

The following test script will be used to verify your solution:

```bash
{test_code}
```
"""


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    task_description: str,
    test_sh_content: str,
    original_script: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory with tests in instruction."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create environment directory with STANDARD Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_standard_dockerfile(), encoding="utf-8")

    # Create tests directory
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Write test.sh (the adapted verifier script)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(test_sh_content, encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Create instruction.md WITH the test code included
    instruction_content = create_instruction_with_tests(task_description, original_script)
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    # Create task.toml
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(create_standard_task_toml(), encoding="utf-8")

    # Create metadata.json with original script
    if metadata is None:
        metadata = {}
    metadata["original_verifier"] = original_script
    metadata["generation_model"] = MODEL
    metadata["source_dataset"] = "nemotron"
    metadata_path = task_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_bash_tasks_with_tests(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "nemotron-bash-withtests-gpt5mini",
) -> str:
    """Generate harbor-format task directories with tests in instruction."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating task directories"
    )):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "source_size": sample.get("size", 0),
            "code_lines": sample.get("code_lines", 0),
        }

        create_harbor_task_directory(
            output_dir=temp_dir,
            task_id=i,
            task_description=instruction,
            test_sh_content=sample["adapted_text"],
            original_script=sample["text"],
            dataset_prefix=dataset_prefix,
            metadata=metadata,
        )

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    """Main pipeline for generating bash verifier tasks with tests in instruction using GPT-5-mini."""

    # Step 1: Filter for verifier scripts from Nemotron
    print("Step 1: Filtering verifier scripts from Nemotron...")
    verifier_samples = filter_verifiers_from_nemotron(LIMIT)
    print(f"  -> {len(verifier_samples)} verifier scripts found")

    if not verifier_samples:
        print("\nNo verifier samples found. Exiting.")
        return

    # Step 2: Generate task instructions via LLM (GPT-5-mini)
    print(f"\nStep 2: Generating task instructions from verifiers with {MODEL}...")
    dataset = Dataset.from_list(verifier_samples)

    instruction_result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": VERIFIER_TO_TASK_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instruction_dataset = instruction_result.dataset
    instructions = instruction_dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    # Step 3: Generate harbor task directories WITH tests in instruction
    print("\nStep 3: Generating harbor task directories (with tests in instruction)...")
    task_dir = generate_bash_tasks_with_tests(
        verifier_samples, instructions, "nemotron-bash-withtests-gpt5mini"
    )
    print(f"  -> Task directory: {task_dir}")

    # Step 4: Upload to HuggingFace
    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-bash-withtests-gpt5mini")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated and uploaded {len(verifier_samples)} bash tasks (with tests) using {MODEL}!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
