#!/usr/bin/env python3
"""
Generate tasks from Dockerfiles in The Stack using GPT-4o.
The Dockerfile becomes the verifier ENVIRONMENT, and LLM generates a task + tests
that make sense for that environment.

This is the GPT-4o version (higher quality than gpt-4o-mini).
"""

import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-5-mini-2025-08-07"  # Upgraded from gpt-4o-mini for higher quality generation

# =============================================================================
# LLM Prompts
# =============================================================================

DOCKERFILE_INSTRUCTION_PROMPT = """You are an expert at creating coding tasks.

Given the following Dockerfile that sets up a development environment, create a coding task that an AI agent could complete IN this environment. The task should:
1. Make use of the tools/languages/frameworks installed in this Dockerfile
2. Be a realistic, self-contained coding task (e.g., "Write a Python script that...", "Create a web server that...")
3. Have clear requirements and expected behavior
4. NOT be about Docker or modifying the Dockerfile itself

The agent's solution should be placed in /app/ directory.

Dockerfile (this is the environment the task will run in):
{{text}}

Create a task description that makes sense for this environment (just the task, no preamble):"""

DOCKERFILE_TEST_PROMPT = """You are an expert at creating test scripts.

Given the following task description and the Dockerfile environment it runs in, create a test.sh script that verifies the solution works correctly.

The test script should:
1. Run the solution (in /app/ directory)
2. Verify it produces correct outputs
3. Write "1" to /logs/verifier/reward.txt if tests pass
4. Write "0" to /logs/verifier/reward.txt if tests fail
5. Start with #!/bin/bash and create /logs/verifier directory

Task description:
{{task_description}}

Dockerfile environment:
{{text}}

Output only the shell script code (no markdown fences):"""

# =============================================================================
# Filtering Functions
# =============================================================================


def calculate_dockerfile_complexity(content: str) -> int:
    """Calculate complexity score for a Dockerfile."""
    score = 0

    # Multi-stage build
    from_count = len(re.findall(r'^FROM\s+', content, re.MULTILINE | re.IGNORECASE))
    if from_count > 1:
        score += 2

    # Package managers
    if re.search(r'apt-get\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'pip\s+install|pip3\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'npm\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'yum\s+install', content, re.IGNORECASE):
        score += 1
    if re.search(r'apk\s+add', content, re.IGNORECASE):
        score += 1

    # Instructions
    if re.search(r'^COPY\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1
    if re.search(r'^EXPOSE\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1
    if re.search(r'^ENV\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1
    if re.search(r'^WORKDIR\s+', content, re.MULTILINE | re.IGNORECASE):
        score += 1

    return score


def extract_environment_info(content: str) -> Dict:
    """Extract useful info about what's installed in the Dockerfile."""
    info = {
        "base_image": "",
        "languages": [],
        "frameworks": [],
        "tools": [],
    }

    # Extract base image
    match = re.search(r'^FROM\s+(\S+)', content, re.MULTILINE | re.IGNORECASE)
    if match:
        info["base_image"] = match.group(1)

    # Detect languages
    if re.search(r'python|pip', content, re.IGNORECASE):
        info["languages"].append("python")
    if re.search(r'node|npm|yarn', content, re.IGNORECASE):
        info["languages"].append("nodejs")
    if re.search(r'ruby|gem|bundler', content, re.IGNORECASE):
        info["languages"].append("ruby")
    if re.search(r'golang|go\s+build', content, re.IGNORECASE):
        info["languages"].append("go")
    if re.search(r'rustc|cargo', content, re.IGNORECASE):
        info["languages"].append("rust")
    if re.search(r'java|maven|gradle', content, re.IGNORECASE):
        info["languages"].append("java")

    # Detect frameworks
    if re.search(r'flask', content, re.IGNORECASE):
        info["frameworks"].append("flask")
    if re.search(r'django', content, re.IGNORECASE):
        info["frameworks"].append("django")
    if re.search(r'express', content, re.IGNORECASE):
        info["frameworks"].append("express")
    if re.search(r'react', content, re.IGNORECASE):
        info["frameworks"].append("react")
    if re.search(r'rails', content, re.IGNORECASE):
        info["frameworks"].append("rails")

    return info


def is_valid_dockerfile(content: str) -> bool:
    """Check if content is a valid Dockerfile for task generation."""
    # Must have FROM instruction
    if not re.search(r'^FROM\s+', content, re.MULTILINE | re.IGNORECASE):
        return False

    # Count non-comment, non-empty lines
    lines = [l.strip() for l in content.split('\n')
             if l.strip() and not l.strip().startswith('#')]
    if len(lines) < 5:
        return False

    # Should install something useful (not just a bare image)
    has_install = re.search(r'apt-get|pip|npm|yum|apk|gem|cargo', content, re.IGNORECASE)
    if not has_install:
        return False

    return True


def filter_dockerfiles_from_stack(limit: int, max_scan: int = 500_000, min_complexity: int = 3) -> List[Dict]:
    """Filter The Stack for Dockerfile content."""
    print(f"Loading The Stack (Dockerfile subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/dockerfile',
        split='train',
        streaming=True
    )

    dockerfile_samples = []
    scanned = 0

    print(f"Scanning for Dockerfiles (limit={limit}, max_scan={max_scan}, min_complexity={min_complexity})...")
    pbar = tqdm(total=limit, desc="Finding Dockerfiles")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')

        if is_valid_dockerfile(content):
            complexity = calculate_dockerfile_complexity(content)
            if complexity >= min_complexity:
                env_info = extract_environment_info(content)
                dockerfile_samples.append({
                    'text': content,
                    'path': sample.get('path', ''),
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'complexity': complexity,
                    'env_info': env_info,
                })
                pbar.update(1)

                if len(dockerfile_samples) >= limit:
                    break

        if scanned % 10000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(dockerfile_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(dockerfile_samples)} valid Dockerfiles")
    return dockerfile_samples


# =============================================================================
# Task Generation Functions
# =============================================================================


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    dockerfile_content: str,
    test_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create environment directory - USE THE ORIGINAL DOCKERFILE as the environment
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

    # Create tests directory
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    # Write test.sh (LLM-generated)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(test_code, encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Create instruction.md
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    # Create task.toml
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(create_standard_task_toml(), encoding="utf-8")

    # Create metadata.json
    if metadata is None:
        metadata = {}
    metadata["generation_model"] = MODEL
    metadata_path = task_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_dockerfile_tasks(
    samples: List[Dict],
    instructions: List[str],
    tests: List[str],
    dataset_prefix: str = "stack-dockerfile-gpt4o",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction, test_code) in enumerate(tqdm(
        zip(samples, instructions, tests),
        total=len(samples),
        desc="Creating task directories"
    )):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "source_size": sample.get("size", 0),
            "complexity_score": sample.get("complexity", 0),
            "env_info": sample.get("env_info", {}),
        }

        create_harbor_task_directory(
            output_dir=temp_dir,
            task_id=i,
            instruction_content=instruction,
            dockerfile_content=sample["text"],
            test_code=test_code,
            dataset_prefix=dataset_prefix,
            metadata=metadata,
        )

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    """Main pipeline for generating Dockerfile-environment tasks with GPT-4o."""

    # Step 1: Filter Dockerfiles from The Stack
    print("Step 1: Filtering Dockerfiles from The Stack...")
    dockerfile_samples = filter_dockerfiles_from_stack(LIMIT, min_complexity=3)
    print(f"  -> {len(dockerfile_samples)} Dockerfiles found")

    if not dockerfile_samples:
        print("\nNo Dockerfile samples found. Exiting.")
        return

    # Step 2: Generate task instructions via LLM (GPT-4o)
    print(f"\nStep 2: Generating task instructions with {MODEL}...")
    dataset = Dataset.from_list(dockerfile_samples)

    instruction_result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": DOCKERFILE_INSTRUCTION_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instruction_dataset = instruction_result.dataset
    print(f"  -> Generated {len(instruction_dataset)} task descriptions")

    # Step 3: Generate test.sh via LLM (GPT-4o)
    print(f"\nStep 3: Generating test scripts with {MODEL}...")
    test_result = run_completions(
        instruction_dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": DOCKERFILE_TEST_PROMPT,
            "output_column": "test_code"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    final_dataset = test_result.dataset

    instructions = final_dataset["task_description"]
    tests = final_dataset["test_code"]
    print(f"  -> Generated {len(tests)} test files")

    # Step 4: Generate harbor task directories
    print("\nStep 4: Generating harbor task directories...")
    task_dir = generate_dockerfile_tasks(
        dockerfile_samples, instructions, tests, "stack-dockerfile-gpt4o"
    )
    print(f"  -> Task directory: {task_dir}")

    # Step 5: Upload to HuggingFace
    print("\nStep 5: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-dockerfile-gpt4o")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated and uploaded {len(dockerfile_samples)} tasks with {MODEL}!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
