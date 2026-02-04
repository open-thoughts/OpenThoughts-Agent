#!/usr/bin/env python3
"""
Generate PHPUnit tasks from The Stack - filters for PHP test files.
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
MODEL = "gpt-5-nano-2025-08-07"

PHPUNIT_TO_TASK_PROMPT = """You are an expert at creating coding tasks from PHP PHPUnit test code.

Given the following PHPUnit test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_phpunit_file(content: str) -> bool:
    """Check if content is a PHPUnit test file."""
    has_phpunit_import = 'PHPUnit' in content
    has_test_case = 'extends TestCase' in content or 'extends PHPUnit' in content
    has_test_annotation = '@test' in content
    has_test_method = bool(re.search(r'function\s+test\w+', content))
    has_assertions = any(a in content for a in ['assertEquals', 'assertTrue', 'assertFalse', 'assertSame', 'assertNull'])

    return (has_phpunit_import or has_test_case) and (has_test_method or has_test_annotation) and has_assertions


def count_test_methods(content: str) -> int:
    """Count number of test methods."""
    test_funcs = len(re.findall(r'function\s+test\w+', content))
    test_annotations = len(re.findall(r'@test', content))
    return max(test_funcs, test_annotations)


def filter_phpunit_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for PHPUnit test content."""
    print("Loading The Stack (PHP subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/php',
        split='train',
        streaming=True
    )

    php_samples = []
    scanned = 0

    print(f"Scanning for PHPUnit files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding PHPUnit files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '')

        # Check content directly (path may be empty in The Stack)
        if is_phpunit_file(content):
            test_count = count_test_methods(content)
            if test_count >= 2:
                php_samples.append({
                    'text': content,
                    'path': sample.get('path', ''),
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(php_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(php_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(php_samples)} PHPUnit files")
    return php_samples


def create_php_dockerfile() -> str:
    """Create PHP Dockerfile for test environment."""
    return r"""FROM php:8.2-cli

WORKDIR /app

# Install required tools including bash for Harbor agent
RUN apt-get update && apt-get install -y unzip git bash && rm -rf /var/lib/apt/lists/*

# Install Composer
RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

# Install PHPUnit globally
RUN composer global require phpunit/phpunit
ENV PATH="${PATH}:/root/.composer/vendor/bin"

# Create autoload file for test classes using heredoc to avoid escaping issues
RUN cat > /app/autoload.php << 'AUTOLOAD'
<?php
spl_autoload_register(function($class) {
    $file = "/app/" . str_replace("\\", "/", $class) . ".php";
    if (file_exists($file)) {
        require $file;
    }
});
AUTOLOAD
"""


def create_test_sh() -> str:
    """Create test.sh that runs PHPUnit and reports results."""
    return '''#!/bin/bash
set -e

mkdir -p /logs/verifier

cleanup() {
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}
trap cleanup EXIT

cd /app

# Install dependencies if composer.json exists
if [ -f composer.json ]; then
    composer install --quiet 2>/dev/null || true
fi

# Create phpunit.xml for better test discovery
cat > /app/phpunit.xml << 'EOF'
<?xml version="1.0"?>
<phpunit bootstrap="/app/autoload.php">
  <testsuites>
    <testsuite name="Tests">
      <directory>/tests</directory>
    </testsuite>
  </testsuites>
</phpunit>
EOF

# Run PHPUnit with configuration
echo "Running PHPUnit tests..."
timeout 300 phpunit --configuration /app/phpunit.xml 2>&1 | tee /logs/verifier/test_output.txt

exit ${PIPESTATUS[0]}
'''


def create_harbor_task_directory(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    test_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """Create a harbor-format task directory."""
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_php_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "TestSolution.php").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata:
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_php_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "stack-php") -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, description) in enumerate(tqdm(zip(samples, task_descriptions), total=len(samples), desc="Creating tasks")):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "test_count": sample.get("test_count", 0),
        }
        create_harbor_task_directory(temp_dir, i, description, sample["text"], dataset_prefix, metadata)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main() -> None:
    """Main pipeline."""
    print("Step 1: Filtering PHPUnit files from The Stack...")
    php_samples = filter_phpunit_from_stack(LIMIT)
    print(f"  -> {len(php_samples)} PHPUnit files found")

    if not php_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(php_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": PHPUNIT_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_php_tasks(php_samples, task_descriptions, "stack-php")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-php")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(php_samples)} PHP tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
