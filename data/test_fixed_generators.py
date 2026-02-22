#!/usr/bin/env python3
"""
Quick test script to generate small datasets and test pass rates.
Generates 50 tasks each for key languages to verify fixes work.
"""
import os
import sys
import tempfile
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.commons import upload_tasks_to_hf

# Import generator functions from each script
from data.dclm_mine.generate_go_test_tasks import (
    filter_go_tests_from_stack,
    create_go_dockerfile,
    create_test_sh as create_go_test_sh,
    create_harbor_task_directory as create_go_task,
)
from data.dclm_mine.generate_rspec_tasks import (
    filter_ruby_tests_from_stack,
    create_ruby_dockerfile,
    create_test_sh as create_ruby_test_sh,
    create_harbor_task_directory as create_ruby_task,
)
from data.dclm_mine.generate_pytest_tasks import (
    filter_pytest_from_stack,
    create_test_sh as create_pytest_test_sh,
    create_harbor_task_directory as create_pytest_task,
)
from data.dclm_mine.generate_jest_tasks import (
    filter_jest_from_stack,
    create_node_dockerfile,
    create_test_sh as create_jest_test_sh,
    create_harbor_task_directory as create_jest_task,
)
from data.dclm_mine.generate_phpunit_tasks import (
    filter_phpunit_from_stack,
    create_php_dockerfile,
    create_test_sh as create_php_test_sh,
    create_harbor_task_directory as create_php_task,
)

from data.completions import run_completions
from data.commons import create_standard_task_toml, create_standard_dockerfile
from datasets import Dataset
from tqdm import tqdm

# Test configuration
TEST_LIMIT = 50  # Generate 50 tasks for quick testing
TEST_SUFFIX = "-v2-test"

# Prompts for task synthesis
PROMPTS = {
    "go": """You are an expert at creating coding tasks from Go test code.
Given the following Go test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):""",
    "ruby": """You are an expert at creating coding tasks from Ruby RSpec/Minitest test code.
Given the following Ruby test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):""",
    "pytest": """You are an expert at creating coding tasks from pytest test code.
Given the following pytest code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable

Pytest code:
{{text}}

Create a task description (just the task, no preamble):""",
    "jest": """You are an expert at creating coding tasks from JavaScript/TypeScript test code.
Given the following Jest/Mocha test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):""",
    "php": """You are an expert at creating coding tasks from PHP PHPUnit test code.
Given the following PHPUnit test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):""",
}


def generate_and_upload(language: str, filter_func, create_task_func, prompt: str, dataset_prefix: str):
    """Generate tasks for a language and upload to HF."""
    print(f"\n{'='*60}")
    print(f"Generating {language} tasks")
    print(f"{'='*60}")

    # Filter samples
    print(f"Step 1: Filtering {language} test files...")
    samples = filter_func(TEST_LIMIT, max_scan=500_000)
    print(f"  -> Found {len(samples)} samples")

    if not samples:
        print(f"No {language} samples found, skipping")
        return None

    # Synthesize tasks
    print(f"Step 2: Synthesizing task descriptions...")
    dataset = Dataset.from_list(samples)
    result = run_completions(
        dataset, model="gpt-4o-mini", map_type="chat",
        map_config={"user_message": prompt, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} descriptions")

    # Create task directories
    print(f"Step 3: Creating harbor task directories...")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_test_"))

    for i, (sample, description) in enumerate(tqdm(zip(samples, task_descriptions), total=len(samples))):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
        }
        create_task_func(temp_dir, i, description, sample["text"], dataset_prefix, metadata)

    # Upload
    print(f"Step 4: Uploading to HuggingFace...")
    repo_id = f"DCAgent/exp_rpt_{dataset_prefix}{TEST_SUFFIX}"
    repo_url = upload_tasks_to_hf(str(temp_dir), repo_id)
    print(f"  -> Uploaded to: {repo_url}")

    return repo_id


def main():
    """Generate test datasets for all key languages."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["go", "ruby", "pytest", "jest", "php"],
                       help="Languages to generate (default: all)")
    args = parser.parse_args()

    generators = {
        "go": (filter_go_tests_from_stack, create_go_task, PROMPTS["go"], "stack-go"),
        "ruby": (filter_ruby_tests_from_stack, create_ruby_task, PROMPTS["ruby"], "stack-ruby"),
        "pytest": (filter_pytest_from_stack, create_pytest_task, PROMPTS["pytest"], "stack-pytest"),
        "jest": (filter_jest_from_stack, create_jest_task, PROMPTS["jest"], "stack-jest"),
        "php": (filter_phpunit_from_stack, create_php_task, PROMPTS["php"], "stack-php"),
    }

    results = []
    for lang in args.languages:
        if lang not in generators:
            print(f"Unknown language: {lang}")
            continue

        filter_func, create_func, prompt, prefix = generators[lang]
        try:
            repo_id = generate_and_upload(lang, filter_func, create_func, prompt, prefix)
            if repo_id:
                results.append((lang, repo_id, "success"))
        except Exception as e:
            print(f"Error generating {lang}: {e}")
            import traceback
            traceback.print_exc()
            results.append((lang, None, str(e)))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for lang, repo_id, status in results:
        if status == "success":
            print(f"OK: {lang} -> {repo_id}")
        else:
            print(f"FAIL: {lang} - {status}")

    # Print datasets for pass rate testing
    print(f"\n{'='*60}")
    print("Datasets ready for pass rate testing:")
    print(f"{'='*60}")
    for lang, repo_id, status in results:
        if status == "success":
            print(f"  {repo_id}")


if __name__ == "__main__":
    main()
