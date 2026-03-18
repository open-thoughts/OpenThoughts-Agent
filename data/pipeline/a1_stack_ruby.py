#!/usr/bin/env python3
"""
Generate RSpec/Minitest tasks from The Stack - filters for Ruby test files.
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

RSPEC_TO_TASK_PROMPT = """You are an expert at creating coding tasks from Ruby RSpec/Minitest test code.

Given the following Ruby test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_ruby_test_file(content: str) -> bool:
    """Check if content is a Ruby test file (RSpec or Minitest)."""
    # RSpec patterns
    has_rspec_describe = 'RSpec.describe' in content or re.search(r'describe\s+[A-Z"\']', content)
    has_it_block = re.search(r'\bit\s+["\']', content) or re.search(r'\bit\s+do\b', content)
    has_expect = 'expect(' in content

    # Minitest patterns
    has_minitest = 'require "minitest' in content or "require 'minitest" in content
    has_test_class = 'class' in content and '< Minitest::Test' in content
    has_assert = 'assert_equal' in content or 'assert ' in content or 'refute' in content

    is_rspec = (has_rspec_describe or has_it_block) and has_expect
    is_minitest = has_minitest or (has_test_class and has_assert)

    if not (is_rspec or is_minitest):
        return False

    # Reject tests that require external helper files we don't have
    helper_requires = ['spec_helper', 'test_helper', 'rails_helper', 'support/', '../']
    for helper in helper_requires:
        if f"require '{helper}" in content or f'require "{helper}' in content:
            return False

    # Reject Rails tests (need full Rails environment)
    if 'ActiveRecord' in content or 'ActionController' in content or 'rails' in content.lower():
        return False

    # Reject relative requires
    if re.search(r"require_relative\s+['\"]", content):
        return False

    return True


def count_test_cases(content: str) -> int:
    """Count number of test cases."""
    it_count = len(re.findall(r'\bit\s+["\']', content))
    it_do_count = len(re.findall(r'\bit\s+do\b', content))
    def_test_count = len(re.findall(r'def\s+test_', content))
    return it_count + it_do_count + def_test_count


def filter_ruby_tests_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for Ruby test content."""
    print("Loading The Stack (Ruby subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/ruby',
        split='train',
        streaming=True
    )

    ruby_samples = []
    scanned = 0

    print(f"Scanning for Ruby test files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding Ruby test files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '')

        # Check content directly (path may be empty in The Stack)
        if is_ruby_test_file(content):
            test_count = count_test_cases(content)
            if test_count >= 2:
                ruby_samples.append({
                    'text': content,
                    'path': sample.get('path', ''),
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(ruby_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(ruby_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(ruby_samples)} Ruby test files")
    return ruby_samples


def create_ruby_dockerfile() -> str:
    """Create Ruby Dockerfile for test environment."""
    return """FROM ruby:3.2-slim

WORKDIR /app

# Install build tools required for native gem extensions, plus bash for Harbor agent
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    ruby-dev \\
    libffi-dev \\
    bash \\
    && rm -rf /var/lib/apt/lists/*

RUN gem install rspec minitest --no-document
"""


def create_test_sh() -> str:
    """Create test.sh that runs Ruby tests and reports results."""
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

# Install dependencies if Gemfile exists
if [ -f Gemfile ]; then
    bundle install --quiet 2>/dev/null || true
fi

# Create a helper that loads all .rb files from /app before running tests
cat > /tmp/load_app.rb << 'RUBY'
# Add /app to load path
$LOAD_PATH.unshift('/app')

# Require all .rb files from /app (recursively)
Dir.glob('/app/**/*.rb').sort.each do |file|
  begin
    require file
  rescue LoadError, NameError => e
    # Ignore load errors - some files may have dependencies
  end
end
RUBY

# Run tests
echo "Running Ruby tests..."
if grep -q "RSpec" /tests/test_solution.rb 2>/dev/null || grep -q "describe" /tests/test_solution.rb 2>/dev/null; then
    rspec --require /tmp/load_app.rb /tests/test_solution.rb 2>&1 | tee /logs/verifier/test_output.txt
else
    ruby -r /tmp/load_app.rb /tests/test_solution.rb 2>&1 | tee /logs/verifier/test_output.txt
fi

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
    (env_dir / "Dockerfile").write_text(create_ruby_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "test_solution.rb").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata:
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_ruby_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "stack-ruby") -> str:
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
    print("Step 1: Filtering Ruby test files from The Stack...")
    ruby_samples = filter_ruby_tests_from_stack(LIMIT)
    print(f"  -> {len(ruby_samples)} Ruby test files found")

    if not ruby_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(ruby_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": RSPEC_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_ruby_tasks(ruby_samples, task_descriptions, "stack-ruby")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-ruby")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(ruby_samples)} Ruby tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
