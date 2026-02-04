#!/usr/bin/env python3
"""
Generate Jest/Mocha tasks from The Stack - filters for JavaScript/TypeScript test files.
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

JEST_TO_TASK_PROMPT = """You are an expert at creating coding tasks from JavaScript/TypeScript test code.

Given the following Jest/Mocha test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_jest_file(content: str) -> bool:
    """Check if content is a Jest/Mocha test file."""
    # Must have test structure
    has_describe = 'describe(' in content
    has_it = 'it(' in content or 'test(' in content
    has_expect = 'expect(' in content

    if not ((has_describe or has_it) and has_expect):
        return False

    # Reject ES6 imports (needs Babel transform which we don't have)
    if re.search(r'^import\s+', content, re.MULTILINE):
        return False

    # Reject JSX syntax (needs React/Babel preset)
    if '</' in content and re.search(r'<[A-Z][a-zA-Z]*', content):
        return False

    # Reject AngularJS patterns (incompatible with Jest)
    if 'beforeEach(module(' in content or 'angular.mock' in content:
        return False

    # Reject relative imports to project modules (we don't have them)
    if re.search(r"require\(['\"]\.\.?/(?!tests)", content):
        return False

    # Reject tests requiring DOM/jsdom (browser environment)
    if 'document.' in content or 'window.' in content:
        return False

    return True


def count_test_cases(content: str) -> int:
    """Count number of test cases."""
    it_count = len(re.findall(r'\bit\s*\(', content))
    test_count = len(re.findall(r'\btest\s*\(', content))
    return it_count + test_count


def filter_jest_from_stack(limit: int, max_scan: int = 5_000_000) -> List[Dict]:
    """Filter The Stack for Jest/Mocha test content."""
    print("Loading The Stack (JavaScript subset, streaming)...")
    ds = load_dataset(
        'bigcode/the-stack',
        data_dir='data/javascript',
        split='train',
        streaming=True
    )

    jest_samples = []
    scanned = 0

    print(f"Scanning for Jest/Mocha files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding Jest/Mocha files")

    for sample in itertools.islice(ds, max_scan):
        scanned += 1
        content = sample.get('content', '')
        path = sample.get('path', '')

        # Check content directly (path may be empty in The Stack)
        if is_jest_file(content):
            test_count = count_test_cases(content)
            if test_count >= 2:  # At least 2 test cases
                jest_samples.append({
                    'text': content,
                    'path': sample.get('path', ''),
                    'repo': sample.get('repository_name', ''),
                    'size': sample.get('size', 0),
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(jest_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(jest_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(jest_samples)} Jest/Mocha files")
    return jest_samples


def create_node_dockerfile() -> str:
    """Create Node.js Dockerfile for test environment."""
    return """FROM node:20-slim

WORKDIR /app

# Install test frameworks globally
RUN npm install -g jest mocha chai

# Create minimal jest.config.js so Jest can find tests
RUN echo 'module.exports = { testEnvironment: "node", testMatch: ["**/tests/**/*.js", "**/*.test.js", "**/*.spec.js"] };' > /app/jest.config.js
"""


def create_test_sh() -> str:
    """Create test.sh that runs Jest and reports results."""
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

# Create jest.config.js at runtime to ensure correct paths
cat > /app/jest.config.js << 'EOF'
module.exports = {
  testEnvironment: "node",
  rootDir: "/app",
  testMatch: ["**/tests/**/*.js", "**/*.test.js"],
  moduleDirectories: ["node_modules", "/app", "/app/lib"]
};
EOF

# Install dependencies if package.json exists
if [ -f package.json ]; then
    npm install --silent 2>/dev/null || true
fi

# Auto-detect and install missing npm modules from test file
echo "Checking for missing dependencies..."
REQUIRES=$(grep -oP "require\\(['\"]\\K[^'\"@./][^'\"]*" /tests/test_solution.js 2>/dev/null | sort -u || true)
for pkg in $REQUIRES; do
    if [ ! -d "/app/node_modules/$pkg" ] && [ ! -d "/usr/local/lib/node_modules/$pkg" ]; then
        echo "Installing $pkg..."
        npm install --silent "$pkg" 2>/dev/null || true
    fi
done

# Create symlinks so relative requires from /tests/ can find /app/lib
if [ -d /app/lib ]; then
    ln -sf /app/lib /lib 2>/dev/null || true
fi

# Copy test file to /app/tests for Jest to find it
mkdir -p /app/tests
cp /tests/test_solution.js /app/tests/

# Run Jest with explicit config
echo "Running tests..."
timeout 300 npx jest /app/tests/test_solution.js --config /app/jest.config.js --no-coverage 2>&1 | tee /logs/verifier/test_output.txt

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
    (env_dir / "Dockerfile").write_text(create_node_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "test_solution.js").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata:
        (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_jest_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "stack-jest") -> str:
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
    print("Step 1: Filtering Jest/Mocha files from The Stack...")
    jest_samples = filter_jest_from_stack(LIMIT)
    print(f"  -> {len(jest_samples)} Jest/Mocha files found")

    if not jest_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(jest_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": JEST_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_jest_tasks(jest_samples, task_descriptions, "stack-jest")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-jest")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(jest_samples)} Jest/Mocha tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
