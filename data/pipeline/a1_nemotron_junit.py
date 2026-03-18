#!/usr/bin/env python3
"""
Generate JUnit tasks from Nemotron - filters for Java test files.
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
from data.commons import create_standard_task_toml, upload_tasks_to_hf

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 5_000
MODEL = "gpt-5-nano-2025-08-07"

JUNIT_TO_TASK_PROMPT = """You are an expert at creating coding tasks from Java JUnit test code.

Given the following JUnit test code, create a clear, specific task description that an AI agent could complete. The task should:
1. Describe what functionality needs to be implemented (not just "make the tests pass")
2. Include specific requirements that would make the tests pass
3. Be self-contained and actionable
4. The solution should be placed in /app/

Test code:
{{text}}

Create a task description (just the task, no preamble):"""


def is_junit_file(content: str) -> bool:
    """Check if content is a JUnit test file."""
    has_test_annotation = '@Test' in content
    has_junit_import = 'import org.junit' in content or 'import static org.junit' in content
    has_assertions = any(a in content for a in ['assertEquals', 'assertTrue', 'assertFalse', 'assertNotNull', 'assertThrows'])

    return has_test_annotation and (has_junit_import or has_assertions)


def count_test_methods(content: str) -> int:
    """Count number of @Test methods."""
    return len(re.findall(r'@Test', content))


def filter_junit_from_nemotron(limit: int, max_scan: int = 50_000_000) -> List[Dict]:
    """Filter Nemotron for JUnit test content."""
    print("Loading Nemotron (Java subset, streaming)...")
    ds = load_nemotron_stream("java")

    junit_samples = []
    scanned = 0

    print(f"Scanning for JUnit files (limit={limit}, max_scan={max_scan})...")
    pbar = tqdm(total=limit, desc="Finding JUnit files")

    for raw_sample in itertools.islice(ds, max_scan):
        scanned += 1

        # CC-Code-v1: extract code blocks from web pages
        if "text" in raw_sample and "content" not in raw_sample:
            content = extract_code_from_sample(raw_sample, "java")
            if not content:
                continue
            sample = {"content": content, "path": raw_sample.get("uuid", ""), "repo": "cc-code-v1", "size": len(content)}
        else:
            sample = normalize_sample(raw_sample)
            content = sample['content']

        # Check content directly
        if is_junit_file(content):
            test_count = count_test_methods(content)
            if test_count >= 2:
                junit_samples.append({
                    'text': content,
                    'path': sample['path'],
                    'repo': sample['repo'],
                    'size': sample['size'],
                    'test_count': test_count,
                })
                pbar.update(1)

                if len(junit_samples) >= limit:
                    break

        if scanned % 50000 == 0:
            pbar.set_postfix({'scanned': f'{scanned:,}', 'found': len(junit_samples)})

    pbar.close()
    print(f"Scanned {scanned:,} files, found {len(junit_samples)} JUnit files")
    return junit_samples


def create_java_dockerfile() -> str:
    """Create Java Dockerfile for test environment."""
    return """FROM eclipse-temurin:17-jdk

WORKDIR /app

# Install maven and bash for Harbor agent
RUN apt-get update && apt-get install -y maven bash && rm -rf /var/lib/apt/lists/*

# Download JUnit standalone console for manual test execution
RUN mkdir -p /junit && \\
    curl -L -o /junit/junit-platform-console-standalone.jar \\
    https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.0/junit-platform-console-standalone-1.10.0.jar
"""


def create_test_sh() -> str:
    """Create test.sh that runs JUnit and reports results."""
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

# Compile and run tests
echo "Compiling and running JUnit tests..."
if [ -f pom.xml ]; then
    timeout 300 mvn test 2>&1 | tee /logs/verifier/test_output.txt
    exit ${PIPESTATUS[0]}
else
    # Manual compilation with JUnit standalone
    JUNIT_JAR=/junit/junit-platform-console-standalone.jar

    # Compile all Java files
    mkdir -p /app/classes
    JAVA_FILES=$(find /app -maxdepth 1 -name "*.java" 2>/dev/null | tr '\n' ' ')

    echo "Compiling..."
    javac -cp "$JUNIT_JAR" -d /app/classes /tests/TestSolution.java $JAVA_FILES 2>&1 | tee /logs/verifier/compile_output.txt

    echo "Running tests..."
    timeout 300 java -jar "$JUNIT_JAR" --class-path /app/classes --scan-class-path 2>&1 | tee /logs/verifier/test_output.txt
    exit ${PIPESTATUS[0]}
fi
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
    (env_dir / "Dockerfile").write_text(create_java_dockerfile(), encoding="utf-8")

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh(), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)
    (tests_dir / "TestSolution.java").write_text(test_code, encoding="utf-8")

    (task_dir / "instruction.md").write_text(instruction_content, encoding="utf-8")
    (task_dir / "task.toml").write_text(create_standard_task_toml(), encoding="utf-8")

    if metadata is None:
        metadata = {}
    metadata["source_dataset"] = "nemotron"
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_junit_tasks(samples: List[Dict], task_descriptions: List[str], dataset_prefix: str = "nemotron-junit") -> str:
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
    print("Step 1: Filtering JUnit files from Nemotron...")
    junit_samples = filter_junit_from_nemotron(LIMIT)
    print(f"  -> {len(junit_samples)} JUnit files found")

    if not junit_samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Synthesizing tasks...")
    dataset = Dataset.from_list(junit_samples)
    result = run_completions(
        dataset, model=MODEL, map_type="chat",
        map_config={"user_message": JUNIT_TO_TASK_PROMPT, "output_column": "task_description"},
        max_requests_per_minute=500, max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    task_descriptions = result.dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_junit_tasks(junit_samples, task_descriptions, "nemotron-junit")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-junit")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(junit_samples)} JUnit tasks!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
