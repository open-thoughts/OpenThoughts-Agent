#!/usr/bin/env python3
"""
Generate tasks from Methods2Test dataset (780K Java focal-test pairs).

Methods2Test is a Microsoft dataset containing Java method-test pairs.
Each entry has:
- focal_method: The Java method being tested
- test_method: The JUnit test for that method
- focal_class: The class containing the focal method
- test_class: The test class
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import (
    FOCAL_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_junit_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"


def load_methods2test(limit: int = LIMIT) -> List[Dict]:
    """
    Load Methods2Test dataset from HuggingFace.

    The dataset is available at microsoft/methods2test on HuggingFace.
    """
    print("Loading Methods2Test dataset...")

    # Try multiple possible sources (updated Jan 2026)
    repo_names = [
        "jitx/Methods2Test_java_unit_test_code",  # Current working source
        "microsoft/methods2test",
        "methods2test/methods2test",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="train", streaming=True, trust_remote_code=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        raise ValueError("Could not load Methods2Test dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Extract focal and test methods
        # Dataset fields: target, src_fm, src_fm_fc, src_fm_fc_co, src_fm_fc_ms, src_fm_fc_ms_ff
        focal = (
            sample.get("src_fm_fc_ms") or  # source with focal method, class, method signature
            sample.get("src_fm_fc") or     # source with focal method and class
            sample.get("src_fm") or        # source with focal method
            sample.get("focal_method") or
            sample.get("method") or
            sample.get("focal", "")
        )
        test = (
            sample.get("target") or        # the test code
            sample.get("test_method") or
            sample.get("test") or
            sample.get("test_code", "")
        )

        if not focal or not test:
            continue

        samples.append({
            "focal_function": focal,
            "test_function": test,
            "focal_class": sample.get("focal_class", sample.get("class_name", "")),
            "test_class": sample.get("test_class", ""),
            "repo": sample.get("repo_name", sample.get("repository", "")),
            "path": sample.get("file_path", sample.get("path", "")),
            "imports": sample.get("imports", ""),
            "package": sample.get("package", ""),
        })

    print(f"Loaded {len(samples)} focal-test pairs")
    return samples


def create_test_file_content(sample: Dict) -> str:
    """Create JUnit test file content from sample."""
    test_code = sample.get("test_function", "")
    test_class = sample.get("test_class", "")
    imports = sample.get("imports", "")
    package = sample.get("package", "")

    # Build the test file
    content_parts = []

    # Add package if present
    if package:
        content_parts.append(f"package {package};")
        content_parts.append("")

    # Add standard JUnit imports
    content_parts.append("import org.junit.jupiter.api.*;")
    content_parts.append("import static org.junit.jupiter.api.Assertions.*;")

    # Add any additional imports from the sample
    if imports:
        content_parts.append(imports)

    content_parts.append("")

    # Add test class
    if test_class and "class" in test_class:
        content_parts.append(test_class)
    else:
        # Wrap test method in a class
        content_parts.append("public class TestSolution {")
        content_parts.append("    " + test_code.replace("\n", "\n    "))
        content_parts.append("}")

    return "\n".join(content_parts)


def create_maven_pom() -> str:
    """Create a minimal pom.xml for JUnit tests."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.test</groupId>
    <artifactId>solution</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <sourceDirectory>/app/src/main/java</sourceDirectory>
        <testSourceDirectory>/tests</testSourceDirectory>
    </build>
</project>
"""


def create_java_test_sh() -> str:
    """Create test.sh for Java/Maven tests."""
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

# Create Maven project structure if needed
mkdir -p src/main/java

# Copy pom.xml if not exists
if [ ! -f pom.xml ]; then
    cp /tests/pom.xml .
fi

echo "Compiling and running tests..."
mvn test -q 2>&1 | tee /logs/verifier/test_output.txt

exit ${PIPESTATUS[0]}
'''


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a Methods2Test sample."""
    dockerfile = get_dockerfile("java")

    test_files = {
        "TestSolution.java": create_test_file_content(sample),
        "pom.xml": create_maven_pom(),
    }

    metadata = {
        "source": "methods2test",
        "repo": sample.get("repo", ""),
        "path": sample.get("path", ""),
        "package": sample.get("package", ""),
    }

    # Include focal function as solution hint
    solution_files = None
    if sample.get("focal_function"):
        solution_files = {"Solution.java": sample["focal_function"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_java_test_sh(),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "methods2test",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction) in enumerate(tqdm(
        zip(samples, instructions),
        total=len(samples),
        desc="Creating tasks"
    )):
        create_harbor_task(temp_dir, i, instruction, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(limit: int = LIMIT) -> None:
    """Main pipeline for generating Methods2Test tasks."""

    print("Step 1: Loading Methods2Test dataset...")
    samples = load_methods2test(limit=limit)
    print(f"  -> {len(samples)} focal-test pairs loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating task descriptions from focal functions...")
    dataset = Dataset.from_list(samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": FOCAL_TO_INSTRUCTION_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Generated {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    task_dir = generate_tasks(samples, instructions, "methods2test")
    print(f"  -> Task directory: {task_dir}")

    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_methods2test")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} Methods2Test tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from Methods2Test dataset")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")

    args = parser.parse_args()
    main(limit=args.limit)
