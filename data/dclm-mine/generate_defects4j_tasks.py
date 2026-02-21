#!/usr/bin/env python3
"""
Generate tasks from Defects4J dataset (467 real-world Java bugs).

Uses the real Defects4J bug data from rufimelo/defects4j on HuggingFace.
Each entry provides:
- bug_id: e.g. "Compress-35", "Lang-5"
- func_before: The buggy Java function
- func_after: The fixed Java function

Since the HF dataset only has function-level diffs (no full project or tests),
we wrap each bug in a self-contained Solution.java class and use the LLM to
generate JUnit 5 test cases that fail on the buggy code and pass on the fixed code.
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
from data.commons import (
    create_harbor_task_directory_generic,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"

# Custom Dockerfile: JDK + JUnit standalone jar (no Maven needed)
DEFECTS4J_DOCKERFILE = """FROM eclipse-temurin:17-jdk-jammy

WORKDIR /app

RUN apt-get update && apt-get install -y bash curl && rm -rf /var/lib/apt/lists/*

# Download JUnit 5 platform console standalone jar
RUN mkdir -p /junit && \\
    curl -sL https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.0/junit-platform-console-standalone-1.10.0.jar \\
    -o /junit/junit-platform-console-standalone.jar

RUN mkdir -p /logs/verifier /workspace /tests /app/classes
"""

# =============================================================================
# LLM Prompt - only used to generate test cases for the REAL bugs
# =============================================================================

GENERATE_TESTS_PROMPT = """You are an expert Java developer. Given a buggy Java function and its fixed version from the Defects4J benchmark, generate:

1. A complete, self-contained `Solution.java` file that wraps the buggy function in a class named `Solution`. Add any necessary imports, helper methods, or inner classes so it compiles standalone. Do NOT use a package statement.

2. A complete JUnit 5 test file `TestSolution.java` with at least 4 @Test methods that:
   - FAIL when run against the buggy version
   - PASS when run against the fixed version
   - Test edge cases related to the bug

**Bug ID:** {{bug_id}}
**Project:** {{project}}

**Buggy function:**
```java
{{func_before}}
```

**Fixed function:**
```java
{{func_after}}
```

Output in EXACTLY this format:

===SOLUTION_JAVA===
[Complete Solution.java wrapping the BUGGY function. No package statement. Must compile with javac.]

===SOLUTION_FIXED===
[Complete Solution.java wrapping the FIXED function. No package statement. Must compile with javac.]

===TEST_JAVA===
[Complete TestSolution.java with JUnit 5 tests. No package statement. Import org.junit.jupiter.api.* and static Assertions.*. Tests must FAIL on buggy and PASS on fixed.]

===INSTRUCTION===
[A markdown task description starting with "# Java Bug Repair Task" that:
- Describes what the function should do
- Explains the symptoms of the bug (wrong output, exception, etc.)
- Tells the agent the buggy file is at /app/Solution.java
- Does NOT reveal the fix]"""


# =============================================================================
# Data loading
# =============================================================================

def load_defects4j(limit: int = LIMIT) -> List[Dict]:
    """Load real Defects4J bugs from HuggingFace."""
    print("Loading Defects4J dataset from rufimelo/defects4j...")

    repo_names = [
        ("rufimelo/defects4j", "train"),
        ("CoQuIR/Defects4J", "test"),
    ]

    ds = None
    for repo_name, split in repo_names:
        try:
            ds = load_dataset(repo_name, split=split, streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        raise ValueError("Could not load Defects4J dataset from any source")

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        bug_id = sample.get("bug_id", sample.get("id", ""))
        func_before = sample.get("func_before", sample.get("buggy_code", ""))
        func_after = sample.get("func_after", sample.get("fixed_code", ""))

        if not func_before:
            continue

        # Extract project from bug_id (e.g., "Compress-35" -> "Compress")
        project = ""
        if "-" in str(bug_id):
            project = str(bug_id).rsplit("-", 1)[0]

        samples.append({
            "bug_id": bug_id,
            "project": project,
            "func_before": func_before,
            "func_after": func_after,
        })

    print(f"Loaded {len(samples)} real Defects4J bugs")
    return samples


# =============================================================================
# Parsing helpers
# =============================================================================

def _extract_section(text: str, marker: str) -> str:
    """Extract content after ===MARKER=== up to the next === or end."""
    pattern = rf"==={re.escape(marker)}===\s*\n(.*?)(?=\n===\w|$)"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    code = code.strip()
    if code.startswith("```"):
        first_nl = code.find("\n")
        if first_nl != -1:
            code = code[first_nl + 1:]
    if code.endswith("```"):
        code = code[:-3].rstrip()
    return code.strip()


def parse_generated_tests(response: str) -> Optional[Dict]:
    """Parse LLM response for test generation."""
    solution_java = _strip_code_fences(_extract_section(response, "SOLUTION_JAVA"))
    solution_fixed = _strip_code_fences(_extract_section(response, "SOLUTION_FIXED"))
    test_java = _strip_code_fences(_extract_section(response, "TEST_JAVA"))
    instruction = _extract_section(response, "INSTRUCTION")

    if not solution_java or not test_java:
        return None

    # Basic validation
    if "class Solution" not in solution_java and "class " not in solution_java:
        return None
    if "@Test" not in test_java:
        return None

    return {
        "solution_buggy": solution_java,
        "solution_fixed": solution_fixed or solution_java,
        "test_code": test_java,
        "instruction": instruction,
    }


# =============================================================================
# Harbor task creation
# =============================================================================

def create_defects4j_test_sh() -> str:
    """Create test.sh that compiles and runs JUnit tests."""
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

# If the agent hasn't placed a Solution.java yet, copy the initial buggy one
if [ ! -f /app/Solution.java ]; then
    cp /tests/initial/Solution.java /app/Solution.java
fi

echo "=== Compiling solution and tests ==="
mkdir -p /app/classes
javac -cp /junit/junit-platform-console-standalone.jar:/app:/tests \\
    /app/Solution.java /tests/TestSolution.java -d /app/classes 2>&1
if [ $? -ne 0 ]; then
    echo "COMPILATION FAILED"
    exit 1
fi

echo "=== Running JUnit tests ==="
java -jar /junit/junit-platform-console-standalone.jar \\
    --class-path /app/classes \\
    --scan-class-path 2>&1 | tee /logs/verifier/test_output.txt
exit_code=${PIPESTATUS[0]}

if [ $exit_code -ne 0 ]; then
    echo "TESTS FAILED (exit code $exit_code)"
    exit 1
fi

echo "ALL TESTS PASSED"
exit 0
'''


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a Defects4J bug."""
    test_files = {
        "TestSolution.java": sample["test_code"],
        "initial/Solution.java": sample["solution_buggy"],
    }

    metadata = {
        "source": "defects4j",
        "bug_id": sample.get("bug_id", ""),
        "project": sample.get("project", ""),
    }

    solution_files = None
    if sample.get("solution_fixed"):
        solution_files = {"Solution.java": sample["solution_fixed"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=sample.get("instruction", "Fix the bug in /app/Solution.java"),
        dockerfile=DEFECTS4J_DOCKERFILE,
        test_sh=create_defects4j_test_sh(),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    dataset_prefix: str = "defects4j",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    created = 0
    skipped = 0
    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        if not sample.get("solution_buggy") or not sample.get("test_code"):
            skipped += 1
            continue
        create_harbor_task(temp_dir, created, sample, dataset_prefix)
        created += 1

    print(f"Generated {created} harbor tasks ({skipped} skipped)")
    return str(temp_dir)


# =============================================================================
# Main pipeline
# =============================================================================

def main(limit: int = LIMIT, upload: bool = True) -> Optional[str]:
    """Main pipeline for generating Defects4J tasks.

    Uses real bugs from rufimelo/defects4j. LLM is only used to:
    1. Wrap the function snippets into compilable Solution.java files
    2. Generate JUnit test cases that expose the bug
    """

    # -- Step 1: Load real Defects4J bugs -------------------------------------
    print("Step 1: Loading real Defects4J bugs...")
    raw_samples = load_defects4j(limit=limit)
    print(f"  -> {len(raw_samples)} real bugs loaded")

    if not raw_samples:
        print("\nNo samples found. Exiting.")
        return None

    # -- Step 2: Generate tests via LLM (wrapping real bugs) ------------------
    print(f"\nStep 2: Generating test cases for real bugs via {MODEL}...")
    print("  (LLM wraps real bug functions into compilable files + generates JUnit tests)")
    dataset = Dataset.from_list(raw_samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": GENERATE_TESTS_PROMPT,
            "output_column": "llm_output",
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )

    llm_outputs = result.dataset["llm_output"]
    print(f"  -> Received {len(llm_outputs)} LLM responses")

    # -- Step 3: Parse outputs ------------------------------------------------
    print("\nStep 3: Parsing LLM outputs...")
    samples = []
    parse_failures = 0
    for i, (raw, output) in enumerate(zip(raw_samples, llm_outputs)):
        parsed = parse_generated_tests(output)
        if parsed is None:
            parse_failures += 1
            continue
        # Carry over real bug metadata
        parsed["bug_id"] = raw["bug_id"]
        parsed["project"] = raw["project"]
        samples.append(parsed)

    print(f"  -> Successfully parsed {len(samples)} tasks ({parse_failures} parse failures)")

    if not samples:
        print("\nNo valid samples after parsing. Exiting.")
        return None

    # -- Step 4: Create harbor task directories --------------------------------
    print(f"\nStep 4: Generating harbor task directories...")
    task_dir = generate_tasks(samples, "defects4j")
    print(f"  -> Task directory: {task_dir}")

    # -- Step 5: Upload --------------------------------------------------------
    if upload:
        print("\nStep 5: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_defects4j")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} Defects4J tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate tasks from real Defects4J bugs"
    )
    parser.add_argument(
        "--limit", type=int, default=LIMIT,
        help="Maximum bugs to process"
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip HuggingFace upload"
    )

    args = parser.parse_args()
    main(limit=args.limit, upload=not args.no_upload)
