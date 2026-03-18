#!/usr/bin/env python3
"""
Generate tasks from CoderEval dataset (230 real-world Python coding tasks).

CoderEval provides:
- Real coding tasks from open-source Python projects
- Function implementations with docstrings
- Human-readable task labels
- Dependency level classification (self_contained, slib_runnable, etc.)
- Tests are generated using an LLM from the actual function code

Data source: https://github.com/CoderEval/CoderEval (CoderEval4Python.json)
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import (
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 230  # Full Python dataset (230 tasks)
MODEL = "gpt-4o-mini"
DEFAULT_LANGUAGE = "python"

# GitHub raw URL for the actual CoderEval dataset
CODEREVAL_PYTHON_URL = "https://raw.githubusercontent.com/CoderEval/CoderEval/master/CoderEval4Python.json"

# LLM prompt for generating instructions and tests.
# Uses {{column_name}} syntax for template substitution via ChatMap.
COMBINED_PROMPT = """You are an expert Python developer and test engineer. Given a Python function with its name, docstring, implementation, and dependency level, you must:
1. Create a clear task description for implementing this function
2. Generate comprehensive, concrete pytest test cases that will actually pass when run against the implementation

Function name: {{func_name}}

Dependency level: {{level}}

Human description: {{human_label}}

Complete function implementation:
```python
{{code}}
```

Docstring:
{{docstring}}

Provide your response in the following format:

===INSTRUCTION===
[Write a clear, specific task description for implementing this function. Include:
- What the function should do
- Expected input/output behavior based on the docstring and implementation
- Any edge cases to handle
- Note: The solution should be placed in /app/solution.py
- The function must be importable as: from solution import FUNC_NAME]

===TESTS===
[Write comprehensive pytest test cases that would pass against the implementation above.
CRITICAL RULES:
- Import the function with: from solution import FUNC_NAME (replace FUNC_NAME with the actual function name)
- Write REAL assertions based on the actual implementation logic, NOT placeholders
- Each test must use concrete input values and expected outputs derived from the code
- If the function uses external libraries, import them in the test
- If the docstring contains examples (e.g., doctests), convert them into test cases
- Include at least 3 test cases covering: basic functionality, edge cases, and type handling
- All tests must be runnable with pytest
- Do NOT use placeholder strings like "expected_output" or "input1"
- Start with: import pytest]

Remember:
- The instruction should NOT reveal the implementation
- Tests must be CONCRETE and RUNNABLE - no placeholders
- Every assertion must use actual values that the function would return"""


def load_codereval(language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load CoderEval dataset from GitHub.
    Downloads CoderEval4Python.json directly from the CoderEval/CoderEval repository.
    """
    print(f"Loading CoderEval dataset ({language})...")

    if language.lower() != "python":
        raise ValueError(f"Only Python is currently supported, got: {language}")

    # Download the dataset from GitHub
    cache_dir = Path(__file__).parent / ".cache" / "codereval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "CoderEval4Python.json"

    if cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"Downloading from {CODEREVAL_PYTHON_URL}...")
        response = requests.get(CODEREVAL_PYTHON_URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Cache locally
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Cached to: {cache_file}")

    # The dataset has a top-level "RECORDS" key containing a list of entries
    records = data.get("RECORDS", data if isinstance(data, list) else [])
    print(f"Found {len(records)} total records")

    samples = []
    for record in records[:limit]:
        code = record.get("code", "")
        name = record.get("name", "")
        docstring = record.get("docstring", "")
        human_label = record.get("human_label", "")
        level = record.get("level", "")
        file_path = record.get("file_path", "")
        project = record.get("project", "")
        record_id = record.get("_id", "")

        if not code or not name:
            continue

        # Extract function signature (first line of code with def)
        func_match = re.search(r'(def\s+\w+\s*\([^)]*\))', code)
        func_signature = func_match.group(1) if func_match else f"def {name}()"

        samples.append({
            "id": record_id,
            "func_name": name,
            "func_signature": func_signature,
            "docstring": docstring or "(no docstring)",
            "code": code,
            "human_label": human_label or f"Implement the {name} function",
            "level": level,
            "file_path": file_path,
            "project": project,
            "language": language,
        })

    print(f"Loaded {len(samples)} tasks")
    return samples


def create_test_file_content(sample: Dict, generated_test: str = None) -> str:
    """Create test file from sample with proper imports."""
    test_code = generated_test or ""

    imports = "import pytest\nimport sys\nsys.path.insert(0, '/app')\n"

    if "import pytest" not in test_code:
        test_code = imports + "\n" + test_code
    elif "sys.path.insert" not in test_code:
        # Ensure sys.path is set even if pytest is already imported
        test_code = "import sys\nsys.path.insert(0, '/app')\n" + test_code

    return test_code


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    generated_test: str,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a CoderEval task."""
    dockerfile = get_dockerfile("python")
    test_sh = create_pytest_test_sh("/tests/test_solution.py")
    test_filename = "test_solution.py"

    test_files = {
        test_filename: create_test_file_content(sample, generated_test),
    }

    metadata = {
        "source": "codereval",
        "id": sample.get("id", ""),
        "func_name": sample.get("func_name", ""),
        "level": sample.get("level", ""),
        "project": sample.get("project", ""),
        "language": sample.get("language", "python"),
    }

    # Include solution if available (the original code)
    solution_files = None
    if sample.get("code"):
        solution_files = {"solution.py": sample["code"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=test_sh,
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    tests: List[str],
    dataset_prefix: str = "codereval",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, instruction, test) in enumerate(tqdm(
        zip(samples, instructions, tests),
        total=len(samples),
        desc="Creating tasks"
    )):
        create_harbor_task(temp_dir, i, instruction, sample, test, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def parse_combined_response(response: str) -> tuple:
    """Parse combined LLM response into instruction and test code."""
    instruction = ""
    test_code = ""

    # Try to parse based on markers
    if "===INSTRUCTION===" in response and "===TESTS===" in response:
        parts = response.split("===TESTS===")
        instruction_part = parts[0]
        test_part = parts[1] if len(parts) > 1 else ""

        # Extract instruction
        if "===INSTRUCTION===" in instruction_part:
            instruction = instruction_part.split("===INSTRUCTION===")[1].strip()
        else:
            instruction = instruction_part.strip()

        test_code = test_part.strip()
    else:
        # Fallback: try to find code blocks
        code_blocks = re.findall(r'```python\s*(.*?)```', response, re.DOTALL)
        if code_blocks:
            # Assume last code block is tests
            test_code = code_blocks[-1].strip()
        # Use the rest as instruction
        instruction = re.sub(r'```python.*?```', '', response, flags=re.DOTALL).strip()

    # Clean up instruction
    instruction = instruction.strip()
    if not instruction:
        instruction = "Implement the function as described in the docstring."

    # Clean up test code - remove markdown code fences
    if test_code.startswith("```python"):
        test_code = test_code[len("```python"):].strip()
    if test_code.startswith("```"):
        test_code = test_code[len("```"):].strip()
    if test_code.endswith("```"):
        test_code = test_code[:-3].strip()

    return instruction, test_code


def main(language: str = DEFAULT_LANGUAGE, limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating CoderEval tasks."""

    print(f"Step 1: Loading CoderEval dataset ({language})...")
    samples = load_codereval(language=language, limit=limit)
    print(f"  -> {len(samples)} tasks loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Generating task descriptions and tests using LLM...")
    # Add a unique cache-busting column to force new cache fingerprint
    import uuid
    cache_buster = str(uuid.uuid4())[:8]
    for i, sample in enumerate(samples):
        sample["_cache_id"] = f"codereval_combined_v3_{cache_buster}_{i}"

    dataset = Dataset.from_list(samples)

    # Generate combined instructions and tests in one call
    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": COMBINED_PROMPT,
            "output_column": "combined_output"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )

    combined_outputs = result.dataset["combined_output"]
    print(f"  -> Generated {len(combined_outputs)} combined outputs")

    print("\nStep 3: Parsing instructions and test cases...")
    instructions = []
    tests = []
    for output in tqdm(combined_outputs, desc="Parsing outputs"):
        instruction, test_code = parse_combined_response(output)
        instructions.append(instruction)
        tests.append(test_code)
    print(f"  -> Parsed {len(instructions)} instructions and {len(tests)} test cases")

    print("\nStep 4: Generating harbor task directories...")
    dataset_prefix = f"codereval-{language}"
    task_dir = generate_tasks(samples, instructions, tests, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 5: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} CoderEval tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from CoderEval dataset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=["python"],
                        help="Programming language (only python supported)")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit, upload=not args.no_upload)
