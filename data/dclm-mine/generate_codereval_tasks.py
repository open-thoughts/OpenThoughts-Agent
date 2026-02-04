#!/usr/bin/env python3
"""
Generate tasks from CoderEval dataset (460 real-world coding tasks).

CoderEval provides:
- Real coding tasks from Python and Java projects
- Function signatures with docstrings (input)
- Complete function implementations (output)
- Tests are generated using an LLM
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
    TEST_TO_INSTRUCTION_PROMPT,
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 460  # Full dataset (230 Python + 230 Java)
MODEL = "gpt-4o-mini"
DEFAULT_LANGUAGE = "python"

COMBINED_PROMPT = """You are an expert Python developer and task designer. Given a Python function with its signature, docstring, and implementation, you need to:
1. Create a clear task description for implementing this function
2. Generate comprehensive pytest test cases

Function signature and docstring:
```python
{input}
```

Complete implementation (for reference - DO NOT reveal this in the task):
```python
{output}
```

Provide your response in the following format:

===INSTRUCTION===
[Write a clear, specific task description for implementing this function. Include:
- What the function should do
- Expected input/output behavior
- Any edge cases to handle
- Note: The solution should be placed in /app/solution.py]

===TESTS===
[Write comprehensive pytest test cases. Include:
- Basic functionality tests
- Edge cases (empty inputs, None values, boundary conditions)
- Error handling tests if applicable
- Start with necessary imports including pytest]

Remember:
- The instruction should NOT reveal the implementation
- The tests should be self-contained and runnable
- Start tests with: import pytest"""


def load_codereval(language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load CoderEval dataset from HuggingFace.
    Uses vitaleantonio/codereval-{language} which has input/output format.
    """
    print(f"Loading CoderEval dataset ({language})...")

    # Dataset mapping by language
    repo_map = {
        "python": "vitaleantonio/codereval-python",
        "java": "vitaleantonio/codereval-java",
    }

    repo_name = repo_map.get(language.lower())
    if not repo_name:
        raise ValueError(f"Unsupported language: {language}")

    try:
        ds = load_dataset(repo_name, split="train", streaming=True)
        print(f"Loaded from {repo_name}")
    except Exception as e:
        raise ValueError(f"Could not load CoderEval dataset: {e}")

    samples = []
    print(f"Collecting up to {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), desc="Loading samples"):
        sample_id = sample.get("id", "")
        input_code = sample.get("input", "")  # Function signature with docstring
        output_code = sample.get("output", "")  # Complete implementation

        if not input_code or not output_code:
            continue

        # Extract function name from the signature
        func_match = re.search(r'def\s+(\w+)\s*\(', input_code)
        func_name = func_match.group(1) if func_match else "function"

        # Extract docstring if present
        docstring_match = re.search(r'"""(.*?)"""', input_code, re.DOTALL)
        if not docstring_match:
            docstring_match = re.search(r"'''(.*?)'''", input_code, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        samples.append({
            "id": sample_id,
            "func_name": func_name,
            "docstring": docstring,
            "input": input_code,  # Function signature with docstring
            "output": output_code,  # Complete implementation (solution)
            "language": language,
        })

    print(f"Loaded {len(samples)} tasks")
    return samples


def create_test_file_content(sample: Dict, generated_test: str = None) -> str:
    """Create test file from sample."""
    language = sample.get("language", "python")
    test_code = generated_test or sample.get("test", "")

    if language == "python":
        imports = "import pytest\nimport sys\nsys.path.insert(0, '/app')\n"

        if "import pytest" not in test_code:
            test_code = imports + "\n" + test_code

        return test_code

    else:  # Java
        if "import org.junit" not in test_code:
            test_code = """import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

""" + test_code

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
    language = sample.get("language", "python")

    if language == "python":
        dockerfile = get_dockerfile("python")
        test_sh = create_pytest_test_sh("/tests/test_solution.py")
        test_filename = "test_solution.py"
        solution_ext = "py"
    else:
        dockerfile = get_dockerfile("java")
        test_sh = create_generic_test_sh(
            test_command="mvn test -q",
            setup_commands=""
        )
        test_filename = "TestSolution.java"
        solution_ext = "java"

    test_files = {
        test_filename: create_test_file_content(sample, generated_test),
    }

    metadata = {
        "source": "codereval",
        "id": sample.get("id", ""),
        "func_name": sample.get("func_name", ""),
        "language": language,
    }

    # Include solution if available
    solution_files = None
    if sample.get("output"):
        solution_files = {f"solution.{solution_ext}": sample["output"]}

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
        import re
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

    # Clean up test code
    if test_code.startswith("```python"):
        test_code = test_code[len("```python"):].strip()
    if test_code.endswith("```"):
        test_code = test_code[:-3].strip()

    return instruction, test_code


def main(language: str = DEFAULT_LANGUAGE, limit: int = LIMIT) -> None:
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
        sample["_cache_id"] = f"codereval_combined_v2_{cache_buster}_{i}"

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

    print("\nStep 5: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} CoderEval tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from CoderEval dataset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=["python", "java"],
                        help="Programming language")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit)
