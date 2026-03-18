#!/usr/bin/env python3
"""
Generate pytest tasks from The Stack - includes test code in instruction.md.
Inherits from generate_pytest_tasks.py but embeds the pytest code in the instruction.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import create_standard_dockerfile, create_standard_task_toml, upload_tasks_to_hf

# Import shared functions from generate_pytest_tasks
from generate_pytest_tasks import (
    filter_pytest_from_stack,
    create_test_sh,
    PYTEST_TO_TASK_PROMPT,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10_000
MODEL = "gpt-4o-mini"


def create_instruction_with_tests(task_description: str, pytest_code: str) -> str:
    """
    Create instruction.md content that includes both the task description
    and the pytest test code.

    Args:
        task_description: Generated task description from LLM
        pytest_code: Original pytest test code

    Returns:
        Combined instruction content
    """
    return f"""{task_description}

## Tests

The following pytest tests must pass:

```python
{pytest_code}
```
"""


def create_harbor_task_directory_with_tests(
    output_dir: Path,
    task_id: int,
    task_description: str,
    pytest_code: str,
    dataset_prefix: str,
    metadata: Optional[Dict] = None,
) -> Path:
    """
    Create a harbor-format task directory with pytest tests included in instruction.md.
    """
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create environment directory with Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text(create_standard_dockerfile(), encoding="utf-8")

    # Create tests directory with test.sh and the pytest file
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(create_test_sh("test_solution.py"), encoding="utf-8")
    os.chmod(test_sh_path, 0o755)

    # Write the original pytest code as the test file
    test_py_path = tests_dir / "test_solution.py"
    test_py_path.write_text(pytest_code, encoding="utf-8")

    # Create instruction.md WITH the test code included
    instruction_content = create_instruction_with_tests(task_description, pytest_code)
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    # Create task.toml
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(create_standard_task_toml(), encoding="utf-8")

    # Create metadata.json if provided
    if metadata is not None:
        metadata_path = task_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return task_dir


def generate_pytest_tasks_with_tests(
    pytest_samples: List[Dict],
    task_descriptions: List[str],
    dataset_prefix: str = "stack-pytest-withtests",
) -> str:
    """
    Generate harbor-format task directories with tests included in instruction.md.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, (sample, description) in enumerate(tqdm(
        zip(pytest_samples, task_descriptions),
        total=len(pytest_samples),
        desc="Creating task directories"
    )):
        metadata = {
            "source_path": sample.get("path", ""),
            "source_repo": sample.get("repo", ""),
            "source_size": sample.get("size", 0),
        }

        create_harbor_task_directory_with_tests(
            output_dir=temp_dir,
            task_id=i,
            task_description=description,
            pytest_code=sample["text"],
            dataset_prefix=dataset_prefix,
            metadata=metadata,
        )

    print(f"Generated {len(pytest_samples)} harbor tasks successfully!")
    return str(temp_dir)


def main() -> None:
    """Main pipeline for generating pytest tasks with tests in instruction."""

    # Step 1: Filter for pytest content from The Stack
    print("Step 1: Filtering pytest files from The Stack...")
    pytest_samples = filter_pytest_from_stack(LIMIT)
    print(f"  -> {len(pytest_samples)} pytest files found")

    if not pytest_samples:
        print("\nNo pytest samples found. Exiting.")
        return

    # Step 2: Synthesize tasks from pytest using run_completions
    print("\nStep 2: Synthesizing tasks from pytest code...")
    dataset = Dataset.from_list(pytest_samples)

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": PYTEST_TO_TASK_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,  # Allow partial results for files exceeding context length
    )
    tasks_dataset = result.dataset
    task_descriptions = tasks_dataset["task_description"]
    print(f"  -> Generated {len(task_descriptions)} task descriptions")

    # Step 3: Generate harbor-format task directories WITH tests in instruction
    print("\nStep 3: Generating harbor task directories (with tests in instruction)...")
    task_dir = generate_pytest_tasks_with_tests(
        pytest_samples, task_descriptions, "stack-pytest-withtests"
    )
    print(f"  -> Task directory: {task_dir}")

    # Step 4: Upload to HuggingFace
    print("\nStep 4: Uploading to HuggingFace...")
    repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_stack-pytest-withtests")
    print(f"  -> Repository: {repo_url}")

    print(f"\n{'='*60}")
    print(f"Successfully generated and uploaded {len(pytest_samples)} pytest tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
