#!/usr/bin/env python3
"""
Generate tasks from Exercism dataset (~15K exercises across 74 language tracks).

Exercism provides:
- Exercise instructions (markdown)
- Test files (language-specific unit tests)
- Example solutions
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

from data.commons import (
    create_harbor_task_directory_generic,
    create_pytest_test_sh,
    create_generic_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 5000
DEFAULT_LANGUAGE = "python"

# Language-specific test configurations
LANGUAGE_CONFIG = {
    "python": {
        "dockerfile": "python",
        "test_file": "test_solution.py",
        "test_cmd": "pytest /tests/test_solution.py -v --tb=short",
        "setup_cmd": "pip3 install --quiet pytest 2>/dev/null || true",
    },
    "javascript": {
        "dockerfile": "node",
        "test_file": "test_solution.js",
        "test_cmd": "npx jest /tests/test_solution.js --verbose",
        "setup_cmd": "npm install --silent jest 2>/dev/null || true",
    },
    "go": {
        "dockerfile": "go",
        "test_file": "solution_test.go",
        "test_cmd": "go test -v /tests/...",
        "setup_cmd": "",
    },
    "java": {
        "dockerfile": "java",
        "test_file": "TestSolution.java",
        "test_cmd": "mvn test -q",
        "setup_cmd": "",
    },
    "rust": {
        "dockerfile": "rust",
        "test_file": "tests.rs",
        "test_cmd": "cargo test --manifest-path /tests/Cargo.toml",
        "setup_cmd": "",
    },
    "ruby": {
        "dockerfile": "ruby",
        "test_file": "test_solution.rb",
        "test_cmd": "ruby /tests/test_solution.rb",
        "setup_cmd": "gem install minitest 2>/dev/null || true",
    },
}


def load_exercism(language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load Exercism dataset from HuggingFace or GitHub.

    Exercism tracks are available at:
    - HuggingFace: Various community uploads
    - GitHub: exercism/{language} repositories
    """
    print(f"Loading Exercism dataset ({language})...")

    # Try HuggingFace sources (updated Jan 2026)
    repo_names = [
        "RajMaheshwari/Exercism-Python",  # Has instruction, signature, test
        "ninja-x/exercism-retry-full",
        f"exercism/exercises-{language}",
        f"exercism/{language}",
        "exercism/exercism",
    ]

    ds = None
    for repo_name in repo_names:
        try:
            ds = load_dataset(repo_name, split="train", streaming=True)
            print(f"Loaded from {repo_name}")
            break
        except Exception as e:
            print(f"Could not load from {repo_name}: {e}")
            continue

    if ds is None:
        # Try with language as config
        try:
            ds = load_dataset("exercism/exercism", language, split="train", streaming=True)
            print(f"Loaded from exercism/exercism with config {language}")
        except Exception as e:
            raise ValueError(f"Could not load Exercism dataset for {language}: {e}")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit), total=limit, desc="Loading samples"):
        # Extract exercise components
        instructions = (
            sample.get("instruction") or  # RajMaheshwari dataset
            sample.get("instructions") or
            sample.get("description") or
            sample.get("readme", "")
        )
        test_code = (
            sample.get("test_code") or
            sample.get("tests") or
            sample.get("test", "")
        )
        solution = (
            sample.get("solution") or
            sample.get("example") or
            sample.get("canonical_solution", "")
        )

        if not instructions or not test_code:
            continue

        samples.append({
            "instructions": instructions,
            "test_code": test_code,
            "solution": solution,
            "exercise_name": sample.get("slug", sample.get("name", sample.get("exercise", ""))),
            "difficulty": sample.get("difficulty", ""),
            "topics": sample.get("topics", sample.get("tags", [])),
            "language": language,
        })

    print(f"Loaded {len(samples)} exercises")
    return samples


def process_test_code(test_code: str, language: str) -> str:
    """Process test code to ensure it can import from /app."""
    language = language.lower()

    if language == "python":
        # Add sys.path for importing from /app
        if "sys.path" not in test_code:
            test_code = "import sys\nsys.path.insert(0, '/app')\n\n" + test_code
        return test_code

    elif language == "javascript":
        # Add require path for /app
        if "require('/app" not in test_code and "require(\"/app" not in test_code:
            test_code = "const solution = require('/app/solution');\n\n" + test_code
        return test_code

    elif language == "go":
        # Add package declaration if missing
        if "package" not in test_code:
            test_code = "package main\n\nimport \"testing\"\n\n" + test_code
        return test_code

    return test_code


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for an Exercism sample."""
    language = sample.get("language", "python")
    config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["python"])

    try:
        dockerfile = get_dockerfile(config["dockerfile"])
    except ValueError:
        dockerfile = get_dockerfile("python")

    # Process test code
    test_code = process_test_code(sample.get("test_code", ""), language)

    test_files = {
        config["test_file"]: test_code,
    }

    # Create instruction - Exercism instructions are already well-formatted
    instruction = sample.get("instructions", "")
    if not instruction.strip().endswith("/app/"):
        instruction += "\n\nPlace your solution in /app/"

    metadata = {
        "source": "exercism",
        "language": language,
        "exercise": sample.get("exercise_name", ""),
        "difficulty": sample.get("difficulty", ""),
        "topics": sample.get("topics", []),
    }

    # Include solution if available
    solution_files = None
    if sample.get("solution"):
        ext = {"python": "py", "javascript": "js", "go": "go", "java": "java",
               "rust": "rs", "ruby": "rb"}.get(language, "py")
        solution_files = {f"solution.{ext}": sample["solution"]}

    test_sh = create_generic_test_sh(
        test_command=config["test_cmd"],
        setup_commands=config["setup_cmd"]
    )

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
    dataset_prefix: str = "exercism",
) -> str:
    """Generate harbor-format task directories."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating harbor tasks in: {temp_dir}")

    for i, sample in enumerate(tqdm(samples, desc="Creating tasks")):
        create_harbor_task(temp_dir, i, sample, dataset_prefix)

    print(f"Generated {len(samples)} harbor tasks successfully!")
    return str(temp_dir)


def main(language: str = DEFAULT_LANGUAGE, limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating Exercism tasks."""

    print(f"Step 1: Loading Exercism dataset ({language})...")
    samples = load_exercism(language=language, limit=limit)
    print(f"  -> {len(samples)} exercises loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    # Exercism instructions are already well-formatted, no LLM processing needed
    print("\nStep 2: Generating harbor task directories...")
    dataset_prefix = f"exercism-{language}"
    task_dir = generate_tasks(samples, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 3: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} Exercism tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from Exercism dataset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        choices=list(LANGUAGE_CONFIG.keys()),
                        help="Programming language")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit, upload=not args.no_upload)
