#!/usr/bin/env python3
"""
Generate tasks from IBM CodeNet dataset (14M solutions).

CodeNet contains competitive programming problems with:
- Problem descriptions
- Multiple solutions in various languages
- Input/output test cases
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.completions import run_completions
from data.commons import (
    PROBLEM_STATEMENT_CLEANUP_PROMPT,
    create_harbor_task_directory_generic,
    create_io_test_sh,
    get_dockerfile,
    upload_tasks_to_hf,
)

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 10000
MODEL = "gpt-4o-mini"
DEFAULT_LANGUAGE = "python"


def load_codenet(language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load CodeNet dataset from HuggingFace.

    The dataset is available at various HuggingFace repos.
    """
    print(f"Loading CodeNet dataset ({language})...")

    # Try multiple possible sources (updated Jan 2026)
    # IBM/CodeNet was removed - using alternatives
    repo_names = [
        "sumuks/CodeNet-16K",        # 16k curated CodeNet samples
        "petersa2/CodeNet",          # Alternative CodeNet upload
        "deepmind/code_contests",    # Best alternative with competitive programming (4k samples)
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
        raise ValueError("Could not load CodeNet/competitive programming dataset from any source")

    samples = []
    print(f"Collecting {limit} samples...")

    for sample in tqdm(itertools.islice(ds, limit * 2), desc="Loading samples"):
        # Handle deepmind/code_contests schema:
        # name, description, public_tests, private_tests, generated_tests, source, difficulty, solutions
        problem = (
            sample.get("description") or
            sample.get("problem_description") or
            sample.get("problem", "")
        )

        # Get solutions - code_contests has a list of solution objects
        solutions = sample.get("solutions", {})
        solution = ""
        sample_lang = language.lower()

        # Extract solution for the requested language
        if isinstance(solutions, dict):
            lang_map = {"python": 3, "cpp": 2, "java": 4}  # PYTHON3=3, CPP=2, JAVA=4
            lang_code = lang_map.get(language.lower(), 3)
            sol_languages = solutions.get("language", [])
            sol_solutions = solutions.get("solution", [])
            for i, lang in enumerate(sol_languages):
                if lang == lang_code and i < len(sol_solutions):
                    solution = sol_solutions[i]
                    break

        if not problem:
            continue

        # Get test cases from public_tests
        public_tests = sample.get("public_tests", {})
        inputs = public_tests.get("input", []) if isinstance(public_tests, dict) else []
        outputs = public_tests.get("output", []) if isinstance(public_tests, dict) else []

        # Handle both list and string formats
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        samples.append({
            "problem_description": problem,
            "solution": solution,
            "language": sample_lang or language,
            "problem_id": sample.get("name", sample.get("id", "")),
            "inputs": inputs,
            "outputs": outputs,
            "difficulty": str(sample.get("difficulty", "")),
        })

        if len(samples) >= limit:
            break

    print(f"Loaded {len(samples)} problems")
    return samples


def create_test_files(sample: Dict) -> Dict[str, str]:
    """Create input/output test files from sample."""
    test_files = {}

    inputs = sample.get("inputs", [])
    outputs = sample.get("outputs", [])

    # Create input/output files
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        test_files[f"inputs/input_{i}.txt"] = str(inp)
        test_files[f"outputs/output_{i}.txt"] = str(out)

    # If no test cases, create placeholder
    if not inputs:
        test_files["inputs/input_0.txt"] = ""
        test_files["outputs/output_0.txt"] = ""

    return test_files


def create_harbor_task(
    output_dir: Path,
    task_id: int,
    instruction: str,
    sample: Dict,
    dataset_prefix: str,
) -> Path:
    """Create a harbor task directory for a CodeNet sample."""
    language = sample.get("language", "python")

    # Map to dockerfile language
    dockerfile_lang = language
    if "python" in language.lower():
        dockerfile_lang = "python"
    elif "java" in language.lower():
        dockerfile_lang = "java"
    elif "c++" in language.lower() or "cpp" in language.lower():
        dockerfile_lang = "cpp"
    elif "go" in language.lower():
        dockerfile_lang = "go"
    else:
        dockerfile_lang = "python"

    try:
        dockerfile = get_dockerfile(dockerfile_lang)
    except ValueError:
        dockerfile = get_dockerfile("python")

    test_files = create_test_files(sample)

    # Determine solution command based on language
    if "python" in dockerfile_lang:
        solution_cmd = "python3 /app/solution.py"
    elif "java" in dockerfile_lang:
        solution_cmd = "java -cp /app Solution"
    elif "cpp" in dockerfile_lang:
        solution_cmd = "/app/solution"
    else:
        solution_cmd = "python3 /app/solution.py"

    metadata = {
        "source": "codenet",
        "problem_id": sample.get("problem_id", ""),
        "language": language,
        "difficulty": sample.get("difficulty", ""),
        "num_tests": len(sample.get("inputs", [])),
    }

    # Include solution if available
    solution_files = None
    if sample.get("solution"):
        ext = {"python": "py", "java": "java", "cpp": "cpp", "c++": "cpp", "go": "go"}.get(dockerfile_lang, "py")
        solution_files = {f"solution.{ext}": sample["solution"]}

    return create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=dockerfile,
        test_sh=create_io_test_sh(solution_cmd=solution_cmd),
        test_files=test_files,
        dataset_prefix=dataset_prefix,
        metadata=metadata,
        solution_files=solution_files,
    )


def generate_tasks(
    samples: List[Dict],
    instructions: List[str],
    dataset_prefix: str = "codenet",
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


def main(language: str = DEFAULT_LANGUAGE, limit: int = LIMIT, upload: bool = True) -> None:
    """Main pipeline for generating CodeNet tasks."""

    print(f"Step 1: Loading CodeNet dataset ({language})...")
    samples = load_codenet(language=language, limit=limit)
    print(f"  -> {len(samples)} problems loaded")

    if not samples:
        print("\nNo samples found. Exiting.")
        return

    print("\nStep 2: Cleaning up problem descriptions...")
    # Use problem descriptions as instructions (minimal LLM processing)
    dataset = Dataset.from_list([{"description": s["problem_description"]} for s in samples])

    result = run_completions(
        dataset,
        model=MODEL,
        map_type="chat",
        map_config={
            "user_message": PROBLEM_STATEMENT_CLEANUP_PROMPT,
            "output_column": "task_description"
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
    )
    instructions = result.dataset["task_description"]
    print(f"  -> Processed {len(instructions)} task descriptions")

    print("\nStep 3: Generating harbor task directories...")
    dataset_prefix = f"codenet-{language}"
    task_dir = generate_tasks(samples, instructions, dataset_prefix)
    print(f"  -> Task directory: {task_dir}")

    if upload:
        print("\nStep 4: Uploading to HuggingFace...")
        repo_url = upload_tasks_to_hf(task_dir, f"DCAgent/exp_rpt_{dataset_prefix}")
        print(f"  -> Repository: {repo_url}")
    else:
        repo_url = "Not uploaded"

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(samples)} CodeNet tasks!")
    print(f"Output directory: {task_dir}")
    print(f"Repository: {repo_url}")
    print(f"{'='*60}")

    return task_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tasks from CodeNet dataset")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Programming language filter")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Maximum samples to process")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")

    args = parser.parse_args()
    main(language=args.language, limit=args.limit, upload=not args.no_upload)
