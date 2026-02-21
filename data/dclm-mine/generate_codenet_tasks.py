#!/usr/bin/env python3
"""
Generate tasks from competitive programming datasets (CodeNet / code_contests).

Uses deepmind/code_contests which contains competitive programming problems with:
- Problem descriptions
- Multiple solutions in various languages
- Input/output test cases (public, private, and generated)
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
# Maximum number of test cases to include per problem (keeps task dirs small)
MAX_TESTS_PER_PROBLEM = 10


def _extract_tests_from_dict(test_dict: Optional[Dict], max_count: int = 0) -> Tuple[List[str], List[str]]:
    """
    Extract (inputs, outputs) lists from a test dict like
    {"input": [...], "output": [...]}.

    Handles None, non-dict, string-vs-list values.
    Returns two parallel lists of strings.
    """
    if not test_dict or not isinstance(test_dict, dict):
        return [], []

    inputs = test_dict.get("input", [])
    outputs = test_dict.get("output", [])

    # Handle single-string values
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(outputs, str):
        outputs = [outputs]

    # Keep only paired entries
    paired_count = min(len(inputs), len(outputs))
    inputs = [str(v) for v in inputs[:paired_count]]
    outputs = [str(v) for v in outputs[:paired_count]]

    if max_count > 0:
        inputs = inputs[:max_count]
        outputs = outputs[:max_count]

    return inputs, outputs


def load_codenet(language: str = "python", limit: int = LIMIT) -> List[Dict]:
    """
    Load competitive programming problems from HuggingFace.

    Primary source: deepmind/code_contests
    Schema: name, description, public_tests, private_tests,
            generated_tests, source, difficulty, solutions
    """
    print(f"Loading CodeNet/code_contests dataset ({language})...")

    repo_names = [
        "deepmind/code_contests",
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
    skipped_no_problem = 0
    skipped_no_tests = 0
    print(f"Collecting up to {limit} samples (with valid test cases)...")

    for sample in tqdm(itertools.islice(ds, limit * 3), desc="Loading samples"):
        # Extract problem description -- supports multiple schemas
        problem = (
            sample.get("description") or
            sample.get("problem_description") or
            sample.get("problem", "")
        )

        if not problem:
            skipped_no_problem += 1
            continue

        # --- Extract solutions for the requested language ---
        # deepmind/code_contests language codes: 1=C++, 2=C++, 3=PYTHON3, 4=JAVA
        solutions = sample.get("solutions", {})
        solution = ""
        sample_lang = language.lower()

        if isinstance(solutions, dict):
            lang_map = {"python": 3, "cpp": 2, "java": 4}  # PYTHON3=3, CPP=2, JAVA=4
            lang_code = lang_map.get(language.lower(), 3)
            sol_languages = solutions.get("language", [])
            sol_solutions = solutions.get("solution", [])
            for i, lang in enumerate(sol_languages):
                if lang == lang_code and i < len(sol_solutions):
                    solution = sol_solutions[i]
                    break

        # --- Collect test cases from ALL available sources ---
        # Priority: public_tests > private_tests > generated_tests
        all_inputs: List[str] = []
        all_outputs: List[str] = []

        for test_key in ("public_tests", "private_tests", "generated_tests"):
            test_dict = sample.get(test_key)
            inp, out = _extract_tests_from_dict(test_dict)
            all_inputs.extend(inp)
            all_outputs.extend(out)

        # Also check top-level input/output fields (other CodeNet schemas)
        if not all_inputs:
            top_input = sample.get("input")
            top_output = sample.get("output")
            if top_input is not None and top_output is not None:
                if isinstance(top_input, str):
                    top_input = [top_input]
                if isinstance(top_output, str):
                    top_output = [top_output]
                if isinstance(top_input, list) and isinstance(top_output, list):
                    paired = min(len(top_input), len(top_output))
                    all_inputs = [str(v) for v in top_input[:paired]]
                    all_outputs = [str(v) for v in top_output[:paired]]

        # Cap the number of test cases
        if len(all_inputs) > MAX_TESTS_PER_PROBLEM:
            all_inputs = all_inputs[:MAX_TESTS_PER_PROBLEM]
            all_outputs = all_outputs[:MAX_TESTS_PER_PROBLEM]

        # Skip samples that have no test cases at all
        if not all_inputs or not all_outputs:
            skipped_no_tests += 1
            continue

        samples.append({
            "problem_description": problem,
            "solution": solution,
            "language": sample_lang or language,
            "problem_id": sample.get("name", sample.get("id", "")),
            "inputs": all_inputs,
            "outputs": all_outputs,
            "difficulty": str(sample.get("difficulty", "")),
        })

        if len(samples) >= limit:
            break

    print(f"Loaded {len(samples)} problems "
          f"(skipped {skipped_no_problem} without description, "
          f"{skipped_no_tests} without test cases)")
    return samples


def create_test_files(sample: Dict) -> Dict[str, str]:
    """Create input/output test files from sample.

    Callers must ensure sample has non-empty inputs/outputs lists
    (load_codenet already filters for this).
    """
    test_files = {}

    inputs = sample.get("inputs", [])
    outputs = sample.get("outputs", [])

    if not inputs or not outputs:
        raise ValueError(
            f"Sample {sample.get('problem_id', '?')} has no test cases. "
            "This should have been filtered out during loading."
        )

    # Create input/output files for each test case
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        test_files[f"inputs/input_{i}.txt"] = str(inp)
        test_files[f"outputs/output_{i}.txt"] = str(out)

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
