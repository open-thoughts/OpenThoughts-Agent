#!/usr/bin/env python3
"""
General evaluation script for running LLM agents on RL tasks.

Usage:
    python evaluate.py --model openai/gpt-5-chat-latest --dataset DCAgent/nl2bash  # All tasks
    python evaluate.py --model openai/gpt-5-chat-latest --dataset DCAgent/nl2bash --n-tasks 100  # First 100
    python evaluate.py --help
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download

# Add the dc-agent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.sandboxes.tasks_parquet_converter import from_parquet


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM agents on RL tasks from HuggingFace datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GPT-5 on ALL nl2bash tasks
  python evaluate.py --model openai/gpt-5-chat-latest --dataset DCAgent/nl2bash

  # Run GPT-5 on first 100 nl2bash tasks
  python evaluate.py --model openai/gpt-5-chat-latest --dataset DCAgent/nl2bash --n-tasks 100

  # Run GPT-4o-mini on 50 tasks with higher concurrency
  python evaluate.py --model openai/gpt-4o-mini --dataset DCAgent/nl2bash --n-tasks 50 --n-concurrent 8

  # Run Claude on cleaned oracle dataset with retries
  python evaluate.py --model anthropic/claude-sonnet-4-5-20250929 --dataset DCAgent2/nl2bash-tasks-cleaned-oracle --n-attempts 3

  # Use Docker instead of Daytona
  python evaluate.py --model openai/gpt-4o --dataset DCAgent/nl2bash --n-tasks 10 --env docker
        """
    )

    # Required arguments
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name (e.g., openai/gpt-5-chat-latest, openai/gpt-4o-mini, anthropic/claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="HuggingFace dataset repo ID (e.g., DCAgent/nl2bash, DCAgent2/nl2bash-tasks-cleaned-oracle)"
    )

    # Optional arguments
    parser.add_argument(
        "--agent", "-a",
        default="terminus-2",
        help="Agent name (default: terminus-2)"
    )
    parser.add_argument(
        "--n-tasks", "-n",
        type=int,
        default=None,
        help="Number of tasks to evaluate (first N tasks, default: all tasks)"
    )
    parser.add_argument(
        "--n-concurrent", "-c",
        type=int,
        default=4,
        help="Number of concurrent tasks (default: 4)"
    )
    parser.add_argument(
        "--n-attempts", "-k",
        type=int,
        default=1,
        help="Number of retry attempts per task (default: 1)"
    )
    parser.add_argument(
        "--env", "-e",
        default="daytona",
        choices=["daytona", "docker"],
        help="Environment type (default: daytona)"
    )
    parser.add_argument(
        "--job-name", "-j",
        default=None,
        help="Custom job name (default: auto-generated)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output results file path (default: auto-generated)"
    )

    return parser.parse_args()


def download_dataset(repo_id: str) -> Path:
    """
    Download dataset from HuggingFace and extract tasks.

    Args:
        repo_id: HuggingFace dataset repo ID

    Returns:
        Path to the extracted tasks directory
    """
    print(f"\n{'='*80}")
    print(f"Downloading dataset: {repo_id}")
    print("="*80)

    # Download the dataset
    dataset_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset downloaded to: {dataset_path}\n")

    # Check for parquet files
    parquet_file = Path(dataset_path) / "tasks.parquet"
    if not parquet_file.exists():
        parquet_file = Path(dataset_path) / "nl2bash_tasks.parquet"

    if not parquet_file.exists():
        # Try to find any .parquet file
        parquet_files = list(Path(dataset_path).glob("*.parquet"))
        if parquet_files:
            parquet_file = parquet_files[0]
            print(f"Found parquet file: {parquet_file.name}")

    if parquet_file.exists():
        print(f"Extracting tasks from parquet file: {parquet_file.name}...")

        # Create extraction directory based on dataset name
        dataset_name = repo_id.replace("/", "_")
        extract_dir = Path(f"./extracted_{dataset_name}")
        extract_dir.mkdir(exist_ok=True)

        # Extract tasks from parquet
        extracted_paths = from_parquet(
            parquet_path=str(parquet_file),
            base=str(extract_dir),
            on_exist="overwrite"
        )

        print(f"Extracted {len(extracted_paths)} tasks to: {extract_dir}")
        return extract_dir
    else:
        # Already in directory format
        print("Dataset is already in directory format")
        return Path(dataset_path)


def select_first_n_tasks(dataset_dir: Path, n: int, job_name: str) -> tuple[Path, int]:
    """
    Create a new directory with only the first N tasks (or all if n is None).

    Args:
        dataset_dir: Directory containing all tasks
        n: Number of tasks to select (None for all)
        job_name: Job name for directory naming

    Returns:
        Tuple of (Path to directory containing tasks, actual number of tasks)
    """
    # Get all task directories, sorted by name
    all_tasks = sorted([
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    ])

    total_available = len(all_tasks)

    if total_available == 0:
        raise ValueError("No valid tasks found in dataset!")

    if n is None:
        print(f"\nUsing all {total_available} tasks...")
        n = total_available
    else:
        print(f"\nSelecting first {n} tasks...")
        if total_available < n:
            print(f"Warning: Only {total_available} tasks available, using all of them")
            n = total_available

    selected_tasks = all_tasks[:n]

    # Create a new directory for selected tasks
    selected_dir = Path(f"./selected_tasks_{job_name}")
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True)

    # Copy selected tasks
    print(f"Copying {n} tasks...")
    for task in selected_tasks:
        dest = selected_dir / task.name
        shutil.copytree(task, dest)

    print(f"Created dataset with {n} tasks: {selected_dir}")
    return selected_dir, n


def run_harbor_evaluation(
    dataset_path: Path,
    agent_name: str,
    model_name: str,
    n_concurrent: int,
    n_attempts: int,
    env_type: str,
    job_name: str
) -> Path:
    """
    Run Harbor evaluation on the dataset.

    Args:
        dataset_path: Path to directory containing tasks
        agent_name: Name of agent to use
        model_name: LLM model to use
        n_concurrent: Number of tasks to run in parallel
        n_attempts: Number of retry attempts per task
        env_type: Environment type (daytona or docker)
        job_name: Name for this evaluation job

    Returns:
        Path to the job directory
    """
    print(f"\n{'='*80}")
    print("Running Harbor Evaluation")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Agent: {agent_name}")
    print(f"Model: {model_name}")
    print(f"Environment: {env_type}")
    print(f"Concurrent tasks: {n_concurrent}")
    print(f"Attempts per task: {n_attempts}")
    print(f"Job name: {job_name}")
    print("="*80)

    # Find harbor executable
    harbor_cmd = shutil.which("harbor")
    if not harbor_cmd:
        possible_paths = [
            Path.home() / ".local" / "bin" / "harbor",
            "/usr/local/bin/harbor",
            "/opt/homebrew/bin/harbor",
        ]
        for path in possible_paths:
            if path.exists():
                harbor_cmd = str(path)
                break

        if not harbor_cmd:
            raise RuntimeError(
                "Could not find 'harbor' command. Please install it:\n"
                "  uv tool install harbor"
            )

    # Clean up old job directory if it exists
    job_dir = Path("jobs") / job_name
    if job_dir.exists():
        print(f"\nRemoving old job directory: {job_dir}")
        shutil.rmtree(job_dir)

    # Build harbor CLI command
    cmd = [
        harbor_cmd,
        "jobs",
        "start",
        "-p", str(dataset_path),
        "--agent", agent_name,
        "--model", model_name,
        "--env", env_type,
        "--n-attempts", str(n_attempts),
        "--n-concurrent", str(n_concurrent),
        "--job-name", job_name
    ]

    print(f"\nStarting evaluation...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run harbor command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Harbor command failed!")
        print(f"Error: {e}")
        raise

    print(f"\nEvaluation complete! Results saved to: {job_dir}")
    return job_dir


def generate_results_summary(job_dir: Path, output_file: str, model_name: str, n_tasks: int) -> dict:
    """
    Generate a summary file with PASS/FAIL for each task.

    Args:
        job_dir: Path to the job directory
        output_file: Path to output file
        model_name: Model name for the report
        n_tasks: Number of tasks evaluated

    Returns:
        Dictionary with summary statistics
    """
    print(f"\n{'='*80}")
    print("Generating Results Summary")
    print("="*80)

    # Read the main result.json file
    result_file = job_dir / "result.json"

    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, 'r') as f:
        results = json.load(f)

    # Get all task result directories
    task_dirs = sorted([
        d for d in job_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    # Collect results for each task
    task_results = []
    total_tasks = 0
    passed_tasks = 0
    failed_tasks = 0
    error_tasks = 0

    for task_dir in task_dirs:
        task_result_file = task_dir / "result.json"

        if not task_result_file.exists():
            continue

        with open(task_result_file, 'r') as f:
            task_result = json.load(f)

        task_name = task_result.get("task_name", task_dir.name)
        verifier_result = task_result.get("verifier_result") or {}
        rewards = verifier_result.get("rewards", {})
        reward = rewards.get("reward", 0.0)
        exception_info = task_result.get("exception_info")

        total_tasks += 1

        if exception_info:
            status = "ERROR"
            error_tasks += 1
        elif reward > 0:
            status = "PASS"
            passed_tasks += 1
        else:
            status = "FAIL"
            failed_tasks += 1

        # Get cost information
        agent_result = task_result.get("agent_result") or {}
        cost_usd = agent_result.get("cost_usd", 0)

        task_results.append({
            "task_name": task_name,
            "status": status,
            "reward": reward,
            "cost": cost_usd
        })

    # Sort by task name
    task_results.sort(key=lambda x: x["task_name"])

    # Calculate statistics
    total_cost = sum(t["cost"] for t in task_results)
    pass_rate = (passed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    completed_tasks = total_tasks - error_tasks
    pass_rate_completed = (passed_tasks / completed_tasks * 100) if completed_tasks > 0 else 0

    # Write results to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Evaluation Results\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Requested tasks: {n_tasks}\n")
        f.write(f"Actual tasks: {total_tasks}\n")
        f.write(f"Completed: {completed_tasks} ({completed_tasks/total_tasks*100:.1f}%)\n")
        f.write(f"Errors: {error_tasks} ({error_tasks/total_tasks*100:.1f}%)\n")
        f.write(f"\n")
        f.write(f"Passed: {passed_tasks} ({pass_rate:.1f}% of all, {pass_rate_completed:.1f}% of completed)\n")
        f.write(f"Failed: {failed_tasks} ({failed_tasks/total_tasks*100:.1f}%)\n")
        f.write(f"\n")
        f.write(f"Total cost: ${total_cost:.4f}\n")
        f.write(f"Average cost per task: ${total_cost/total_tasks:.6f}\n" if total_tasks > 0 else "")
        f.write("\n" + "="*80 + "\n")
        f.write("Individual Task Results\n")
        f.write("="*80 + "\n\n")

        # Write each task result
        for i, task in enumerate(task_results, 1):
            status_icon = "PASS" if task["status"] == "PASS" else "FAIL" if task["status"] == "FAIL" else "ERROR"
            f.write(f"{i:3d}. {status_icon:5s} | {task['task_name']:40s} | Reward: {task['reward']:.2f} | Cost: ${task['cost']:.6f}\n")

        f.write("\n" + "="*80 + "\n")

    # Create summary dict
    summary = {
        "model": model_name,
        "total_tasks": total_tasks,
        "passed": passed_tasks,
        "failed": failed_tasks,
        "errors": error_tasks,
        "pass_rate": pass_rate,
        "pass_rate_completed": pass_rate_completed,
        "total_cost": total_cost,
        "results_file": str(output_path),
        "job_dir": str(job_dir)
    }

    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total: {total_tasks}")
    print(f"  Passed: {passed_tasks} ({pass_rate:.1f}%)")
    print(f"  Failed: {failed_tasks} ({failed_tasks/total_tasks*100:.1f}%)")
    print(f"  Errors: {error_tasks} ({error_tasks/total_tasks*100:.1f}%)")
    print(f"  Pass rate (completed only): {pass_rate_completed:.1f}%")
    print(f"  Total cost: ${total_cost:.4f}")

    return summary


def main():
    """Main function."""
    args = parse_args()

    # Check for required API keys
    if args.model.startswith("openai/") and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return 1

    if args.model.startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please run: export ANTHROPIC_API_KEY='your-key-here'")
        return 1

    if args.env == "daytona" and not os.getenv("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY environment variable not set!")
        print("Please run: export DAYTONA_API_KEY='your-key-here'")
        return 1

    # Generate job name if not provided
    if args.job_name:
        job_name = args.job_name
    else:
        model_short = args.model.split("/")[-1].replace("-", "_")
        dataset_short = args.dataset.split("/")[-1].replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_tasks_str = str(args.n_tasks) if args.n_tasks else "all"
        job_name = f"{model_short}_{dataset_short}_{n_tasks_str}_{timestamp}"

    # Generate output file name if not provided
    if args.output:
        output_file = args.output
    else:
        output_file = f"results_{job_name}.txt"

    print("\n" + "="*80)
    print("LLM Agent Evaluation")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Agent: {args.agent}")
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.n_tasks if args.n_tasks else 'all'}")
    print(f"Concurrent: {args.n_concurrent}")
    print(f"Attempts: {args.n_attempts}")
    print(f"Environment: {args.env}")
    print(f"Job name: {job_name}")
    print(f"Output: {output_file}")
    print("="*80)

    try:
        # Step 1: Download dataset
        dataset_dir = download_dataset(args.dataset)

        # Step 2: Select first N tasks (or all)
        selected_dir, actual_n_tasks = select_first_n_tasks(dataset_dir, args.n_tasks, job_name)

        # Step 3: Run Harbor evaluation
        job_dir = run_harbor_evaluation(
            dataset_path=selected_dir,
            agent_name=args.agent,
            model_name=args.model,
            n_concurrent=args.n_concurrent,
            n_attempts=args.n_attempts,
            env_type=args.env,
            job_name=job_name
        )

        # Step 4: Generate results summary
        summary = generate_results_summary(job_dir, output_file, args.model, actual_n_tasks)

        print(f"\n{'='*80}")
        print("Evaluation Complete!")
        print("="*80)
        print(f"\nPass Rate: {summary['pass_rate']:.1f}%")
        print(f"Pass Rate (completed): {summary['pass_rate_completed']:.1f}%")
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"\nResults file: {output_file}")
        print(f"Job directory: {job_dir}")
        print(f"\nTo view results:")
        print(f"  cat {output_file}")

        # Return pass rate as exit code (0-100)
        return 0

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
