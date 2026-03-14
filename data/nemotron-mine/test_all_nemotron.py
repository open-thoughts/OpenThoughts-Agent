#!/usr/bin/env python3
"""
Generate 25 tasks from each Nemotron generator and test with Harbor.

Usage:
    python test_all_nemotron.py                  # Run all generators
    python test_all_nemotron.py bash junit rust   # Run specific generators
    python test_all_nemotron.py --generate-only   # Generate tasks but skip harbor
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "harbor"))

from datasets import Dataset
from tqdm import tqdm

from data.completions import run_completions

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 25
HARBOR_MODEL = "openai/gpt-5-nano-2025-08-07"
N_CONCURRENT = 25
CONFIG_PATH = Path(__file__).parent.parent.parent / "eval" / "tacc" / "dcagent_eval_config.yaml"

# =============================================================================
# Generator Registry
# =============================================================================
# Each entry: (module_name, filter_func, gen_func, prompt_attr, model_attr, prefix, hf_repo)
GENERATORS = {
    "bash": {
        "module": "generate_bash_tasks",
        "filter_func": "filter_verifiers_from_nemotron",
        "gen_func": "generate_bash_tasks",
        "prompt_attr": "VERIFIER_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-bash",
        "instruction_col": "task_description",
    },
    "bash-withtests": {
        "module": "generate_bash_tasks_with_tests",
        "filter_func": "filter_verifiers_from_nemotron",
        "gen_func": "generate_bash_tasks_with_tests",
        "prompt_attr": "VERIFIER_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-bash-withtests",
        "instruction_col": "task_description",
    },
    "bash-withtests-gpt5mini": {
        "module": "generate_bash_tasks_with_tests_gpt5mini",
        "filter_func": "filter_verifiers_from_nemotron",
        "gen_func": "generate_bash_tasks_with_tests",
        "prompt_attr": "VERIFIER_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-bash-withtests-gpt5mini",
        "instruction_col": "task_description",
    },
    "cpp": {
        "module": "generate_cpp_test_tasks",
        "filter_func": "filter_cpp_tests_from_nemotron",
        "gen_func": "generate_cpp_tasks",
        "prompt_attr": "CPP_TEST_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-cpp",
        "instruction_col": "task_description",
    },
    "csharp": {
        "module": "generate_csharp_test_tasks",
        "filter_func": "filter_csharp_tests_from_nemotron",
        "gen_func": "generate_csharp_tasks",
        "prompt_attr": "CSHARP_TEST_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-csharp",
        "instruction_col": "task_description",
    },
    "junit": {
        "module": "generate_junit_tasks",
        "filter_func": "filter_junit_from_nemotron",
        "gen_func": "generate_junit_tasks",
        "prompt_attr": "JUNIT_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-junit",
        "instruction_col": "task_description",
    },
    "pytest": {
        "module": "generate_pytest_tasks_gpt5mini",
        "filter_func": "filter_pytest_files_from_nemotron",
        "gen_func": "generate_tasks",
        "prompt_attr": "PYTEST_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-pytest-synthetic",
        "instruction_col": "task_description",
    },
    "ruby": {
        "module": "generate_rspec_tasks",
        "filter_func": "filter_ruby_tests_from_nemotron",
        "gen_func": "generate_ruby_tasks",
        "prompt_attr": "RSPEC_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-ruby",
        "instruction_col": "task_description",
    },
    "rust": {
        "module": "generate_rust_test_tasks",
        "filter_func": "filter_rust_tests_from_nemotron",
        "gen_func": "generate_rust_tasks",
        "prompt_attr": "RUST_TEST_TO_TASK_PROMPT",
        "model_attr": "MODEL",
        "prefix": "nemotron-rust",
        "instruction_col": "task_description",
    },
}

# Aliases for convenience
GENERATOR_ALIASES = {
    "rspec": "ruby",
    "junit": "junit",
    "pytest": "pytest",
}


def run_generator(name: str, config: dict) -> dict:
    """Run a single generator: filter -> LLM -> create tasks. Returns task_dir path."""
    print(f"\n{'='*60}")
    print(f"  Generator: {name} ({config['module']})")
    print(f"{'='*60}")

    # Import the module
    mod = importlib.import_module(config["module"])

    # Step 1: Filter samples from Nemotron
    filter_fn = getattr(mod, config["filter_func"])
    print(f"\n  Step 1: Filtering from Nemotron (limit={LIMIT})...")
    samples = filter_fn(LIMIT)
    print(f"  -> Found {len(samples)} samples")

    if not samples:
        return {"generator": name, "error": "No samples found", "n_samples": 0}

    # Step 2: Generate instructions via LLM
    model = getattr(mod, config["model_attr"])
    prompt = getattr(mod, config["prompt_attr"])
    print(f"\n  Step 2: Generating instructions with {model}...")

    dataset = Dataset.from_list(samples)
    result = run_completions(
        dataset,
        model=model,
        map_type="chat",
        map_config={
            "user_message": prompt,
            "output_column": config["instruction_col"],
        },
        max_requests_per_minute=500,
        max_tokens_per_minute=1_000_000,
        require_all_responses=False,
    )
    instructions = result.dataset[config["instruction_col"]]
    print(f"  -> Generated {len(instructions)} instructions")

    # Step 3: Create harbor task directories
    gen_fn = getattr(mod, config["gen_func"])
    print(f"\n  Step 3: Creating harbor tasks (prefix={config['prefix']})...")
    task_dir = gen_fn(samples, instructions, config["prefix"])
    n_tasks = len(list(Path(task_dir).glob("*/instruction.md")))
    print(f"  -> Created {n_tasks} tasks at {task_dir}")

    return {
        "generator": name,
        "task_dir": task_dir,
        "n_samples": len(samples),
        "n_instructions": len(instructions),
        "n_tasks": n_tasks,
    }


def run_harbor(task_dir: str, name: str) -> dict:
    """Run harbor evaluation on a task directory."""
    os.chdir(Path(__file__).parent.parent.parent)
    job_name = f"nemotron_test_{name}_{os.getpid()}"

    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", str(N_CONCURRENT),
        "--agent", "terminus-2",
        "--model", HARBOR_MODEL,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
    ]

    if CONFIG_PATH.exists():
        cmd.extend(["--config", str(CONFIG_PATH)])

    print(f"\n  Harbor cmd: {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=False)

    # Parse results
    result_file = Path("jobs") / job_name / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)

        # Harbor result.json uses stats.evals.<key>.reward_stats and metrics
        stats = data.get("stats", {})
        total = stats.get("n_trials", 0)
        n_errors = stats.get("n_errors", 0)

        # Get the first (and typically only) eval entry
        evals = stats.get("evals", {})
        mean_reward = 0.0
        passed = 0
        for eval_key, eval_data in evals.items():
            metrics = eval_data.get("metrics", [{}])
            mean_reward = metrics[0].get("mean", 0.0) if metrics else 0.0
            reward_stats = eval_data.get("reward_stats", {}).get("reward", {})
            # Count tasks with reward > 0
            for reward_val, task_list in reward_stats.items():
                if float(reward_val) > 0:
                    passed += len(task_list)

        return {
            "generator": name,
            "total": total,
            "passed": passed,
            "errors": n_errors,
            "mean_reward": mean_reward,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
        }

    return {"generator": name, "error": "No result.json found"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Nemotron generators with Harbor")
    parser.add_argument("generators", nargs="*", default=list(GENERATORS.keys()),
                        help=f"Generators to test. Available: {', '.join(GENERATORS.keys())}")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate tasks but don't run harbor")
    parser.add_argument("--limit", type=int, default=25,
                        help="Number of tasks to generate per generator (default: 25)")
    args = parser.parse_args()

    global LIMIT
    LIMIT = args.limit

    all_results = []

    for name in args.generators:
        # Resolve aliases (e.g. "rspec" -> "ruby")
        name = GENERATOR_ALIASES.get(name, name)
        if name not in GENERATORS:
            print(f"Unknown generator: {name}")
            print(f"Available: {', '.join(GENERATORS.keys())}")
            continue

        # Step 1: Generate tasks
        try:
            gen_result = run_generator(name, GENERATORS[name])
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"generator": name, "error": str(e)})
            continue

        if "error" in gen_result:
            all_results.append(gen_result)
            continue

        if args.generate_only:
            print(f"  --generate-only: skipping harbor. Tasks at {gen_result['task_dir']}")
            all_results.append(gen_result)
            continue

        # Step 2: Run harbor
        print(f"\n  Step 4: Running Harbor evaluation...")
        try:
            harbor_result = run_harbor(gen_result["task_dir"], name)
            harbor_result["task_dir"] = gen_result["task_dir"]
            harbor_result["n_tasks"] = gen_result["n_tasks"]
            all_results.append(harbor_result)
            print(f"\n  Result: {json.dumps(harbor_result, indent=2)}")
        except Exception as e:
            print(f"  ERROR running harbor for {name}: {e}")
            gen_result["harbor_error"] = str(e)
            all_results.append(gen_result)

    # Summary
    print(f"\n\n{'='*70}")
    print("NEMOTRON GENERATOR TEST RESULTS")
    print(f"{'='*70}")

    if any("pass_rate" in r for r in all_results):
        print(f"{'Generator':<30} {'Tasks':>6} {'Passed':>8} {'Rate':>8}")
        print("-" * 55)
        for r in all_results:
            if "error" in r:
                print(f"{r['generator']:<30} {'ERROR':>6}  {r.get('error', 'unknown')}")
            elif "pass_rate" in r:
                print(f"{r['generator']:<30} {r['total']:>6} {r['passed']:>8} {r['pass_rate']:>7.1f}%")
            else:
                print(f"{r['generator']:<30} {r.get('n_tasks', '?'):>6} tasks generated")
    else:
        for r in all_results:
            if "error" in r:
                print(f"  {r['generator']}: ERROR - {r.get('error', 'unknown')}")
            else:
                print(f"  {r['generator']}: {r.get('n_tasks', '?')} tasks at {r.get('task_dir', '?')}")

    # Save results
    out_file = Path(__file__).parent / "nemotron_test_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
