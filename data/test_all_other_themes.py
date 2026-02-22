#!/usr/bin/env python3
"""Generate and test experiments from Themes 1, 2, 4, 5, 7 with harbor pass rates.

Usage:
    python data/test_all_other_themes.py --generate-only
    python data/test_all_other_themes.py --test-only
    python data/test_all_other_themes.py
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "harbor"))

LIMIT = 5
AGENT_MODEL = "openai/gpt-5-nano-2025-08-07"
CONFIG = str(PROJECT_ROOT / "eval/tacc/dcagent_eval_config.yaml")
HF_SUFFIX = "_test5"

# ─── Experiment definitions ─────────────────────────────────────

def _run_script(script_path: str, env_overrides: dict = None) -> str:
    """Run a generation script, return the task directory path."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    # Override the limit and HF suffix via env vars
    env["EXP_LIMIT"] = str(LIMIT)
    env["EXP_HF_SUFFIX"] = HF_SUFFIX

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:500]}")
        return None

    # Parse task dir from stdout (scripts print "Done: <url>" and task dir)
    for line in result.stdout.splitlines():
        if "-> OK:" in line and "tasks at" in line:
            return line.split("tasks at")[-1].strip()
    return None


def gen_from_script(script_dir: str, script_name: str) -> str:
    """Run a single generation script and return task dir."""
    script = Path(PROJECT_ROOT) / "data" / script_dir / script_name
    if not script.exists():
        print(f"  Script not found: {script}")
        return None
    print(f"  Running {script_name}...")
    return _run_script(str(script))


# Since generation scripts use hardcoded limits, we need to generate inline
# using the shared utilities directly.
from data.exp_general_utils import (
    transform_tasks,
    filter_tasks,
    llm_score_and_filter,
    llm_generate_and_create,
    make_test_sh,
)
from data.commons import load_harbor_tasks_as_dicts


def _quick_passthrough(prefix, hf_repo, source_repo="DCAgent/exp_rpt_stack-pytest"):
    """Load 5 tasks from source, create task dirs (no transform). Used for baselines."""
    tasks = load_harbor_tasks_as_dicts(source_repo, limit=LIMIT)
    print(f"  Loaded {len(tasks)} tasks")
    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    for i, row in enumerate(tasks[:LIMIT]):
        from data.commons import create_harbor_task_directory_generic
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=prefix,
            metadata={"source": row.get("task_dir_name", "")},
        )
    return str(out)


def _quick_transform(transform_fn, prefix, hf_repo):
    """Quick 5-task transform for testing."""
    tasks = load_harbor_tasks_as_dicts("DCAgent/exp_rpt_stack-pytest", limit=LIMIT)
    print(f"  Loaded {len(tasks)} tasks")
    from data.commons import create_harbor_task_directory_generic
    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    created = 0
    for i, row in enumerate(tasks[:LIMIT]):
        result = transform_fn(row)
        if result is None:
            continue
        create_harbor_task_directory_generic(
            output_dir=out, task_id=created,
            instruction=result["instruction"],
            dockerfile=result["dockerfile"],
            test_sh=result["test_sh"],
            test_files=json.loads(result["test_files_json"]),
            dataset_prefix=prefix,
            metadata={"source": row.get("task_dir_name", "")},
            solution_files=result.get("solution_files"),
        )
        created += 1
    print(f"  Created {created} tasks at {out}")
    return str(out)


# ─── Experiment lambdas ─────────────────────────────────────────

EXPERIMENTS = {}

# Theme 1: Just load and passthrough (the transforms are LLM-based scoring which is slow)
# Instead we test the source dataset directly as a baseline
EXPERIMENTS["baseline_source"] = lambda: _quick_passthrough("baseline", "DCAgent/baseline_test5")

# Theme 7 experiments that do programmatic transforms (no LLM needed):
# 7.6 temporal reward - just modifies test.sh
def _make_speed_bonus(row):
    """Add speed bonus to test.sh."""
    row = dict(row)
    original_test_sh = row["test_sh"]
    # Wrap with timing
    row["test_sh"] = original_test_sh.replace(
        'exit 0',
        'echo "SPEED_BONUS: task completed within timeout"\nexit 0',
    )
    return row

EXPERIMENTS["7.6_speed_bonus"] = lambda: _quick_transform(_make_speed_bonus, "exp-7-6-speed", "DCAgent/exp_7_6_speed_test5")

# For LLM-based experiments, we need to run their actual generation scripts
# but with limit=5. Let's import and run them directly.

def _gen_and_find_dir(script_rel_path):
    """Run a generation script with monkey-patched limit and no HF upload."""
    script = PROJECT_ROOT / "data" / script_rel_path
    if not script.exists():
        print(f"  Script not found: {script}")
        return None

    # Load secret env for API keys (needed for LLM-based generation)
    env = os.environ.copy()
    secret_env = Path("/scratch/10000/eguha3/old-dc-agent/secret.env")
    if secret_env.exists():
        for line in secret_env.read_text().splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                key, val = line[7:].split("=", 1)
                env[key] = os.path.expandvars(val)

    script_dir = str(script.parent)
    project_root = str(PROJECT_ROOT)
    script_str = str(script)

    # Wrapper that monkey-patches load_harbor_tasks_as_dicts to cap at LIMIT
    # and upload_tasks_to_hf to skip upload, then runs the script via runpy
    wrapper = f'''import sys, os
sys.path.insert(0, "{project_root}")
sys.path.insert(0, "{project_root}/data")
sys.path.insert(0, "{script_dir}")

# 1. Monkey-patch load_harbor_tasks_as_dicts to cap at {LIMIT}
import data.commons as _commons
_orig_load = _commons.load_harbor_tasks_as_dicts
def _capped_load(repo, limit=None, **kw):
    effective = min(limit or {LIMIT}, {LIMIT})
    return _orig_load(repo, limit=effective, **kw)
_commons.load_harbor_tasks_as_dicts = _capped_load

# 2. Monkey-patch upload_tasks_to_hf to skip and print task dir marker
def _skip_upload(task_dir, repo, *a, **kw):
    print(f"__TASK_DIR__:{{task_dir}}", flush=True)
    return f"https://huggingface.co/{{repo}}"
_commons.upload_tasks_to_hf = _skip_upload

# 3. Pre-import utility modules so they pick up patched bindings
for mod_name in ["exp_general_utils", "data.exp_general_utils",
                 "exp_8_utils", "data.exp_8_utils"]:
    try:
        _mod = __import__(mod_name, fromlist=["_"])
        if hasattr(_mod, "load_harbor_tasks_as_dicts"):
            _mod.load_harbor_tasks_as_dicts = _capped_load
        if hasattr(_mod, "upload_tasks_to_hf"):
            _mod.upload_tasks_to_hf = _skip_upload
    except ImportError:
        pass

# 4. Run the target script (runpy sets __file__ correctly)
import runpy
runpy.run_path("{script_str}", run_name="__main__")
'''

    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True, text=True, timeout=600,
        cwd=project_root, env=env,
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if stderr:
            print(f"  stderr: {stderr[-500:]}")

    # Parse __TASK_DIR__ marker from stdout
    for line in stdout.splitlines():
        if "__TASK_DIR__:" in line:
            p = line.split("__TASK_DIR__:")[-1].strip()
            if Path(p).is_dir():
                tasks = list(Path(p).glob("*/instruction.md"))
                if tasks:
                    return p

    # Fallback: search for temp dirs in output
    import re
    for line in (stdout + "\n" + stderr).splitlines():
        m = re.search(r'(/\S+(?:exp-|tmp)\S+)', line)
        if m:
            p = m.group(1).rstrip(".")
            if Path(p).is_dir():
                tasks = list(Path(p).glob("*/instruction.md"))
                if tasks:
                    return p

    if stdout:
        print(f"  stdout tail: {stdout[-300:]}")
    else:
        print("  (no stdout)")
    return None


# Register all generation scripts
GEN_SCRIPTS = {
    # Theme 1
    "1.1_goldilocks": "exp_1_1_difficulty_mixing/generate_goldilocks.py",
    "1.1_70_20_10": "exp_1_1_difficulty_mixing/generate_70_20_10.py",
    "1.2_minimal": "exp_1_2_instruction_specificity/generate_minimal.py",
    "1.2_rich": "exp_1_2_instruction_specificity/generate_rich.py",
    "1.3_top50": "exp_1_3_test_quality/generate_top50.py",
    "1.4_deduplicated": "exp_1_4_deduplication/generate_deduplicated.py",
    # Theme 2
    "2.1_proportional": "exp_2_1_dense_rewards/generate_proportional.py",
    # Theme 4
    "4.3_curated_1k": "exp_4_3_skill_distillation/generate_curated_1k.py",
    # Theme 5
    "5.1_solvable": "exp_5_1_consistency_verification/generate_filtered_solvable.py",
    "5.2_adversarial": "exp_5_2_adversarial_tests/generate_adversarial.py",
    "5.4_2skill": "exp_5_4_compositional_tasks/generate_2skill.py",
    # Theme 7
    "7.1_d1_full": "exp_7_1_dockerfile_curriculum/generate_d1_full.py",
    "7.1_d5_bare": "exp_7_1_dockerfile_curriculum/generate_d5_bare.py",
    "7.2_random_30": "exp_7_2_info_dropout/generate_random_30.py",
    "7.3_go": "exp_7_3_cross_language/generate_go.py",
    "7.4_pytest_reverse": "exp_7_4_test_first/generate_pytest_reverse.py",
    "7.5_staged": "exp_7_5_staged_rewards/generate_staged.py",
}

for name, script_path in GEN_SCRIPTS.items():
    EXPERIMENTS[name] = lambda sp=script_path: _gen_and_find_dir(sp)


# ─── Harbor testing ─────────────────────────────────────────────

def run_harbor_passrate(task_dir: str, name: str) -> dict:
    """Run harbor pass rate test on a local task directory."""
    print(f"\n--- Harbor test: {name} ---")
    job_name = f"other_{name}_{os.getpid()}"

    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", "8",
        "--agent", "terminus-2",
        "--model", AGENT_MODEL,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", CONFIG,
    ]

    print(f"  Running harbor ({AGENT_MODEL})...")
    os.chdir(str(PROJECT_ROOT))
    env = os.environ.copy()
    secret_env = Path("/scratch/10000/eguha3/old-dc-agent/secret.env")
    if secret_env.exists():
        for line in secret_env.read_text().splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                key, val = line[7:].split("=", 1)
                env[key] = os.path.expandvars(val)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if result.returncode != 0 and result.stderr:
        print(f"  stderr: {result.stderr[:300]}")

    job_dir = PROJECT_ROOT / "jobs" / job_name
    if not job_dir.exists():
        print(f"  No job directory found")
        return {"name": name, "error": "no job dir"}

    passed = total = errors = 0
    for trial_dir in sorted(job_dir.iterdir()):
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            data = json.load(f)
        vr = data.get("verifier_result")
        if vr and "rewards" in vr:
            reward = vr["rewards"].get("reward", 0)
            total += 1
            if reward > 0:
                passed += 1
        elif data.get("exception_info"):
            errors += 1
            total += 1

    if total == 0:
        return {"name": name, "error": "no results"}

    rate = (passed / total * 100) if total > 0 else 0
    print(f"  Result: {passed}/{total} ({rate:.1f}%) [{errors} errors]")
    return {"name": name, "passed": passed, "total": total, "pass_rate": rate, "errors": errors}


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--experiments", nargs="+",
                        help="Filter by prefix (e.g. '1.1' '7.3')")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.experiments:
        exps = {k: v for k, v in exps.items()
                if any(k.startswith(p) for p in args.experiments)}
    print(f"Running {len(exps)} experiments with limit={LIMIT}")

    manifest_path = PROJECT_ROOT / "data" / "other_themes_manifest.json"
    task_dirs = {}

    # Phase 1: Generate
    if not args.test_only:
        print(f"\n{'#'*60}")
        print(f"PHASE 1: GENERATING ({len(exps)} experiments, {LIMIT} tasks each)")
        print(f"{'#'*60}")

        for name, gen_fn in sorted(exps.items()):
            print(f"\n{'='*60}")
            print(f"GENERATING: {name}")
            print(f"{'='*60}")
            try:
                task_dir = gen_fn()
                if task_dir and Path(task_dir).exists():
                    n = len(list(Path(task_dir).glob("*/instruction.md")))
                    task_dirs[name] = task_dir
                    print(f"  -> OK: {n} tasks at {task_dir}")
                else:
                    print(f"  -> FAILED (no output dir)")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback; traceback.print_exc()

        print(f"\nGeneration: {len(task_dirs)}/{len(exps)} succeeded")
        # Merge with existing manifest (don't overwrite entries from other runs)
        existing = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = json.load(f)
        existing.update(task_dirs)
        with open(manifest_path, "w") as f:
            json.dump(existing, f, indent=2)

    if args.generate_only:
        print("--generate-only: done")
        return

    # Phase 2: Harbor tests (parallel)
    if args.test_only:
        if manifest_path.exists():
            with open(manifest_path) as f:
                task_dirs = json.load(f)
            if args.experiments:
                task_dirs = {k: v for k, v in task_dirs.items()
                             if any(k.startswith(p) for p in args.experiments)}
            print(f"Loaded {len(task_dirs)} from manifest")
        else:
            print("No manifest. Run without --test-only first.")
            return

    print(f"\n{'#'*60}")
    print(f"PHASE 2: HARBOR PASS RATE ({len(task_dirs)} experiments)")
    print(f"Model: {AGENT_MODEL}")
    print(f"{'#'*60}")

    # Launch all harbor tests in parallel
    import concurrent.futures
    all_results = []

    def _test_one(name_and_dir):
        name, task_dir = name_and_dir
        if not Path(task_dir).exists():
            return {"name": name, "error": "dir missing"}
        try:
            return run_harbor_passrate(task_dir, name)
        except Exception as e:
            return {"name": name, "error": str(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_dirs)) as pool:
        futures = {pool.submit(_test_one, (n, d)): n for n, d in task_dirs.items()}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            all_results.append(r)
            if "error" not in r:
                print(f"  DONE {r['name']}: {r['passed']}/{r['total']} ({r['pass_rate']:.1f}%)")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Experiment':<30} {'Pass Rate':>12} {'Detail':>12}")
    print("-" * 56)
    for r in sorted(all_results, key=lambda x: x["name"]):
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>12} {r['error']}")
        else:
            print(f"{r['name']:<30} {r['pass_rate']:>11.1f}% {r['passed']}/{r['total']}")

    with open(PROJECT_ROOT / "data" / "other_themes_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to data/other_themes_results.json")


if __name__ == "__main__":
    main()
