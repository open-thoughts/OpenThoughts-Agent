import json
from pathlib import Path
from collections import defaultdict

def find_passing_tasks(job_dirs):
    """Find tasks with pass@3 > 0 in the given job directories."""

    # Group trials by task name
    task_results = defaultdict(list)

    for job_dir in job_dirs:
        job_path = Path(job_dir)
        if not job_path.exists():
            print(f"Warning: {job_dir} does not exist")
            continue

        print(f"\nSearching in: {job_dir}")

        # Find all result.json files
        for result_file in job_path.rglob("result.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Extract task name and reward
                task_name = data.get('task_name', 'unknown')
                verifier_result = data.get('verifier_result')
                reward = verifier_result.get('reward', 0.0) if verifier_result else 0.0

                task_results[task_name].append({
                    'trial_name': data.get('trial_name', 'unknown'),
                    'reward': reward,
                    'file': str(result_file)
                })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Error reading {result_file}: {e}")

    # Calculate pass@k for each task
    passing_tasks = []

    print(f"\n{'='*60}")
    print("Calculating pass@k metrics...")
    print(f"{'='*60}")

    for task_name, trials in task_results.items():
        # Sort by reward
        trials.sort(key=lambda x: x['reward'], reverse=True)

        # Calculate pass@k (at least one success in k trials)
        num_trials = len(trials)
        num_successes = sum(1 for t in trials if t['reward'] > 0)

        # pass@k is 1 if any of the first k trials succeeded
        pass_at_1 = 1.0 if num_trials >= 1 and trials[0]['reward'] > 0 else 0.0
        pass_at_2 = 1.0 if num_trials >= 2 and any(t['reward'] > 0 for t in trials[:2]) else 0.0
        pass_at_3 = 1.0 if num_trials >= 3 and any(t['reward'] > 0 for t in trials[:3]) else 0.0

        if pass_at_3 > 0:
            passing_tasks.append({
                'task_name': task_name,
                'num_trials': num_trials,
                'num_successes': num_successes,
                'pass@1': pass_at_1,
                'pass@2': pass_at_2,
                'pass@3': pass_at_3,
                'trials': trials
            })
            print(f"  ✓ {task_name}")
            print(f"    Trials: {num_trials}, Successes: {num_successes}")
            print(f"    pass@1: {pass_at_1}, pass@2: {pass_at_2}, pass@3: {pass_at_3}")

    return passing_tasks

if __name__ == "__main__":
    job_directories = [
        "/Users/etashguha/Documents/ot-agent/jobs/bespoke_verification_v2",
        "/Users/etashguha/Documents/ot-agent/jobs/bespoke_verification_v2_try1",
        "/Users/etashguha/Documents/ot-agent/jobs/bespoke_verification_v2_try3"
    ]

    print("Finding tasks with pass@3 > 0...")
    passing = find_passing_tasks(job_directories)

    print(f"\n{'='*60}")
    print(f"Summary: Found {len(passing)} tasks with pass@3 > 0")
    print(f"{'='*60}")

    if passing:
        print("\nTask names with pass@3 > 0:")
        for task_info in passing:
            print(f"  {task_info['task_name']}")
