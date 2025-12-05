import json
import shutil
from pathlib import Path
from collections import defaultdict

def find_passing_tasks(job_dir):
    """Find tasks with pass@3 > 0 in the given job directory."""

    # Group trials by task name (extracted from folder prefix before __)
    task_results = defaultdict(list)

    job_path = Path(job_dir)
    if not job_path.exists():
        print(f"Warning: {job_dir} does not exist")
        return []

    print(f"\nSearching in: {job_dir}")

    # Find all result.json files
    for result_file in job_path.rglob("result.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Extract task name from the folder name (part before __)
            folder_name = result_file.parent.name
            task_name = folder_name.split('__')[0] if '__' in folder_name else folder_name
            
            verifier_result = data.get('verifier_result')
            reward = verifier_result.get('reward', 0.0) if verifier_result else 0.0

            task_results[task_name].append({
                'trial_name': data.get('trial_name', folder_name),
                'reward': reward,
                'file': str(result_file),
                'folder': folder_name
            })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error reading {result_file}: {e}")

    # Find tasks with at least one success
    passing_tasks = []

    print(f"\n{'='*60}")
    print("Finding tasks solved at least once...")
    print(f"{'='*60}")

    for task_name, trials in task_results.items():
        # Sort by reward
        trials.sort(key=lambda x: x['reward'], reverse=True)

        # Check if task has at least one success
        num_trials = len(trials)
        num_successes = sum(1 for t in trials if t['reward'] > 0)

        if num_successes > 0:
            passing_tasks.append({
                'task_name': task_name,
                'num_trials': num_trials,
                'num_successes': num_successes,
                'trials': trials
            })
            print(f"  ✓ {task_name}")
            print(f"    Trials: {num_trials}, Successes: {num_successes}")

    return passing_tasks

def create_successful_runs_folders(passing_tasks, output_dir):
    """Create folders with task descriptions and successful trials."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for task_info in passing_tasks:
        task_name = task_info['task_name']
        trials = task_info['trials']
        
        # Find the best successful trial (highest reward)
        successful_trials = [t for t in trials if t['reward'] > 0]
        if not successful_trials:
            continue
            
        best_trial = max(successful_trials, key=lambda x: x['reward'])
        
        # Create task folder
        task_folder = output_path / task_name
        task_folder.mkdir(exist_ok=True)
        
        print(f"Creating folder for task: {task_name}")
        
        # Copy the successful trial folder
        source_folder = Path(best_trial['file']).parent
        dest_folder = task_folder / "successful_trial"
        
        if source_folder.exists():
            if dest_folder.exists():
                shutil.rmtree(dest_folder)
            shutil.copytree(source_folder, dest_folder)
            
        # Extract task path from config.json and copy task description
        config_file = source_folder / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                task_path = config_data.get('task', {}).get('path')
                if task_path and Path(task_path).exists():
                    # Copy the entire task directory
                    task_source = Path(task_path)
                    task_dest = task_folder / "task_description"
                    
                    if task_dest.exists():
                        shutil.rmtree(task_dest)
                    shutil.copytree(task_source, task_dest)
                    print(f"    Copied task description from: {task_path}")
                else:
                    print(f"    Warning: Task path not found: {task_path}")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    Error reading config.json: {e}")
        else:
            print(f"    Warning: config.json not found in {source_folder}")

if __name__ == "__main__":
    job_directory = "/Users/etash/Documents/ot-agent/jobs/bespoke_from_sandboxes_v6"

    print("Finding tasks solved at least once...")
    passing = find_passing_tasks(job_directory)

    print(f"\n{'='*60}")
    print(f"Summary: Found {len(passing)} tasks solved at least once")
    print(f"{'='*60}")

    if passing:
        print("\nTask names solved at least once:")
        for task_info in passing:
            print(f"  {task_info['task_name']}")
            
        print(f"\nCreating successful_runs folders...")
        create_successful_runs_folders(passing, "/Users/etash/Documents/ot-agent/successful_runs_again")
        print(f"\nSuccessful runs folders created in: /Users/etash/Documents/ot-agent/successful_runs_again")
