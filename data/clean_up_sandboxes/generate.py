from data.commons import download_hf_dataset, upload_tasks_to_hf
from data.clean_up_sandboxes.utils import filter_failed_tasks
import tempfile
import shutil
from pathlib import Path
from typing import List

def clean_up_sandboxes(dataset_path: str, failed_tasks: List[str]) -> str:
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="clean_sandboxes_"))
    print(f"Creating clean tasks directory at: {temp_dir}")
    
    # Get all task directories from the dataset
    dataset_path = Path(dataset_path)
    all_tasks = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # Filter out failed tasks and tasks containing "feal", then copy the rest
    clean_task_count = 0
    for task_dir in all_tasks:
        task_name = task_dir.name
        task_path = str(task_dir)

        if task_path not in failed_tasks and "feal" not in task_name:
            # Copy the task directory to temp directory
            dest_path = temp_dir / task_name
            shutil.copytree(task_path, dest_path)
            clean_task_count += 1
        else:
            if task_path in failed_tasks:
                print(f"Skipping failed task: {task_path}")
            elif "feal" in task_name:
                print(f"Skipping task containing 'feal': {task_name}")
    
    return temp_dir

def generate_clean_up_sandboxes():
    dataset_path  = download_hf_dataset("mlfoundations-dev/sandboxes-tasks")
    failed_tasks = filter_failed_tasks(dataset_path, num_concurrent=8)
    clean_tasks_dir = clean_up_sandboxes(dataset_path, failed_tasks)
    upload_tasks_to_hf(str(clean_tasks_dir), "mlfoundations-dev/clean-sandboxes-tasks-recleaned")
    

if __name__ == "__main__":
    generate_clean_up_sandboxes()