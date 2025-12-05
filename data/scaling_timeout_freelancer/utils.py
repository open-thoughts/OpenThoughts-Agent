import tempfile
import shutil
from pathlib import Path
from typing import Optional
import toml
from data.gcs_cache import gcs_cache

def modify_task_timeout(task_dir: Path, timeout_sec: float) -> None:
    """
    Modify the agent timeout in a task's task.toml file.

    Args:
        task_dir: Path to the task directory
        timeout_sec: New timeout value in seconds
    """
    task_toml_path = task_dir / "task.toml"

    if not task_toml_path.exists():
        print(f"Warning: task.toml not found in {task_dir}")
        return

    # Read the TOML file
    with open(task_toml_path, 'r') as f:
        config = toml.load(f)

    # Update the timeout
    if 'agent' not in config:
        config['agent'] = {}
    config['agent']['timeout_sec'] = timeout_sec

    # Write back
    with open(task_toml_path, 'w') as f:
        toml.dump(config, f)

@gcs_cache()
def set_timeout_for_all_tasks(dataset_dir: str, timeout_sec: float) -> str:
    """
    Create a new dataset directory with modified timeouts for all tasks.

    Args:
        dataset_dir: Path to the source dataset directory
        timeout_sec: Timeout value in seconds to set for all tasks

    Returns:
        str: Path to the new dataset directory with modified timeouts
    """
    source_path = Path(dataset_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    # Create temporary directory for modified tasks
    temp_dir = Path(tempfile.mkdtemp(prefix=f"timeout_{int(timeout_sec)}s_"))
    print(f"Creating dataset with {timeout_sec}s timeout in: {temp_dir}")

    # Get all task directories
    task_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if len(task_dirs) == 0:
        raise ValueError(f"No task directories found in: {dataset_dir}")

    # Copy tasks and modify timeouts
    for i, task_dir in enumerate(task_dirs):
        dest_path = temp_dir / task_dir.name
        shutil.copytree(task_dir, dest_path)
        modify_task_timeout(dest_path, timeout_sec)

        if (i + 1) % 100 == 0:
            print(f"Modified {i + 1} tasks...")

    print(f"Successfully modified timeout for {len(task_dirs)} tasks to {timeout_sec}s!")
    return str(temp_dir)
