import shutil
import tempfile
from pathlib import Path
from data.gcs_cache import gcs_cache

@gcs_cache()
def dedup_tasks(tasks_path: str | Path) -> Path:
    """Dedup tasks by instruction.md content, return path to new directory with unique tasks."""
    tasks_path = Path(tasks_path)
    out_dir = Path(tempfile.mkdtemp())
    seen = set()
    total = 0
    for task_dir in sorted(tasks_path.iterdir()):
        total += 1
        instruction = (task_dir / "instruction.md").read_text()
        if instruction not in seen:
            seen.add(instruction)
            shutil.copytree(task_dir, out_dir / task_dir.name)
    print(f"{total - len(seen)} removed after dedup.")
    return out_dir