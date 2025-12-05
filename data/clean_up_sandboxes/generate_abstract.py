from typing import Optional

from data.commons import (
    download_hf_dataset,
    finalize_dataset_output,
    upload_tasks_to_hf,
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
)
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

class CleanUpSandboxesGenerator(BaseDataGenerator):
    parser_description = "Clean OT-Agent sandboxes dataset"
    default_target_repo = "mlfoundations-dev/clean-sandboxes-tasks-recleaned"

    def add_arguments(self, parser):
        parser.add_argument(
            "--source-repo",
            type=str,
            default="mlfoundations-dev/sandboxes-tasks",
            help="Source HuggingFace dataset to clean",
        )
        parser.add_argument(
            "--num-concurrent",
            type=int,
            default=8,
            help="Number of concurrent processes for failure detection",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        dataset_path = download_hf_dataset(args.source_repo)
        failed_tasks = filter_failed_tasks(dataset_path, num_concurrent=args.num_concurrent)
        temp_clean_dir = clean_up_sandboxes(dataset_path, failed_tasks)
        output_path = finalize_dataset_output(temp_clean_dir, getattr(args, "output_dir", None))
        dataset_output = str(output_path)

        limit = self._resolve_limit(request)
        if limit and limit > 0:
            from data.generation.limits import limit_tasks_directory

            clean_tasks_dir = Path(
                limit_tasks_directory(dataset_output, limit, dataset_prefix="clean_tasks")
            )
            print(f"Limited cleaned tasks to {limit} entries")
            dataset_output = str(clean_tasks_dir)

        uploaded_repo: Optional[str] = None
        if not getattr(args, "no_upload", False):
            target_repo = args.target_repo or self.default_target_repo
            if not target_repo:
                raise ValueError("No target repo specified")
            upload_tasks_to_hf(dataset_output, target_repo)
            uploaded_repo = target_repo
        else:
            print(f"Skipping upload. Clean dataset available at: {dataset_output}")

        return GenerationResult(
            dataset_path=dataset_output,
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
            metadata={"failed_tasks": len(failed_tasks)},
        )


def main() -> None:
    CleanUpSandboxesGenerator().run()


if __name__ == "__main__":
    main()
