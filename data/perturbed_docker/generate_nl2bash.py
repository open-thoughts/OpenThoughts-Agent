"""
Generate controlled perturbations for Dockerfiles to increase difficulty
using nl2bash instructions as the question source.
"""

from datasets import load_dataset

from data.commons import (
    create_task_from_dockerfiles_and_questions,
    filter_tasks_by_docker_start_working,
    upload_tasks_to_hf,
    upsample_tasks_directory,
)
from data.gcs_cache import gcs_cache
from data.perturbed_docker.utils import generate_dockerfiles_with_perturbations


@gcs_cache()
def generate_nl2bash_instructions(limit: int, offset: int) -> list[str]:
    ds = load_dataset("westenfelder/NL2SH-ALFA", split="train")
    end = min(len(ds), offset + max(0, limit))
    return [x.strip() for x in ds.select(range(offset, end))["nl"]]


def main() -> None:
    questions = generate_nl2bash_instructions(limit=10_000, offset=0)
    dockerfiles = generate_dockerfiles_with_perturbations(questions)
    output_dir = create_task_from_dockerfiles_and_questions(dockerfiles, questions)
    print(f"Generated dataset at: {output_dir}")
    filtered_dataset_dir = filter_tasks_by_docker_start_working(output_dir)
    print(f"Filtered dataset at: {filtered_dataset_dir}")
    upsampled_dataset_dir = upsample_tasks_directory(filtered_dataset_dir, 10_000)
    upload_tasks_to_hf(
        upsampled_dataset_dir, "DCAgent/perturbed-docker-exp-nl2bash-tasks-1"
    )


if __name__ == "__main__":
    main()
