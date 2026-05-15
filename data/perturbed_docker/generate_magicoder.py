"""
Generate controlled perturbations for Dockerfiles to increase difficulty
"""

from data.commons import (
    create_task_from_dockerfiles_and_questions,
    filter_tasks_by_docker_start_working,
    upload_tasks_to_hf,
    upsample_tasks_directory,
)
from data.magicoder.generate import generate_magicoder_instructions
from data.perturbed_docker.utils import generate_dockerfiles_with_perturbations


def main() -> None:
    questions = generate_magicoder_instructions(limit=10_000, offset=110_000)
    dockerfiles = generate_dockerfiles_with_perturbations(questions)
    output_dir = create_task_from_dockerfiles_and_questions(dockerfiles, questions)
    print(f"Generated dataset at: {output_dir}")
    filtered_dataset_dir = filter_tasks_by_docker_start_working(output_dir)
    print(f"Filtered dataset at: {filtered_dataset_dir}")
    upsampled_dataset_dir = upsample_tasks_directory(filtered_dataset_dir, 10_000)
    upload_tasks_to_hf(
        upsampled_dataset_dir, "DCAgent/perturbed-docker-exp-magicoder-tasks-12"
    )
    # hf_dataset = run_dataset_to_traces(output_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    # upload_traces_to_hf(hf_dataset, "DCAgent/perturbed-docker-exp-magicoder-traces", "SFT")


if __name__ == "__main__":
    main()
