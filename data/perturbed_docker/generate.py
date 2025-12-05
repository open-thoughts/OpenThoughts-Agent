"""
Generate controlled perturbations for Dockerfiles to increase difficulty
"""

from data.commons import create_task_from_dockerfiles_and_questions, upload_traces_to_hf, upload_tasks_to_hf , filter_tasks_by_docker_start, upsample_tasks_directory
from data.perturbed_docker.utils import generate_dockerfiles_with_perturbations
from data.taskmaster2.generate import download_taskmaster2_data, extract_questions_from_conversations
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from harbor.models.environment_type import EnvironmentType


def main() -> None:
    conversations = download_taskmaster2_data()
    questions = extract_questions_from_conversations(conversations)
    dockerfiles = generate_dockerfiles_with_perturbations(questions)
    output_dir = create_task_from_dockerfiles_and_questions(dockerfiles, questions)
    filtered_dataset_dir = filter_tasks_by_docker_start(output_dir, max_concurrent=256)
    upsampled_dataset_dir = upsample_tasks_directory(filtered_dataset_dir, 10_000)
    hf_dataset = run_dataset_to_traces(output_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/perturbed-docker-exp-taskmaster2-tasks")
    upload_traces_to_hf(hf_dataset, "DCAgent/perturbed-docker-exp-taskmaster2-traces", "SFT")


if __name__ == "__main__":
    main()
