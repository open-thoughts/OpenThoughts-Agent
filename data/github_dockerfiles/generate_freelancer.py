from data.taskmaster2.generate import download_taskmaster2_data, extract_questions_from_conversations
from data.commons import generate_tasks_from_questions, subsample_tasks_directory, create_task_from_dockerfiles_and_questions, upload_tasks_to_hf, upload_traces_to_hf, filter_tasks_by_docker_start, upsample_tasks_directory
from data.github_dockerfiles.utils import convert_prompt_to_dockerfile_batched_sync
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from harbor.models.environment_type import EnvironmentType
from data.freelancer.generate import scrape_freelancer_projects

def main() -> None:
    questions = scrape_freelancer_projects(
        use_discovery=True,
        max_projects=None,
        use_all_categories=True
    )
    dockerfiles = convert_prompt_to_dockerfile_batched_sync(questions)
    final_dataset_dir = create_task_from_dockerfiles_and_questions(dockerfiles, questions)
    filtered_dataset_dir = filter_tasks_by_docker_start(final_dataset_dir, max_concurrent=32)
    upsampled_dataset_dir = upsample_tasks_directory(filtered_dataset_dir, 10_000)
    hf_dataset = run_dataset_to_traces(upsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=128, agent_kwargs={"max_episodes": 8})
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/github-dockerfiles-docker-exp-freelancer-tasks")
    upload_traces_to_hf(hf_dataset, "DCAgent/github-dockerfiles-docker-exp-freelancer", "SFT")

if __name__ == "__main__":
    main()
