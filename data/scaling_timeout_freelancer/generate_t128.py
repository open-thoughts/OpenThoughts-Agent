from data.freelancer.generate import scrape_freelancer_projects
from data.commons import generate_tasks_from_questions, upsample_tasks_directory, upload_traces_to_hf
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.scaling_timeout_freelancer.utils import set_timeout_for_all_tasks

def main() -> None:
    """Main function - coordinates the pipeline with timeout=128s"""
    questions = scrape_freelancer_projects(
        use_discovery=True,
        max_projects=None,
        use_all_categories=True
    )
    final_dataset_dir = generate_tasks_from_questions(questions, "freelancer")
    timeout_dataset_dir = set_timeout_for_all_tasks(final_dataset_dir, timeout_sec=128.0)
    upsampled_dataset_dir = upsample_tasks_directory(timeout_dataset_dir, 10_000)
    hf_dataset = run_dataset_to_traces(
        upsampled_dataset_dir,
        model_name="gpt-5-nano-2025-08-07",
        agent_name="terminus-2",
        n_concurrent=256,
        agent_kwargs={"max_episodes": 32}
    )
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-t128s-32ep", "SFT")

if __name__ == "__main__":
    main()
