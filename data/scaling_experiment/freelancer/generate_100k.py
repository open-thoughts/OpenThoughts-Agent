import time
from data.freelancer.generate import (
    generate_tasks_from_questions,
    run_dataset_to_traces,
    scrape_freelancer_projects,
    upload_tasks_to_hf,
    upload_traces_to_hf,
    upsample_tasks_directory,
)

def main() -> None:
    """Main function - coordinates the pipeline"""
    t1 = time.time()
    questions = scrape_freelancer_projects(
        use_discovery=True,
        max_projects=None,
        use_all_categories=True
    )
    final_dataset_dir = generate_tasks_from_questions(questions, "freelancer")
    upsampled_dataset_dir = upsample_tasks_directory(final_dataset_dir, 100_000)
    t2 = time.time()
    print(t2 - t1)
    hf_dataset = run_dataset_to_traces(upsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=4500, agent_kwargs={"max_episodes": 8})
    print("done")
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-projects-100k-traces", "SFT")
    upload_tasks_to_hf(upsampled_dataset_dir, "DCAgent/freelancer-projects-100k")

if __name__ == "__main__":
    main()
