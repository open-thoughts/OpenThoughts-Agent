import datasets
from data.freelancer.generate import scrape_freelancer_projects, generate_tasks_from_questions, upsample_tasks_directory, run_dataset_to_traces, upload_traces_to_hf, upload_tasks_to_hf
from data.commons import upload_tasks_to_hf,  upload_traces_to_hf

def main() -> None:
    """Main function - coordinates the pipeline"""
    # questions = scrape_freelancer_projects(
    #     use_discovery=True,
    #     max_projects=None,
    #     use_all_categories=True
    # )
    # final_dataset_dir = generate_tasks_from_questions(questions, "freelancer")
    # upsampled_dataset_dir = upsample_tasks_directory(final_dataset_dir, 10_000)
    # hf_dataset = run_dataset_to_traces(upsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    hf_dataset = datasets.load_dataset("mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2", split="train")
    hf_dataset = hf_dataset.shuffle(seed=42)
    upload_traces_to_hf(hf_dataset, "DCAgent/freelancer-projects-10k-traces", "SFT")
    # upload_tasks_to_hf(upsampled_dataset_dir, "DCAgent/freelancer-projects-10k")

if __name__ == "__main__":
    main()