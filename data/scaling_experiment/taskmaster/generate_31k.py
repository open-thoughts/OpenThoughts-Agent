import datasets
from data.taskmaster2.generate import download_taskmaster2_data, extract_questions_from_conversations, generate_tasks_from_questions, subsample_tasks_directory, upload_tasks_to_hf, run_dataset_to_traces, upload_traces_to_hf

def main() -> None:
    """Main function - coordinates the pipeline with temp directories"""
    conversations = download_taskmaster2_data()
    questions = extract_questions_from_conversations(conversations)
    final_dataset_dir = generate_tasks_from_questions(questions, "taskmaster2")
    upsampled_dataset_dir = upsample_tasks_directory(subsampled_dataset_dir, 100_000)

    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/taskmaster2-10k")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    hf_dataset = datasets.load_dataset("mlfoundations-dev/taskmaster2-sandboxes-traces-terminus-2", split="train")
    hf_dataset = hf_dataset.shuffle(seed=42)
    upload_traces_to_hf(hf_dataset, "DCAgent/taskmaster2-10k-traces", "SFT")
    
if __name__ == "__main__":
    main()