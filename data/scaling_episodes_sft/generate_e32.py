from data.taskmaster2.generate import download_taskmaster2_data, extract_questions_from_conversations
from data.commons import generate_tasks_from_questions, subsample_tasks_directory, upload_traces_to_hf
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.scaling_episodes_sft.utils import subsample_traces

def main() -> None:
    """Main function - coordinates the pipeline with temp directories"""
    conversations = download_taskmaster2_data()
    questions = extract_questions_from_conversations(conversations)
    final_dataset_dir = generate_tasks_from_questions(questions, "taskmaster")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 64})
    hf_dataset = subsample_traces(hf_dataset, 32)
    upload_traces_to_hf(hf_dataset, "DCAgent/taskmaster2-32ep", "SFT")
    
if __name__ == "__main__":
    main()