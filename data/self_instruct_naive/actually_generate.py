from datasets import load_dataset
from data.commons import upload_traces_to_hf
from scripts.harbor.tasks_parquet_converter import from_hf_dataset
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
def main():
    dataset = from_hf_dataset("DCAgent/selfinstruct-naive-sandboxes-2")
    hf_dataset = run_dataset_to_traces(dataset, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "DCAgent/selfinstruct-naive-sandboxes-2-traces", "SFT")

if __name__ == "__main__":
    main()
