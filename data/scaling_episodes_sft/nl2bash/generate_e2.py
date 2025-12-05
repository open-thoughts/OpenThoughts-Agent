from datasets import load_dataset
from data.commons import upload_traces_to_hf
from data.scaling_episodes_sft.utils import subsample_traces

def main() -> None:
    """Load nl2bash dataset and subsample to 2 episodes"""
    hf_dataset = load_dataset("DCAgent/nl2bash-GLM-4.6-traces", split="train")
    hf_dataset = subsample_traces(hf_dataset, 2)
    upload_traces_to_hf(hf_dataset, "DCAgent/nl2bash-2ep", "SFT")

if __name__ == "__main__":
    main()
