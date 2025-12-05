import datasets
from data.commons import upload_traces_to_hf

def main() -> None:
    """Main function - subsamples nl2bash-GLM-4.6-traces for 0-1k scale"""
    hf_dataset = datasets.load_dataset("DCAgent/nl2bash-GLM-4.6-traces", split="train")
    hf_dataset = hf_dataset.shuffle(seed=42).select(range(100))
    upload_traces_to_hf(hf_dataset, "DCAgent/nl2bash-0-1k-traces", "SFT")

if __name__ == "__main__":
    main()
