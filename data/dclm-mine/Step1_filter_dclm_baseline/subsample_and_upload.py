#!/usr/bin/env python3
"""
Subsample sequences from sharded JSONL files and upload to Hugging Face.
Selects each sequence with probability 0.035.
"""

import json
import random
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

# Configuration
INPUT_DIR = Path("/data/horse/ws/rehe951g-dcagent/dclm-terminal-like/results_all_20251117_155756")
SAMPLE_PROBABILITY = 0.05
HF_DATASET_NAME = "DCAgent2/dclm-baseline-terminal-candidates-100k"

def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Collect all shard files
    shard_files = sorted(INPUT_DIR.glob("bash_sequences_shard_*.jsonl"))
    print(f"Found {len(shard_files)} shard files")

    # Sample sequences
    sampled_sequences = []
    total_count = 0

    for shard_file in shard_files:
        print(f"Processing {shard_file.name}...")
        with open(shard_file, 'r') as f:
            for line in f:
                total_count += 1
                if random.random() < SAMPLE_PROBABILITY:
                    sampled_sequences.append(json.loads(line))

        if total_count % 100000 == 0:
            print(f"  Processed {total_count} sequences, sampled {len(sampled_sequences)}")

    print(f"\nTotal sequences processed: {total_count}")
    print(f"Sampled sequences: {len(sampled_sequences)}")
    print(f"Actual sampling rate: {len(sampled_sequences) / total_count:.4f}")

    # Create Hugging Face dataset
    print("\nCreating Hugging Face dataset...")
    dataset = Dataset.from_list(sampled_sequences)
    print(f"Dataset created with {len(dataset)} examples")

    # Upload to Hugging Face
    print(f"\nUploading to {HF_DATASET_NAME}...")
    dataset.push_to_hub(HF_DATASET_NAME, private=False)
    print("Upload complete!")

if __name__ == "__main__":
    main()
