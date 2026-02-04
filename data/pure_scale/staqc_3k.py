#!/usr/bin/env python3
"""
Subsample the existing STAQC GLM traces dataset to 3.16K samples.
Source: DCAgent/exp-psu-stackoverflow-31K_glm_4.7_traces
"""

import random
from datasets import load_dataset

SOURCE_DATASET = "DCAgent/exp-psu-stackoverflow-31K_glm_4.7_traces"
TARGET_DATASET = "DCAgent/exp-psu-stackoverflow-3K_glm_4.7_traces"
SAMPLE_SIZE = 3_160


def main() -> None:
    print(f"Loading source dataset: {SOURCE_DATASET}")
    dataset = load_dataset(SOURCE_DATASET, split="train")
    print(f"Source dataset size: {len(dataset)}")

    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)

    subset = dataset.select(indices[:SAMPLE_SIZE])
    print(f"Uploading {len(subset)} samples to {TARGET_DATASET}")
    subset.push_to_hub(TARGET_DATASET, private=False)
    print(f"Done: {TARGET_DATASET}")


if __name__ == "__main__":
    main()
