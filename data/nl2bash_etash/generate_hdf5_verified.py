#!/usr/bin/env python3
"""
HDF5-optimized version of generate.py for nl2bash dataset.

This version is much faster because:
1. Downloads dataset directly to HDF5 format (no directory extraction)
2. Computes checksum on single HDF5 file (much faster than directory tree)
3. Extracts to persistent directory for proper job resuming
4. Reuses extracted directory across runs to enable resuming
"""

from datasets import load_dataset
from data.commons import upload_traces_to_hf
from scripts.sandboxes.tasks_parquet_converter import from_hf_dataset_hdf5
from scripts.sandboxes.run_and_export_traces import run_dataset_to_traces_hdf5


def main():
    # Download HF dataset and convert to HDF5 format (fast)
    print("Step 1: Downloading dataset and converting to HDF5...")
    hdf5_dataset = from_hf_dataset_hdf5("DCAgent/nl2bash-verified")
    print(f"HDF5 dataset created: {hdf5_dataset}")

    # Run traces on HDF5 dataset (now with proper resuming support)
    print("\nStep 2: Running dataset to traces (HDF5-optimized with resuming)...")

    hf_dataset = run_dataset_to_traces_hdf5(
        hdf5_dataset,
        model_name="gpt-5-nano-2025-08-07",
        agent_name="terminus-2",
        n_concurrent=1_500,
        agent_kwargs={"max_episodes": 8}
    )

    # Upload traces to HuggingFace
    print("\nStep 3: Uploading traces to HuggingFace...")
    upload_traces_to_hf(
        hf_dataset,
        "EtashGuha/nl2bash_verified_gpt-5-nano-traces",
        "SFT"
    )

    print("\n✓ Pipeline completed successfully!")
    print(f"HDF5 file: {hdf5_dataset}")


if __name__ == "__main__":
    main()
