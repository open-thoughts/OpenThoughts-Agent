#!/usr/bin/env python3

from data.commons import (
    upload_traces_to_hf
)
from datasets import load_dataset


def main() -> None:
    """Main function - coordinates the pipeline using HDF5 format"""



    ## I AM SO SORRY< I HAD TO FORCE RESUME THIS JOB, THIS WAS THE CODE, THIS WAS BEFORE RUN_DATASET_TO_TRACES_HDF5 WORKED I AM RESUMING FROM THIS EXISTING DATSET :(

    dataset = load_dataset("DCAgent2/freelancer-projects-100k-traces")
    subset = dataset['train'].shuffle(seed=42).select(range(31_600))
    upload_traces_to_hf(
        subset,
        "DCAgent2/freelancer-projects-31k-traces",
        "SFT"
    )


if __name__ == "__main__":
    main()
