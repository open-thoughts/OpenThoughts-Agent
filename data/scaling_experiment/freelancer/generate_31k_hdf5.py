#!/usr/bin/env python3

from data.commons import (
    generate_tasks_from_questions_hdf5,
    upsample_tasks_hdf5,
    extract_hdf5_to_task_paths,
    upload_tasks_to_hf,
    upload_traces_to_hf
)
from datasets import load_dataset
import time
from data.freelancer.generate import scrape_freelancer_projects
from scripts.sandboxes.run_and_export_traces import run_dataset_to_traces, only_export_traces, force_resume


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
