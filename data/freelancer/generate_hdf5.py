#!/usr/bin/env python3

from data.commons import (
    generate_tasks_from_questions_hdf5,
    upsample_tasks_hdf5,
    extract_hdf5_to_task_paths,
    upload_tasks_to_hf,
    upload_traces_to_hf
)
from data.freelancer.generate import scrape_freelancer_projects
from scripts.sandboxes.run_and_export_traces import run_dataset_to_traces


def main_hdf5() -> None:
    print("=" * 80)
    print("FREELANCER DATASET GENERATION - HDF5 OPTIMIZED")
    print("=" * 80)

    print("\n[1/5] Scraping freelancer projects...")
    questions = scrape_freelancer_projects(
        use_discovery=True,
        max_projects=None,
        use_all_categories=True
    )
    print(f"Scraped {len(questions)} project descriptions")

    print("\n[2/5] Generating tasks into HDF5 format...")
    hdf5_dataset = generate_tasks_from_questions_hdf5(questions, "freelancer")
    print(f"Created HDF5 dataset: {hdf5_dataset}")

    print("\n[3/5] Upsampling HDF5 dataset to 10,000 tasks...")
    upsampled_hdf5 = upsample_tasks_hdf5(hdf5_dataset, 10_000)
    print(f"Upsampled HDF5 dataset: {upsampled_hdf5}")

    print("\n[4/5] Extracting HDF5 to directory for trace generation...")
    extracted_dir_paths = extract_hdf5_to_task_paths(upsampled_hdf5)
    if not extracted_dir_paths:
        raise RuntimeError("Failed to extract any tasks from the HDF5 file.")
    extracted_dir = str(extracted_dir_paths[0]).rsplit('/', 1)[0]

    print(f"Running trace generation on {len(extracted_dir_paths)} tasks...")
    hf_dataset = run_dataset_to_traces(
        extracted_dir,
        model_name="gpt-5-nano-2025-08-07",
        agent_name="terminus-2",
        n_concurrent=256,
        agent_kwargs={"max_episodes": 8}
    )

    print("\n[5/5] Uploading results to HuggingFace...")
    upload_traces_to_hf(
        hf_dataset,
        "mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2",
        "SFT"
    )
    upload_tasks_to_hf(
        extracted_dir,
        "mlfoundations-dev/freelancer-projects-sandboxes"
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"HDF5 dataset: {upsampled_hdf5}")
    print(f"Extracted directory: {extracted_dir}")
    print(f"Traces uploaded to: mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2")
    print(f"Tasks uploaded to: mlfoundations-dev/freelancer-projects-sandboxes")


if __name__ == "__main__":
    main_hdf5()
