import json
import os
import re
import subprocess as sp
import sys
import tempfile
import uuid
from datetime import datetime

import wandb
import yaml
from huggingface_hub import HfApi, repo_exists, upload_folder, whoami

from typing import Any


def sync_wandb():
    if wandb.run:
        wandb.run.sync()


def upload_to_db(model_configs: Any):
    """
    Upload the given model_config object to the database.
    Establishes a database connection using an engine and session maker, then adds the `model_configs` to the database.

    Args:
        model_configs (Model): The model configuration created from generate_model_configs()
    """
    raise NotImplementedError("Database integration is disabled in dc-agent")


def create_model_repo(model_name):
    api = HfApi()
    org = model_name.split("/")[0]
    model_name = model_name.replace(f"{org}/", "")
    model_name = model_name.split("/")[-1]
    print(len(model_name))
    try:
        repo_url = api.create_repo(
            repo_id=f"{org}/{model_name}", repo_type="model", exist_ok=True
        )
        print(f"Created repo: {repo_url}")
    except Exception as e:
        print(f"Error when creating repo: {e}")

    return f"{org}/{model_name}"


def clean_readme(output_dir):
    with open(os.path.join(output_dir, "README.md"), "r") as file:
        readme = file.read()

    models_dir = os.environ.get("MODELS_DIR", "")
    datasets_dir = os.environ.get("DATASETS_DIR", "")
    readme = readme.replace(models_dir, "").replace(datasets_dir, "")

    with open(os.path.join(output_dir, "README.md"), "w") as file:
        file.write(readme)


def upload_repo_to_hf(model_name, model_path):
    clean_readme(model_path)
    ignore_patterns = [
        "wandb/*",
        "checkpoint-*/*",
        "*start_end.json",
        "eval/*",
        "*.png",
    ]
    if repo_exists(model_name):
        print(f"Model {model_name} already exists, will overwrite.")
    try:
        upload_folder(
            folder_path=model_path,
            repo_id=model_name,
            repo_type="model",
            ignore_patterns=ignore_patterns,
        )
        print(f"Uploaded model to {model_name}")
    except Exception as e:
        print(f"Error when uploading model: {e}")


def wandb_sync(output_dir):
    wandb_dir = os.path.join(output_dir, "wandb", "latest-run")
    if not os.path.exists(wandb_dir):
        print(f"Wandb directory {wandb_dir} does not exist.")
        return
    cmd = f"wandb sync {wandb_dir} --clean-force"
    cmd_debug_output = sp.check_output(cmd, shell=True, universal_newlines=True)

    wandb_regex_match = re.search(r"https://wandb.ai/.*/runs/.* ...", cmd_debug_output)
    if not wandb_regex_match:
       print(f"Wandb run URL not found in command output: {cmd_debug_output}")
       return
    wandb_run_url = wandb_regex_match.group(0).split("...")[0].strip()
    return wandb_run_url


def check_model_exists(hf_model: str):
    raise NotImplementedError("Database integration is disabled in dc-agent")


def generate_model_configs(
    train_yaml,
    start_time,
    end_time,
    base_model,
    wandb_run_url=None,
):
    """
    Takes in parsed arguments and extract necessary fields for the Model object.

    Args:
        args taken from output of run_exp()
        start_time: datetime.now() called when run started
    Returns:
        Model: A model configuration object containing the relevant metadata to be uplaoded to DB.
    """
    raise NotImplementedError("Database-backed model metadata generation is disabled in dc-agent")


def upload_to_db(model_configs: Any):
    """
    Upload the given model_config object to the database.
    Establishes a database connection using an engine and session maker, then adds the `model_configs` to the database.

    Args:
        model_configs (Model): The model configuration created from generate_model_configs()
    """
    raise NotImplementedError("Database integration is disabled in dc-agent")


def upload_to_hf(training_parameters):
    """
    training_parameters is a dict corresponding to the training yaml file.

    This function creates a temporary yaml of it then uploads that yaml to the HF repo in hub_model_id.
    """
    if "hub_model_id" not in training_parameters:
        print(f"hub_model_id not found in parameters: {training_parameters}")
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as temp_file:
            yaml.dump(training_parameters, temp_file)
            temp_file_path = temp_file.name  # Get the path to the temporary file

        api = HfApi()
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo="configs.yaml",  # The filename it will have in the Hugging Face repo
            repo_id=training_parameters["hub_model_id"],
            repo_type="model",  # "dataset" if it's a dataset repository
        )
        print("Model YAML uploaded to hf!")


def upload_local(
    config: str = "config.yaml",
    output_dir: str = None,
    base_model: str = None,
):
    # Database uploads are disabled in this repository; skipping DB setup

    assert output_dir is not None, "Output directory not provided."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."

    wandb_run_url = wandb_sync(output_dir)

    if os.path.exists(os.path.join(output_dir, "start_end.json")):
        with open(config, "r") as file:
            config_yaml = yaml.safe_load(file)
            hub_model_id = config_yaml["hub_model_id"]

        model_name = create_model_repo(hub_model_id)
        print(f"Model name: {model_name}")
        upload_repo_to_hf(model_name, output_dir)
        config_yaml["hub_model_id"] = model_name
        models_dir = os.environ.get("MODELS_DIR", "")
        config_yaml["model_name_or_path"] = config_yaml["model_name_or_path"].replace(
            models_dir, ""
        )
        with open(config, "w") as file:
            yaml.dump(config_yaml, file)

        with open(os.path.join(output_dir, "start_end.json"), "r") as file:
            start_end = json.load(file)
            start_time = start_end["start_time"]
            end_time = start_end["end_time"]
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        # Process README.md to fix model paths
        with open(os.path.join(output_dir, "README.md"), "r") as file:
            readme = file.read()
            # Fix base_model path in YAML header
            base_model_pattern = r"base_model: (.+?)(?=\n)"
            if re.search(base_model_pattern, readme):
                # Use the model_name which is the correct HF model ID
                readme = re.sub(base_model_pattern, f"base_model: {base_model}", readme)

            # Fix model-index name in YAML header
            model_index_pattern = r"- name: (.+?)(?=\n)"
            if re.search(model_index_pattern, readme):
                # Use the model_name which is the correct HF model ID
                readme = re.sub(model_index_pattern, f"- name: {hub_model_id}", readme)

            # Fix the model link in the markdown section - specifically target the model description line
            model_desc_pattern = (
                r"This model is a fine-tuned version of \[(.+?)\]\((.+?)\)"
            )
            if re.search(model_desc_pattern, readme):
                # Replace with the correct model name and HF link
                new_desc = f"This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model})"
                readme = re.sub(model_desc_pattern, new_desc, readme)

            with open(os.path.join(output_dir, "README.md"), "w") as file:
                file.write(readme)

        # Upload updated training parameters YAML to HF for traceability
        upload_to_hf(config_yaml)
        print("Model uploaded to HF; database upload skipped.")


if __name__ == "__main__":
    config = sys.argv[1]
    output_dir = sys.argv[2]
    base_model = sys.argv[3]
    upload_local(config, output_dir, base_model)
