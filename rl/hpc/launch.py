"""
The script to launch a SkyRL training job on an HPC cluster.

Differences between directly running a SkyRL script, say: 
```
python -m skyrl_train.entrypoints.main_base \
  trainer.epochs=10 \
  trainer.train_batch_size=64 \
  ...
```
are:
- Generate an `.sbatch` script that is specific to the cluster (JSC vs. TACC)
- In that .sbatch, we will:
  - Set up environment variables specific to the cluster's compute nodes
  - Launch a ray server (since most of the case we are doing multi-node)
  - Run the SkyRL command
- Watchdog for job status monitoring and Hugging Face uploading at the end
- Retry logic for job submission
- Pre-download datasets and models in case the compute nodes have no internet access (for JSC clusters)
"""

import os
import re
import subprocess
import time
import threading
from collections import defaultdict
from .launch_utils import login_watchdog, login_watchdog_chain
from .launch_utils import pre_download_dataset

try:
    from arguments import parse_args
    from hpc import detect_hpc, set_environment
except ImportError:
    from hpc.arguments import parse_args
    from hpc.hpc import detect_hpc, set_environment


# Import parquet converter
import sys
# Get to ot-agent root (two levels up from hpc/launch.py)
ot_agent_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ot_agent_root)

from jinja2 import Environment, FileSystemLoader

def launch_sbatch(sbatch_script_path, dependency=None, has_internet=False) -> str:
    if not has_internet:
        sbatch_cmd = f"bash {sbatch_script_path} &"
    elif dependency is not None:
        sbatch_cmd = f"sbatch --dependency={dependency} {sbatch_script_path}"
    else:
        sbatch_cmd = f"sbatch {sbatch_script_path}"

    job_id = subprocess.check_output(sbatch_cmd, shell=True).decode("utf-8").strip()
    print(f"Job {job_id} submitted with dependency {dependency}.")
    return job_id


def render_jinja2_template(template_path, template_vars, output_path):
    """
    Render a Jinja2 template with the provided variables.
    
    Args:
        template_path: Path to the Jinja2 template file
        template_vars: Dictionary of variables to pass to the template
        output_path: Path where the rendered template should be saved
    """
    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_path)))
    
    # Load the template
    template = env.get_template(os.path.basename(template_path))
    
    # Render the template with the provided variables
    rendered_content = template.render(**template_vars)
    
    # Write the rendered content to the output file
    with open(output_path, "w") as f:
        f.write(rendered_content)
        
    print(f"Rendered Jinja2 template to {output_path}")


def construct_sbatch_script(exp_args):
    """
    Construct sbatch script using Jinja2 templating for TACC clusters.

    TODO(Charlie): Check JSC support for Jinja2 templating.
    """
    jinja_template_path = exp_args["train_sbatch_jinja_path"]
    assert os.path.exists(jinja_template_path), f"Jinja2 template {jinja_template_path} does not exist."

    # Prepare template variables
    template_vars = defaultdict(str, **exp_args)
    
    # Add TACC-specific template variables
    tacc_vars = {
        "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT", ""),
        "vllm_config_root": os.environ.get("VLLM_CONFIG_ROOT", ""),
        "triton_dump_dir": os.environ.get("TRITON_DUMP_DIR", ""),
        "triton_override_dir": os.environ.get("TRITON_OVERRIDE_DIR", ""),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", ""),
        "flashinfer_workspace_base": os.environ.get("FLASHINFER_WORKSPACE_BASE", ""),
        "uv_cache_dir": os.environ.get("UV_CACHE_DIR", ""),
        "secret_env_path": os.environ.get("SECRET_ENV_PATH", ""),
        "conda_activate_path": os.environ.get("CONDA_ACTIVATE_PATH", ""),
        "extra_pythonpath": os.environ.get("EXTRA_PYTHONPATH", ""),  # TODO(Charlie): handle when it is not provided. This is optional.
        "conda_env_path": os.environ.get("CONDA_ENV_PATH", ""),
        "skyrl_home": os.environ.get("SKYRL_HOME", ""),
        "ot_agent_path": os.environ.get("OT_AGENT", "/scratch/08134/negin/dc-agent-shared/dc-agent"),
    }
    # assert that all the directories exists to prevent erroring out on compute node after queuing
    for key, value in tacc_vars.items():
        assert os.path.exists(value), f"{key} path {value} does not exist. Please check your `tacc.env` file."

    template_vars.update(tacc_vars)
    
    # Set default time limit if not provided
    if template_vars.get("time_limit") is None:
        template_vars["time_limit"] = "01:00:00"
    
    # Create output directory
    sbatch_dir = os.path.join(template_vars["experiments_dir"], "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_script_path = os.path.join(sbatch_dir, f"{template_vars['job_name']}.sbatch")
    
    # Render the Jinja2 template
    render_jinja2_template(jinja_template_path, template_vars, sbatch_script_path)

    return sbatch_script_path


def build_skyrl_command_string(entrypoint, skyrl_args):
    """
    Convert the -S key=value provided in the HPC launch script into a string. e.g.:
    ```
    python -m skyrl_train.entrypoints.main_base \
      trainer.epochs=10 \
      trainer.train_batch_size=64 \
      data.train_data="['/path/a', '/path/b']" \
    ...
    ```
    In the Jinja, 
    """

    args = []
    for i, (key, value) in enumerate(skyrl_args.items()):
        # Ensure list values are passed as a single hydra-compatible string argument.
        # Example: data.train_data="['/path/a', '/path/b']"
        if isinstance(value, (list, tuple)):
            quoted_items = [f"'{str(item)}'" for item in value]
            hydra_list = "[" + ", ".join(quoted_items) + "]"
            current_arg = f'{key}="{hydra_list}"'
        else:
            current_arg = f"{key}={value}"
        # Last arg does not have a trailing \
        if i == len(skyrl_args) - 1:
            args.append(f"  {current_arg} \n")
        else:
            args.append(f"  {current_arg} \\\n")

    return f"python -m {entrypoint} \\\n" + "".join(args)


def inplace_update_skyrl_args(skyrl_args, exp_args):
    """
    While `skyrl_args` contains the arguments passed in with -S key=value, some arguments can only
    be passed in after we processed the `exp_args`. For instance, we can only know the `train_data`
    and `val_data` paths after we pre-download the datasets.
    """
    # TODO(Charlie): are there more to add?
    exp_args_to_skyrl_keys = {
        "train_data": "data.train_data",
        "val_data": "data.val_data",
        "model_path": "trainer.policy.model.path",
    }

    for exp_key, skyrl_key in exp_args_to_skyrl_keys.items():
        assert exp_key in exp_args, f"Experiment argument {exp_key} not found in exp_args"
        if skyrl_key in skyrl_args:
            raise ValueError(
                f"SkyRL configuration {skyrl_key} already exists in skyrl_args. You should not "
                f"pass it in with -S {skyrl_key}={skyrl_args[skyrl_key]}\n"
                f"Instead, pass it in with the experiment argument --{exp_key}"
            )
        skyrl_args[skyrl_key] = exp_args[exp_key]

    return skyrl_args


def submit_job(exp_args=None, dependency=None, has_internet=False):
    exp_args["logs_dir"] = os.path.join(exp_args["experiments_dir"], "logs")
    os.makedirs(exp_args["logs_dir"], exist_ok=True)

    job_ids = []  # Track all job IDs in the restart chain

    if exp_args.get("max_restarts") is not None:
        max_restarts = int(exp_args["max_restarts"])
        if max_restarts > 0:
            for _ in range(max_restarts):
                job_id = launch_sbatch(
                    exp_args["train_sbatch_path_out"], dependency=dependency, has_internet=has_internet
                )
                job_id = job_id.split()[-1]
                job_ids.append(job_id)
                dependency = f"afternotok:{job_id}"

    # Submit final job in chain
    job_id = launch_sbatch(
        exp_args["train_sbatch_path_out"], dependency, has_internet
    )
    job_id = job_id.split()[-1]
    job_ids.append(job_id)

    print(f"Submitted {len(job_ids)} job(s) in restart chain")
    print(f"Writing logs to directory: {exp_args['logs_dir']}")
    # print(f"Writing logs to {exp_args['logs_dir']}/{exp_args['job_name']}_<job_id>.out")
    return job_ids  # Return list of all job IDs


def update_exp_args(exp_args, args):
    for key, value in args.items():
        if key in exp_args and value is None:
            del exp_args[key]
            print(f"Removed {key} from experiment arguments")
        elif key in exp_args and value != exp_args[key]:
            print(f"Overwrote {key} from {exp_args[key]} to {value}")
        exp_args[key] = value
    return exp_args


def pre_validation(exp_args, cli_args):

    def _extract_template_keys(file_path):
        # Curly braces but not those within ${...}
        curly_brace_pattern = r"(?<!\$)\{([^{}]*)\}"
        with open(file_path, "r") as f:
            file = f.read()
        return re.findall(curly_brace_pattern, file)

    # Fill in sbatch template
    if "train_sbatch_jinja_path" in exp_args and os.path.exists(
        exp_args["train_sbatch_jinja_path"]
    ):
        template_keys = _extract_template_keys(exp_args["train_sbatch_jinja_path"])
        missing_keys = []
        for key in template_keys:
            if (
                key not in exp_args
                and key not in cli_args
            ):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Warning: Template keys {missing_keys} not found in experiment arguments or cli arguments.")
            print("These will be replaced with empty strings in the sbatch script.")
    elif "train_sbatch_jinja_path" in exp_args:
        raise FileNotFoundError(
            f"Train sbatch file {exp_args['train_sbatch_jinja_path']} does not exist."
        )


def main():
    """
    Main function that launches training jobs with optional Hugging Face upload watchdog.
    
    New Features:
    - Login watchdog: Monitors job completion and automatically uploads model to Hugging Face
    - To enable: Set --enable_hf_upload flag
    - Optional parameters:
      --hf_repo_name: Hugging Face repository name (default: {project_name}-{final_model_name})
      --hf_token: Hugging Face token (or set HF_TOKEN environment variable)
      --watchdog_check_interval: Check interval in seconds (default: 300)
    """
    print()
    # Parse command line arguments
    cli_args = parse_args()
    for key, value in cli_args.items():
        if type(value) == str:
            value = value.lower()
            if value == "false":
                cli_args[key] = False
            elif value == "true":
                cli_args[key] = True
            elif value == "none":
                cli_args[key] = None
            elif value.isdigit():
                # Convert numeric strings to integers
                cli_args[key] = int(value)
            else:
                # Try to convert to float (handles scientific notation like 1e-6)
                try:
                    cli_args[key] = float(value)
                except ValueError:
                    # If conversion fails, keep as string
                    pass

    # Storing all the arguments in a dictionary that we add to in order of precedence
    exp_args = dict()

    # Add arguments to experiment from automatically detecting HPC
    hpc = detect_hpc()
    set_environment(hpc)

    # Add arguments and validate
    print()
    exp_args = update_exp_args(exp_args, hpc.model_dump())
    exp_args = update_exp_args(exp_args, cli_args)

    # Job name -- TODO(Charlie) CHECK THIS, make it a required field for LaunchArgs
    print(f"Job name: {exp_args['job_name']}")

    # Pre-validation
    pre_validation(exp_args, cli_args)

    # Clean dataset names by removing square brackets
    exp_args["original_model_name"] = exp_args.get("model_path")
    exp_args["original_train_data"] = exp_args.get("train_data", []).copy()
    
    if "train_data" in exp_args:
        exp_args["train_data"] = [d.strip("[]") for d in exp_args["train_data"]]
    if "val_data" in exp_args:
        exp_args["val_data"] = [d.strip("[]") for d in exp_args["val_data"]]
    exp_args, is_hf_available = pre_download_dataset(exp_args)

    # Construct the SkyRL command string (e.g. "python -m skyrl_train.entrypoints.main_base ... ")
    print()
    skyrl_args = exp_args["skyrl_args"]  # those passed in with -S key=value
    del exp_args["skyrl_args"]
    skyrl_args = inplace_update_skyrl_args(skyrl_args, exp_args)
    exp_args["skyrl_command_string"] = build_skyrl_command_string(exp_args["skyrl_entrypoint"], skyrl_args)
    print(f"Running SkyRL command in sbatch: {exp_args['skyrl_command_string']}")

    # Construct the sbatch script
    print()
    train_sbatch_path_out = construct_sbatch_script(exp_args)
    exp_args = update_exp_args(
        exp_args, {"train_sbatch_path_out": train_sbatch_path_out}
    )

    # Display train arguments
    print("=" * 20 + f" Train Args " + "=" * 20)
    for key, value in exp_args.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Set upload sbatch template path if not already set
    if "train_sbatch_jinja_path" in exp_args:
        train_template = exp_args["train_sbatch_jinja_path"]
        if "tacc_train.j2" in train_template:
            exp_args["upload_sbatch_jinja_path"] = train_template.replace("tacc_train.j2", "tacc_upload.j2")
            print(f"\nAuto-configured upload template: {exp_args['upload_sbatch_jinja_path']}")

    # Dry run with no job submission
    if exp_args.get("dry_run", False):
        print(
            "DRY RUN: Job would be submitted with the above parameters, but --dry_run flag was set."
        )
        return

    train_job_ids = submit_job(exp_args=exp_args, dependency=None, has_internet=hpc.internet_node)

    # For backwards compatibility and logging, use the last job ID
    train_job_id = train_job_ids[-1] if train_job_ids else None

    # Start login watchdog if HF upload is enabled
    if exp_args.get("enable_hf_upload", False) and is_hf_available:
        # Get HF repository name from arguments or construct from job name
        hf_repo_name = exp_args.get("hf_repo_name")
        if not hf_repo_name:
            # Use DCAgent/ prefix with final_model_name
            final_model_name = exp_args.get('final_model_name', exp_args.get('job_name', 'model'))
            hf_repo_name = f"DCAgent/{final_model_name}"
        
        # Get HF token from environment or arguments
        hf_token = exp_args.get("hf_token") or os.environ.get("HF_TOKEN")

        # Get model path for upload (export_path contains the final model)
        model_upload_path = skyrl_args.get("trainer.export_path")

        if model_upload_path and os.path.exists(os.path.dirname(model_upload_path)):
            check_interval = exp_args.get("watchdog_check_interval", 300)  # Default 5 minutes

            # Determine target function and args based on job count
            if len(train_job_ids) > 1:
                # Multiple jobs in restart chain - use smart chain watchdog
                target_func = login_watchdog_chain
                target_args = (train_job_ids, model_upload_path, hf_repo_name, hf_token, check_interval, exp_args)
                print(f"\nStarting login watchdog chain for {len(train_job_ids)} jobs")
                print(f"Job chain: {train_job_ids}")
            else:
                # Single job - use original watchdog
                target_func = login_watchdog
                target_args = (train_job_id, model_upload_path, hf_repo_name, hf_token, check_interval, exp_args)
                print(f"\nStarting login watchdog for job {train_job_id}")

            # Common logging and thread setup
            print(f"Model will be uploaded to: https://huggingface.co/{hf_repo_name}")
            print(f"Model path: {model_upload_path}")

            watchdog_thread = threading.Thread(
                target=target_func,
                args=target_args,
                daemon=True
            )
            watchdog_thread.start()
            print("Login watchdog thread started for job(s)")

            print("Login watchdog started. The script will continue running to monitor the job.")
            print("Press Ctrl+C to stop monitoring (the job will continue running).")

            try:
                # Keep the main thread alive to monitor the watchdog
                while watchdog_thread.is_alive():
                    time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\nStopping login watchdog monitoring. The job will continue running.")
                print("The watchdog thread will continue in the background.")
        else:
            print(f"Warning: Model path {model_upload_path} does not exist. Skipping watchdog setup.")
    elif exp_args.get("enable_hf_upload", False) and not is_hf_available:
        print("Warning: HF upload enabled but huggingface_hub not available. Skipping watchdog setup.")


if __name__ == "__main__":
    main()
