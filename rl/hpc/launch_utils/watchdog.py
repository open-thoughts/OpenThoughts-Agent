"""
Watchdog functionality for monitoring job completion and uploading model to Hugging Face.

This script only exports the `login_watchdog` and `login_watchdog_chain` functions.
"""


import time
import os
import subprocess
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Add dc-agent root to path for database imports
# watchdog.py is at: dc-agent/rl/hpc/launch_utils/watchdog.py
# So we need to go up 3 levels to get to dc-agent root
DC_AGENT_ROOT = Path(__file__).resolve().parents[3]
if str(DC_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DC_AGENT_ROOT))

try:
    from huggingface_hub import HfApi, login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Watchdog functionality will be skipped.")

# Database imports
try:
    from database.unified_db.utils import register_trained_model, load_supabase_keys
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    print(f"Warning: Database utilities not available: {e}")


def _find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint directory based on global_step_* pattern"""
    import os
    
    print(f"Searching for checkpoints in: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"  Directory does not exist: {checkpoint_dir}")
        return None
    
    print(f"  Directory exists, listing contents...")
    try:
        items = os.listdir(checkpoint_dir)
        print(f"  Found {len(items)} items")
    except Exception as e:
        print(f"  Error listing directory: {e}")
        return None
    
    checkpoint_dirs = []
    for item in items:
        item_path = os.path.join(checkpoint_dir, item)
        print(f"  Checking: {item}")
        
        if item.startswith("global_step_"):
            if os.path.isdir(item_path):
                try:
                    step_num = int(item.split("_")[-1])
                    checkpoint_dirs.append((step_num, item_path))
                    print(f"    ✓ Valid checkpoint: step {step_num}")
                except ValueError:
                    print(f"    ✗ Invalid step number in: {item}")
                    continue
            else:
                print(f"    ✗ Not a directory: {item}")
    
    if not checkpoint_dirs:
        print(f"  No valid global_step_* directories found")
        return None
    
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoint_dirs[-1]
    print(f"  ✓ Latest checkpoint: {latest_path} (step {latest_step})")
    return latest_path


def _copy_final_checkpoint_policy(checkpoint_dir):
    """Copy policy files from the latest checkpoint to the parent directory"""
    import os
    import shutil
    
    print(f"\n{'='*50}")
    print(f"Starting checkpoint policy copy")
    print(f"{'='*50}")
    
    latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)
    if not latest_checkpoint:
        print(f"No global_step_* checkpoints found in {checkpoint_dir}")
        return False
    
    # Look for policy directory
    policy_base = os.path.join(latest_checkpoint, "policy")
    
    if not os.path.exists(policy_base):
        print(f"Policy directory not found: {policy_base}")
        # Try without policy subdirectory
        policy_base = latest_checkpoint
        print(f"Trying checkpoint root: {policy_base}")
    
    print(f"Found checkpoint directory: {latest_checkpoint}")
    print(f"Copying policy files from: {policy_base}")
    print(f"Copying policy files to: {checkpoint_dir}")
    
    try:
        # Copy all files from policy directory to parent directory
        copied_files = 0
        for item in os.listdir(policy_base):
            src_path = os.path.join(policy_base, item)
            dst_path = os.path.join(checkpoint_dir, item)
            
            # Skip FSDP checkpoint files
            if any(x in item for x in ['world_size', 'optim_', 'extra_state']) and item.endswith('.pt'):
                print(f"  ⊘ Skipping FSDP file: {item}")
                continue
            
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"  ✓ Copied file: {item}")
                copied_files += 1
            elif os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
                print(f"  ✓ Copied directory: {item}")
                copied_files += 1
        
        print(f"\n✓ Successfully copied {copied_files} items from latest checkpoint")
        print(f"{'='*50}\n")
        return True
        
    except Exception as e:
        print(f"Error copying policy files: {e}")
        import traceback
        traceback.print_exc()
        return False


def _upload_to_huggingface(model_path, repo_name, hf_token=None):
    """Upload model and checkpoints to Hugging Face Hub"""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not available. Cannot upload to Hugging Face.")
        return False
    
    try:
        # First, copy the final checkpoint policy files to the parent directory
        print(f"Processing checkpoints in {model_path}...")
        copy_success = _copy_final_checkpoint_policy(model_path)
        if not copy_success:
            print("Warning: Failed to copy final checkpoint policy files, proceeding with upload anyway")
        
        # Login to Hugging Face
        if hf_token:
            login(token=hf_token)
            print(f"Logged in to Hugging Face with provided token")
        else:
            print("Warning: No HF token provided, trying to use cached credentials")
        
        # Initialize HF API
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_name, exist_ok=True)
            print(f"Repository {repo_name} created or already exists")
        except Exception as e:
            print(f"Warning: Could not create repository {repo_name}: {e}")
        
        # Upload the model directory
        if os.path.exists(model_path):
            print(f"Uploading model from {model_path} to {repo_name}...")
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                commit_message=f"Upload model and checkpoints from training job"
            )
            print(f"✓ Successfully uploaded model to https://huggingface.co/{repo_name}")
            return True
        else:
            print(f"Error: Model path {model_path} does not exist")
            return False
            
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        return False


def _check_job_status(job_id):
    """
    Check if a SLURM job is still running or pending with a satisfiable dependency.
    
    Returns:
        True: Job is actively running or pending (and will actually run)
        False: Job has finished or will never run (DependencyNeverSatisfied)
    """
    try:
        # Get both status and reason
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%T %r"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and result.stdout.strip():
            output = result.stdout.strip()
            parts = output.split(maxsplit=1)
            status = parts[0] if parts else ""
            reason = parts[1] if len(parts) > 1 else ""
            
            # Job has finished
            if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                return False
            
            # Job is pending but will NEVER run - treat as finished
            if status == "PENDING" and "DependencyNeverSatisfied" in reason:
                return False
            
            # Job is actually running or will run
            return True
        else:
            # Job not in queue (finished and removed)
            return False
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False


def _register_rl_model_to_database(
    model_path: str,
    hf_repo_name: str,
    exp_args: dict,
    base_model: str = None,
    dataset_name: str = None,
    agent_name: str = None,
) -> dict:
    """
    Register a trained RL model to the Supabase database.
    
    Args:
        model_path: Path to the trained model
        hf_repo_name: HuggingFace repository name (e.g., "DCAgent2/model-name")
        exp_args: Experiment arguments from launch.py
        base_model: Base model name (default: extract from exp_args)
        dataset_name: Dataset name (default: extract from exp_args)
        agent_name: Agent name (default: extract from exp_args)
        
    Returns:
        dict: Registration result with 'success' key
    """
    if not DB_AVAILABLE:
        print("Database registration skipped: database utilities not available")
        return {"success": False, "error": "Database utilities not available"}
    
    # Load Supabase credentials
    if not load_supabase_keys():
        print("Database registration skipped: Supabase credentials not loaded")
        return {"success": False, "error": "Supabase credentials not loaded"}
    
    try:
        from datetime import datetime, timezone, timedelta
        
        # Extract information from exp_args
        job_name = exp_args.get("job_name", "unknown")
        
        # Get agent name
        if not agent_name:
            agent_name = exp_args.get("agent_name", "terminus")
        
        # Get base model from model_path argument
        base_model = exp_args.get("original_model_name") or exp_args.get("model_path", "")
        
        # Get dataset name from train_data (it's a list due to nargs="+")
        dataset_name = exp_args.get("original_train_data")[0] if exp_args.get("original_train_data") else ""
        
        # Training type is RL
        training_type = "RL"
        
        # Get timestamps from checkpoint metadata
        training_start = None
        training_end = None
        
        if os.path.exists(model_path):
            # Use modification time of the export directory for end time
            stat_info = os.stat(model_path)
            training_end = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
            
            # Try to find the earliest checkpoint for start time
            try:
                checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith("global_step_")]
                if checkpoint_dirs:
                    # Sort by step number
                    checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
                    earliest_ckpt = checkpoint_dirs[0]
                    ckpt_path = os.path.join(model_path, earliest_ckpt)
                    if os.path.exists(ckpt_path):
                        ckpt_stat = os.stat(ckpt_path)
                        training_start = datetime.fromtimestamp(ckpt_stat.st_mtime, tz=timezone.utc)
            except Exception as e:
                print(f"Warning: Could not determine training start from checkpoints: {e}")
        
        if not training_start or not training_end:
            # Fallback: use current time and assume 1 hour training
            training_end = datetime.now(timezone.utc)
            training_start = training_end - timedelta(hours=1)
        
        # Build W&B link if all required fields are present
        wandb_link = None
        run_name = exp_args.get("run_name", "")
        project_name = exp_args.get("project_name", "")
        wandb_entity = exp_args.get("wandb_entity", "")
        
        if run_name and wandb_entity and project_name:
            wandb_link = f"https://wandb.ai/{wandb_entity}/{project_name}/runs/{run_name}"
        
        # Build training parameters
        training_parameters = {
            "config_blob": f"https://huggingface.co/{hf_repo_name}/blob/main/config.json",
            "hf_repo": hf_repo_name,
            "job_name": job_name,
            "algorithm": "PPO",  # Default for RL
            "num_nodes": exp_args.get("num_nodes"),
            "model_path": model_path,
        }
        
        # Build the registration record
        record = {
            "agent_name": agent_name,
            "training_start": training_start.isoformat(),
            "training_end": training_end.isoformat() if training_end else None,
            "created_by": hf_repo_name.split("/", 1)[0] if "/" in hf_repo_name else "unknown",
            "base_model_name": base_model,
            "dataset_name": dataset_name,
            "training_type": training_type,
            "training_parameters": training_parameters,
            "wandb_link": wandb_link,
            "traces_location_s3": exp_args.get("traces_s3_path"),
            "model_name": hf_repo_name,
        }
        
        # Register to database - let register_trained_model handle validation
        print(f"\n{'='*50}")
        print("Registering model to database...")
        print(f"  Model: {hf_repo_name}")
        print(f"  Agent: {agent_name}")
        print(f"  Base: {base_model}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        result = register_trained_model(record, forced_update=False)
        
        if result.get("success"):
            model = result.get("model", {})
            status = "updated" if result.get("updated") else "created"
            print(f"✓ Database registration {status}: {model.get('id')} ({model.get('name')})")
        else:
            print(f"✗ Database registration failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Failed to register model to database: {e}"
        print(error_msg)
        traceback.print_exc()
        return {"success": False, "error": error_msg}


def _submit_upload_job(model_path, repo_name, hf_token, exp_args):
    """
    Submit a SLURM job to upload model to HuggingFace from a compute node.
    
    This avoids CPU memory limits on login nodes by offloading the upload
    to a compute node with more resources.
    
    Args:
        model_path: Path to the model directory to upload
        repo_name: HuggingFace repository name
        hf_token: HuggingFace authentication token
        exp_args: Dictionary of experiment arguments for template rendering
        
    Returns:
        str: Job ID if successful, None otherwise
    """
    model_path = os.path.expandvars(model_path)
    
    # Prepare template variables
    template_vars = {
        "job_name": exp_args.get("job_name", "model"),
        "logs_dir": exp_args.get("logs_dir", "/tmp"),
        "account": exp_args.get("account", ""),
        "conda_activate_path": os.environ.get("CONDA_ACTIVATE_PATH", ""),
        "conda_env_path": os.environ.get("CONDA_ENV_PATH", ""),
        "secret_env_path": os.environ.get("SECRET_ENV_PATH", ""),
        "uv_cache_dir": os.environ.get("UV_CACHE_DIR", ""),
        "dc_agent_path": os.environ.get("DC_AGENT", exp_args.get("dc_agent_path")),
        "model_path": model_path,
        "repo_name": repo_name,
        "hf_token": hf_token or os.environ.get("HF_TOKEN", ""),
    }
    
    # Get template path
    upload_template_path = exp_args.get("upload_sbatch_jinja_path")
    if not upload_template_path:
        # Try to infer from train template path
        train_template_path = exp_args.get("train_sbatch_jinja_path", "")
        if "tacc_train.j2" in train_template_path:
            upload_template_path = train_template_path.replace("tacc_train.j2", "tacc_upload.j2")
        else:
            print("Error: Could not determine upload template path")
            return None
    
    if not os.path.exists(upload_template_path):
        print(f"Error: Upload template {upload_template_path} does not exist")
        return None
    
    # Render template
    output_dir = os.path.join(exp_args.get("experiments_dir", "/tmp"), "sbatch_scripts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{template_vars['job_name']}_upload.sbatch")
    
    env = Environment(loader=FileSystemLoader(os.path.dirname(upload_template_path)))
    template = env.get_template(os.path.basename(upload_template_path))
    rendered = template.render(**template_vars)
    
    with open(output_path, "w") as f:
        f.write(rendered)
    
    print(f"Generated upload sbatch script: {output_path}")
    
    # Submit job
    try:
        result = subprocess.run(
            ["sbatch", output_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"✓ Upload job submitted: {job_id}")
            print(f"  Monitor with: squeue -j {job_id}")
            print(f"  Check logs: {template_vars['logs_dir']}/upload_{template_vars['job_name']}_{job_id}.out")
            return job_id
        else:
            print(f"✗ Failed to submit upload job: {result.stderr}")
            return None
    except Exception as e:
        print(f"✗ Error submitting upload job: {e}")
        return None


def _finalize_and_upload_model(model_path, repo_name, hf_token, completion_message, exp_args, wait_seconds=120):
    """
    Helper function to finalize training and upload model to HuggingFace.
    
    Now submits an upload job to a compute node instead of uploading directly
    from the login node to avoid CPU memory limits.

    Args:
        model_path: Path to the model directory
        repo_name: HuggingFace repository name
        hf_token: HuggingFace authentication token
        completion_message: Message to print before upload
        exp_args: Dictionary of experiment arguments for template rendering
        wait_seconds: Seconds to wait for file operations (default: 120)

    Returns:
        bool: True if upload job submitted successfully, False otherwise
    """
    print(completion_message)
    print(f"Waiting {wait_seconds} seconds for final file operations to complete...")
    time.sleep(wait_seconds)

    # Submit upload job instead of uploading directly
    upload_job_id = _submit_upload_job(model_path, repo_name, hf_token, exp_args)
    
    if upload_job_id:
        print("✓ Upload job submitted successfully!")
        print(f"✓ Model will be uploaded to: https://huggingface.co/{repo_name}")
        
        # Register to database after successful upload submission
        try:
            db_result = _register_rl_model_to_database(
                model_path=model_path,
                hf_repo_name=repo_name,
                exp_args=exp_args
            )
        except Exception as e:
            # Don't fail the upload if DB registration fails
            print(f"Warning: Database registration failed but upload succeeded: {e}")
        
        return True
    else:
        print("✗ Failed to submit upload job")
        return False

# ----------------------------
# Functions to be exported
# ----------------------------

def login_watchdog(job_id, model_path, repo_name, hf_token=None, check_interval=300, exp_args=None):
    """
    Watchdog function that monitors job completion and uploads model to Hugging Face.
    
    Now submits upload as a separate SLURM job to avoid login node memory limits.
    
    Args:
        job_id: SLURM job ID to monitor
        model_path: Path to the model directory to upload
        repo_name: Hugging Face repository name
        hf_token: Hugging Face token for authentication
        check_interval: Time in seconds between job status checks (default: 5 minutes)
        exp_args: Dictionary of experiment arguments for template rendering
    """
    print(f"Starting login watchdog for job {job_id}")
    print(f"Will check job status every {check_interval} seconds")
    print(f"Model will be uploaded to: https://huggingface.co/{repo_name}")
    
    # Ensure exp_args is provided
    if exp_args is None:
        print("Error: exp_args is required for upload job submission")
        return
    
    while True:
        if not _check_job_status(job_id):
            print(f"Job {job_id} has finished. Submitting upload job...")
            
            # Wait a bit for any final file writes to complete
            print("Waiting 300 seconds for final file operations to complete...")
            time.sleep(30)
            
            # Submit upload job to compute node
            upload_job_id = _submit_upload_job(model_path, repo_name, hf_token, exp_args)
            
            if upload_job_id:
                print("✓ Login watchdog completed - upload job submitted!")
            else:
                print("✗ Login watchdog failed to submit upload job")
            
            break
        else:
            print(f"Job {job_id} is still running. Next check in {check_interval} seconds...")
            time.sleep(check_interval)
    
    print("Login watchdog exiting.")


def login_watchdog_chain(job_ids, model_path, repo_name, hf_token=None, check_interval=300, exp_args=None):
    """
    Watchdog function that monitors a chain of jobs and uploads model when training completes.
    
    Now submits upload as a separate SLURM job to avoid login node memory limits.

    This is designed to work with max_restarts where multiple jobs are submitted with
    afternotok dependencies. It intelligently tracks which job in the chain is currently
    running and only uploads when the actual final job completes.

    Args:
        job_ids: List of SLURM job IDs in dependency chain order [job1, job2, ..., jobN]
        model_path: Path to the model directory to upload
        repo_name: Hugging Face repository name
        hf_token: Hugging Face token for authentication
        check_interval: Time in seconds between job status checks (default: 5 minutes)
        exp_args: Dictionary of experiment arguments for template rendering
    """
    print(f"Starting login watchdog chain for {len(job_ids)} jobs: {job_ids}")
    print(f"Will check job status every {check_interval} seconds")
    print(f"Model will be uploaded to: https://huggingface.co/{repo_name}")
    
    # Ensure exp_args is provided
    if exp_args is None:
        print("Error: exp_args is required for upload job submission")
        return

    current_job_idx = 0

    while current_job_idx < len(job_ids):
        current_job_id = job_ids[current_job_idx]
        print(f"\nMonitoring job {current_job_id} ({current_job_idx + 1}/{len(job_ids)})...")

        while True:
            status = _check_job_status(current_job_id)

            if status is False:
                # Current job has finished (either completed successfully or failed/timed out)
                print(f"Job {current_job_id} has finished.")

                # Check if there's a next job in the chain
                if current_job_idx < len(job_ids) - 1:
                    next_job_id = job_ids[current_job_idx + 1]
                    print(f"Checking if next job {next_job_id} was triggered...")
                    time.sleep(30)  # Brief wait for SLURM to update job status

                    next_status = _check_job_status(next_job_id)

                    if next_status:
                        # Next job is running (current job timed out or failed)
                        print(f"Next job {next_job_id} is running - training continues with restart")
                        current_job_idx += 1
                        break  # Break inner loop to monitor next job
                    else:
                        # Next job not running (current job completed successfully)
                        _finalize_and_upload_model(
                            model_path,
                            repo_name,
                            hf_token,
                            "No restart triggered - training completed successfully!",
                            exp_args,
                            wait_seconds=120
                        )
                        return
                else:
                    # This was the last job in the chain
                    _finalize_and_upload_model(
                        model_path,
                        repo_name,
                        hf_token,
                        "Final job in chain completed - training finished!",
                        exp_args,
                        wait_seconds=120
                    )
                    return
            else:
                # Job still running
                print(f"Job {current_job_id} is still running. Next check in {check_interval} seconds...")
                time.sleep(check_interval)

    print("Login watchdog chain exiting.")


# TODO(Charlie): does it actually test anything?
def test_watchdog_functionality():
    """Test function to verify watchdog functionality without running actual jobs"""
    print("Testing watchdog functionality...")
    
    # Test job status checking with a non-existent job
    test_job_id = "99999999"
    status = _check_job_status(test_job_id)
    print(f"Job status check for non-existent job {test_job_id}: {status}")
    
    # Test HF upload functionality (dry run)
    if HF_AVAILABLE:
        print("Hugging Face Hub is available - upload functionality should work")
    else:
        print("Hugging Face Hub not available - upload functionality will be skipped")
    
    print("Watchdog functionality test completed.")


if __name__ == "__main__":
    test_watchdog_functionality()
