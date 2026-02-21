# download HF model
import os
import sys
from huggingface_hub import snapshot_download
import argparse

def get_model_path(repo_id):
    """
    Get the path to the model without downloading
    Returns the path if it exists, None otherwise
    
    Args:
        repo_id (str): The repository ID to look for
    """
    
    # Construct the path using HF_HUB_CACHE environment variable
    hf_cache = os.getenv('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
    model_path = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    
    # Find the latest snapshot
    snapshots_dir = os.path.join(model_path, 'snapshots')
    if os.path.exists(snapshots_dir):
        try:
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        except OSError as e:
            print(f"Could not read snapshots directory {snapshots_dir}: {e}", file=sys.stderr)
            snapshots = []
        if snapshots:
            latest_snapshot = sorted(snapshots)[-1]  # Get the latest snapshot
            snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
            
            if os.path.exists(snapshot_path):
                print(f"Found model at {snapshot_path}")
                return snapshot_path
                
    print("Model not found, downloading...")
    return None


def download_model(repo_id, local_dir=None, cache_dir=None):
    """
    Download the model using snapshot_download
    
    Args:
        repo_id (str): The repository ID (e.g., 'mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1')
        local_dir (str, optional): Local directory to save the model
        cache_dir (str, optional): Cache directory for huggingface hub
    """
    
    try:
        print(f"Starting download of {repo_id}...")
        
        # Download the entire model repository
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
        )
        
        print(f"Model downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face model using snapshot_download")
    parser.add_argument("repo_id", type=str, help="The repository ID of the model to download (e.g., 'mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1')")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory to save the model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for huggingface hub")
    
    args = parser.parse_args()
    
    # First try to get the model path without downloading
    model_path = get_model_path(args.repo_id)
    
    if model_path is None:
        # If not found, download the model
        model_path = download_model(args.repo_id, local_dir=args.local_dir, cache_dir=args.cache_dir)
    
    if model_path:
        print(f"MODEL_PATH={model_path}")
    else:
        print("Failed to obtain the model path.")