import os
import sys
import argparse
from huggingface_hub import snapshot_download

def is_valid_task_dir(path):
    """
    Check if a directory is a valid task directory by verifying it has instruction.md
    and excludes Git-related files.
    """
    # Skip Git-related files and directories
    basename = os.path.basename(path)
    if basename.startswith('.git') or basename in {'.gitignore', '.gitattributes', '.github'}:
        return False
    
    # Check if it's a directory and has instruction.md
    return (os.path.isdir(path) and 
            os.path.isfile(os.path.join(path, 'instruction.md')))

def get_dataset_path(repo_id):
    """
    Get the path to the dataset without downloading
    Returns the path if it exists, None otherwise
    
    Args:
        repo_id (str): The repository ID to look for
    """
    
    # Construct the path using HF_HUB_CACHE environment variable
    hf_cache = os.getenv('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
    dataset_path = os.path.join(hf_cache, f"datasets--{repo_id.replace('/', '--')}")
    
    # Find the latest snapshot
    snapshots_dir = os.path.join(dataset_path, 'snapshots')
    if os.path.exists(snapshots_dir):
        try:
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        except OSError as e:
            print(f"Could not read snapshots directory {snapshots_dir}: {e}", file=sys.stderr)
            snapshots = []
        if snapshots:
            latest_snapshot = sorted(snapshots)[-1]  # Get the latest snapshot
            snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
            
            # Verify we have valid task directories
            if os.path.exists(snapshot_path):
                task_dirs = [d for d in os.listdir(snapshot_path) 
                            if is_valid_task_dir(os.path.join(snapshot_path, d))]
                
                if task_dirs:
                    print(f"Found dataset at {snapshot_path}")
                    print(f"Found {len(task_dirs)} valid task directories")
                    return snapshot_path
                else:
                    print("No valid task directories found in snapshot")
                    return None
                
    print("Dataset not found, downloading...")
    return None

def download_sandboxes_dataset(repo_id, local_dir=None, cache_dir=None):
    """
    Download the dataset using snapshot_download
    
    Args:
        repo_id (str): The repository ID (e.g., 'mlfoundations-dev/sandboxes-tasks')
        local_dir (str, optional): Local directory to save the dataset
        cache_dir (str, optional): Cache directory for huggingface hub
    """
    
    try:
        print(f"Starting download of {repo_id}...")
        
        # Download the entire dataset repository
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            cache_dir=cache_dir,
            # You can add these optional parameters:
            # revision="main",  # specific branch/tag
            # allow_patterns=["*.json", "*.parquet"],  # only download specific file types
            # ignore_patterns=["*.md"],  # ignore specific file types
            # token=None,  # if you need authentication
        )
        
        # Remove .gitattributes file if it exists
        gitattributes_path = os.path.join(local_path, '.gitattributes')
        if os.path.exists(gitattributes_path):
            os.remove(gitattributes_path)
            print("Removed .gitattributes file")
        
        # Verify we have valid task directories
        task_dirs = [d for d in os.listdir(local_path) 
                    if is_valid_task_dir(os.path.join(local_path, d))]
        
        if task_dirs:
            print(f"DATASET_PATH={local_path}")
            print(f"Found {len(task_dirs)} valid task directories")
            return local_path
        else:
            print("No valid task directories found in downloaded dataset")
            return None
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download or locate a Hugging Face dataset')
    parser.add_argument('repo_id', help='Repository ID (e.g., mlfoundations-dev/clean-sandboxes-tasks)')
    parser.add_argument('--local-dir', help='Local directory to save the dataset')
    parser.add_argument('--cache-dir', help='Cache directory for huggingface hub')
    
    args = parser.parse_args()
    
    # First try to get existing path
    path = get_dataset_path(args.repo_id)
    if not path:
        # If not found, download it
        print("Dataset not found, downloading...", file=sys.stderr)
        path = download_sandboxes_dataset(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            cache_dir=args.cache_dir
        )

    if path:
        print(f"DATASET_PATH={path}")
        return 0
    else:
        print("Failed to download dataset", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

