"""
Pre-download datasets and models for JSC clusters (no internet on compute nodes).

This script only implements and exports the `pre_download_dataset` function.

It is tested in `test_hpc.py`'s `test_pre_download` function.
"""

import os

# HuggingFace imports
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not available. Dataset/model pre-download will be skipped.")

# TODO(Charlie): check the logic here. Why it might not be available?
try:
    # Add parent directory to path to find scripts module
    import sys
    from pathlib import Path
    # Get the dc-agent root directory (3 levels up from this file)
    dc_agent_root = Path(__file__).resolve().parents[3]
    if str(dc_agent_root) not in sys.path:
        sys.path.insert(0, str(dc_agent_root))

    from scripts.harbor.tasks_parquet_converter import from_parquet
    PARQUET_CONVERTER_AVAILABLE = True
except ImportError as e:
    PARQUET_CONVERTER_AVAILABLE = False
    print(f"Warning: tasks_parquet_converter not available. Parquet extraction will be skipped. Error: {e}")


def pre_download_dataset(exp_args):
    """
    Pre-download datasets and models for JSC clusters (no internet on compute nodes)
    
    Returns:
    - updated_exp_args: Updated exp_args with downloaded paths
    - is_hf_available: Whether huggingface_hub is available
    """
    if not HF_AVAILABLE:
        print("Skipping pre-download: huggingface_hub not available")
        return exp_args, False
    
    print("Pre-downloading datasets and models for JSC cluster...")
    
    # Get cache directories
    hf_cache = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
    # Use cache_dir argument if provided, otherwise fall back to environment variable
    datasets_dir = os.environ.get("DATASETS_DIR", hf_cache)
    models_dir = os.environ.get("MODELS_DIR", hf_cache)
    
    # Pre-download datasets
    train_data = exp_args.get("train_data", [])
    val_data = exp_args.get("val_data", [])
    
    # Convert to lists for easier processing
    if isinstance(train_data, str):
        train_data = [train_data]
    if isinstance(val_data, str):
        val_data = [val_data]
    
    # Track downloaded dataset paths
    downloaded_datasets = {}
    
    all_datasets = set()
    if train_data:
        all_datasets.update(train_data)
    if val_data:
        all_datasets.update(val_data)
        
    for dataset in all_datasets:
        if dataset and not dataset.startswith("/"):  # Only download HF datasets, not local paths
            print(f"Pre-downloading dataset: {dataset}")
            try:
                # Set HF token for authentication
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token and hf_token != "hf_your_token_here":
                    print(f"  Using HF token for authentication")
                else:
                    print(f"  Warning: No valid HF token found, trying without authentication")
                
                # Use snapshot_download to get all dataset files without format issues
                try:
                    # Expand environment variables to get absolute path
                    expanded_datasets_dir = os.path.expandvars(datasets_dir)
                    # Ensure the directory exists
                    os.makedirs(expanded_datasets_dir, exist_ok=True)
                    dataset_download_path = snapshot_download(
                        repo_id=dataset,
                        repo_type="dataset",
                        cache_dir=expanded_datasets_dir,
                        token=hf_token if hf_token and hf_token != "hf_your_token_here" else None
                    )
                    print(f"✓ Dataset {dataset} downloaded to {dataset_download_path}")

                    # Check if this is a parquet dataset and extract it
                    if PARQUET_CONVERTER_AVAILABLE:
                        parquet_files = [f for f in os.listdir(dataset_download_path) if f.endswith('.parquet')]
                        if parquet_files:
                            print(f"Found parquet file(s): {parquet_files}")
                            parquet_file_path = os.path.join(dataset_download_path, parquet_files[0])

                            # Create extraction directory
                            dataset_name = dataset.replace("/", "_").replace("-", "_")
                            extraction_dir = os.path.join(expanded_datasets_dir, f"{dataset_name}_tasks")

                            print(f"Extracting parquet to tasks directory: {extraction_dir}")
                            try:
                                from_parquet(parquet_file_path, extraction_dir, on_exist="skip")
                                print(f"✓ Extracted parquet to {extraction_dir}")
                                # Update the dataset path to point to the extracted tasks directory
                                downloaded_datasets[dataset] = extraction_dir
                            except Exception as e:
                                print(f"✗ Failed to extract parquet: {e}")
                                # Fall back to the downloaded path
                                downloaded_datasets[dataset] = dataset_download_path
                        else:
                            # No parquet files, use the download path as-is
                            downloaded_datasets[dataset] = dataset_download_path
                    else:
                        # Store the downloaded path for this dataset
                        downloaded_datasets[dataset] = dataset_download_path
                except Exception as e:
                    print(f"✗ Failed to download dataset {dataset}: {e}")
                    continue
            except Exception as e:
                print(f"✗ Failed to download dataset {dataset}: {e}")
    
    # Update train_data and val_data with downloaded paths
    if downloaded_datasets:
        print("Updating dataset arguments with downloaded paths...")
        
        # Update train_data with downloaded paths
        if train_data:
            updated_train_data = []
            for dataset in train_data:
                if dataset in downloaded_datasets:
                    updated_train_data.append(downloaded_datasets[dataset])
                    print(f"  Updated train_data: {dataset} -> {downloaded_datasets[dataset]}")
                else:
                    updated_train_data.append(dataset)  # Keep original if not downloaded
            exp_args["train_data"] = updated_train_data
        
        # Update val_data with downloaded paths
        if val_data:
            updated_val_data = []
            for dataset in val_data:
                if dataset in downloaded_datasets:
                    updated_val_data.append(downloaded_datasets[dataset])
                    print(f"  Updated val_data: {dataset} -> {downloaded_datasets[dataset]}")
                else:
                    updated_val_data.append(dataset)  # Keep original if not downloaded
            exp_args["val_data"] = updated_val_data
    
    # Pre-download model
    model_path = exp_args.get("model_path")
    if model_path and not model_path.startswith("/"):  # Only download HF models, not local paths
        print(f"Pre-downloading model: {model_path}")
        try:
            hf_token = os.environ.get("HF_TOKEN")
            # Expand environment variables to get absolute path
            expanded_models_dir = os.path.expandvars(models_dir)
            model_download_path = snapshot_download(
                repo_id=model_path, 
                repo_type="model",
                cache_dir=expanded_models_dir,
                token=hf_token if hf_token and hf_token != "hf_your_token_here" else None
            )
            print(f"✓ Model {model_path} downloaded to {model_download_path}")
            # Update the model path to use the downloaded path
            exp_args["model_path"] = model_download_path
        except Exception as e:
            print(f"✗ Failed to download model {model_path}: {e}")
    
    print("Pre-download completed!")
    return exp_args, True
