# JSC Pre-Download Feature

## Overview

The OT-Agent HPC system now includes automatic pre-download functionality for JSC clusters, which don't have internet access on compute nodes. This ensures that datasets and models are downloaded on the login node before jobs are submitted to compute nodes.

## How It Works

### 1. Automatic Detection
The system automatically detects JSC clusters and triggers pre-download:
```python
if "jsc" in exp_args.get("name", ""):
    exp_args = pre_download_dataset(exp_args)
```

### 2. Pre-Download Process
When launching a job on JSC clusters, the system:

1. **Detects JSC cluster** based on hostname pattern
2. **Pre-downloads datasets** from HuggingFace to the cache directory
3. **Pre-downloads models** from HuggingFace to the cache directory
4. **Updates model paths** to use the downloaded local paths
5. **Sets offline mode** for compute nodes

### 3. Cache Management
- Uses `HF_HUB_CACHE` environment variable for cache location
- Datasets and models are stored in the same cache directory
- Compute nodes run in offline mode (`HF_HUB_OFFLINE=1`)

## Implementation Details

### Files Modified

1. **`launch.py`**:
   - Added `pre_download_dataset()` function
   - Integrated pre-download into main workflow
   - Added HuggingFace imports with error handling

2. **`jsc.env`**:
   - Added `HF_DATASETS_CACHE` and `HF_HOME` variables
   - Ensured proper cache directory configuration

3. **`jsc_helper_training.sh`**:
   - Added offline mode environment variables
   - Updated default configuration to use pre-downloaded datasets
   - Set proper cache directory paths

4. **`README.md`**:
   - Added documentation for pre-download feature
   - Updated JSC-specific usage examples

### Key Functions

#### `pre_download_dataset(exp_args)`
```python
def pre_download_dataset(exp_args):
    """Pre-download datasets and models for JSC clusters (no internet on compute nodes)"""
    # Downloads all datasets from train_data and val_data
    # Downloads model from model_path
    # Updates exp_args with local paths
    # Returns updated exp_args
```

#### Environment Variables Set
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## Usage

### Automatic Usage
The pre-download happens automatically when using JSC clusters.

### Manual Testing
You can test the pre-download functionality in `test_hpc.py`

## Benefits

1. **No Internet Required on Compute Nodes**: JSC compute nodes can run in complete offline mode
2. **Automatic**: No manual intervention required
3. **Efficient**: Downloads happen once on login node, reused for multiple jobs
4. **Transparent**: Users don't need to change their workflow
5. **Robust**: Handles download failures gracefully

## Error Handling

- If HuggingFace libraries are not available, pre-download is skipped with a warning
- Download failures are logged but don't stop the job submission
- Local paths are preserved (not re-downloaded)
- Cache directory is created if it doesn't exist

## Cache Directory Structure

```
$HF_HUB_CACHE/
├── datasets/
│   └── mlfoundations-dev___sandboxes-tasks/
│       └── [dataset files]
└── models/
    └── Qwen___Qwen2.5-7B-Instruct/
        └── [model files]
```

## Testing

The system includes comprehensive tests in **`test_hpc.py`**'s `test_pre_download()`.

## Compatibility

- **JSC Clusters**: Full support with pre-download
- **TACC Clusters**: No pre-download (compute nodes have internet)
- **Other Clusters**: No pre-download (assumes internet access)

This feature ensures that the OT-Agent HPC system works seamlessly on JSC clusters while maintaining the same user experience across all supported clusters.
