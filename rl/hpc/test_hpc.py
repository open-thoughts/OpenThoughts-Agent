#!/usr/bin/env python3
"""
Test script for the OT-Agent HPC system. Include four tests:
- test_hpc_detection
- test_environment_setup
- test_pre_download
- test_dry_run
"""

import os
import sys
import tempfile
import subprocess
import shutil

# Add the hpc directory to the path
hpc_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, hpc_dir)

# Import HPC modules
from hpc import detect_hpc, set_environment
from launch_utils import pre_download_dataset

def test_hpc_detection():
    """Test HPC cluster detection"""
    print("Testing HPC cluster detection...")
    try:
        hpc = detect_hpc()
        print(f"✓ Detected HPC: {hpc.name}")
        print(f"  Hostname: {hpc.hostname}")
        print(f"  Account: {hpc.account}")
        print(f"  Partition: {hpc.partition}")
        print(f"  GPUs per node: {hpc.gpus_per_node}")
        return True
    except Exception as e:
        print(f"✗ HPC detection failed: {e}")
        return False

def test_environment_setup():
    """Test environment variable setup"""
    print("\nTesting environment setup...")
    try:
        hpc = detect_hpc()
        set_environment(hpc)
        print("✓ Environment variables loaded")
        
        # Check some key variables
        key_vars = ['DC_AGENT', 'DC_AGENT_TRAIN', 'SKYRL_HOME']
        for var in key_vars:
            if var in os.environ:
                print(f"  {var}: {os.environ[var]}")
            else:
                print(f"  Warning: {var} not set")
        return True
    except Exception as e:
        print(f"✗ Environment setup failed: {e}")
        return False

def test_dry_run():
    """Test dry run functionality"""
    print("\nTesting dry run...")
    try:
        # Test with minimal arguments - run from parent directory
        # TODO(Charlie): make the test more rigorous if needed.
        cmd = [
            sys.executable, "-m", "hpc.launch",
            "--job_name", "test_job",
            "--time_limit", "01:00:00",
            "--num_nodes", "1", "--final_model_name", "dummy_name",
            "--train_data", "mlfoundations-dev/sandboxes-tasks-hello-world",
            "--val_data", "mlfoundations-dev/sandboxes-tasks-hello-world",
            "--model_path", "Qwen/Qwen2.5-0.5B-Instruct",
            "--skyrl_entrypoint", "skyrl_train.entrypoints.main_base",
            "--dry_run"
        ]

        # Run from the parent directory (dc-agent/train)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=parent_dir)
        
        if result.returncode == 0:
            print("✓ Dry run successful")
            print("  Output preview:")
            print("  " + "\n  ".join(result.stdout.split('\n')[:10]))
            return True
        else:
            print(f"✗ Dry run failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Dry run test failed: {e}")
        return False

def test_pre_download():
    """Test the pre-download functionality for JSC"""
    
    print("Testing JSC pre-download functionality...")
    
    # Set up test environment
    test_cache_dir = tempfile.mkdtemp()
    os.environ['HF_HUB_CACHE'] = test_cache_dir
    os.environ['DATASETS_DIR'] = test_cache_dir
    os.environ['MODELS_DIR'] = test_cache_dir

    # Test arguments
    test_args = {
        "train_data": ["mlfoundations-dev/sandboxes-tasks"],
        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",  # Small model for testing
        "name": "jsc_test"
    }

    print(f"Test cache directory: {test_cache_dir}")
    print(f"Test datasets: {test_args['train_data']}")
    print(f"Test model: {test_args['model_path']}")
    
    try:
        # Run pre-download
        result_args, is_hf_available = pre_download_dataset(test_args)
        if not is_hf_available:
            print("✗ Pre-download failed: huggingface_hub not available")
            return False

        print("✓ Pre-download completed successfully")
        print(f"Updated model path: {result_args.get('model_path', 'Not updated')}")
        
        # Check if files were downloaded
        cache_contents = os.listdir(test_cache_dir)
        print(f"Cache directory contents: {cache_contents}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pre-download failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(test_cache_dir, ignore_errors=True)

def main():
    """Run all tests"""
    print("=== OT-Agent HPC System Test ===\n")
    
    tests = [
        test_hpc_detection,
        test_environment_setup,
        test_pre_download,
        test_dry_run
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
