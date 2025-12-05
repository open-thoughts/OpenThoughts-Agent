#!/usr/bin/env python3
"""
Python wrapper for running terminal-bench with configurable parameters.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_terminal_bench(
    llm_name: str,
    ray_endpoint: str,
    dataset_name: str,
    dataset_version: str,
    max_replicas: int,
    terminal_bench_dir: str = "/home/benjamin/terminal-bench",
    global_test_timeout_sec: int = 1000,
    cleanup: bool = True,
    agent: str = "terminus",
    temperature: float = 0.1,
    seed: int = 42,
    log_level_debug: bool = False,
    litellm_debug: bool = False,
    task_id: str = None
):
    """Run terminal-bench with the specified parameters."""
    
    print("Running terminal-bench...")
    
    # Change to terminal-bench directory
    terminal_bench_path = Path(terminal_bench_dir)
    if not terminal_bench_path.exists():
        print(f"Error: Terminal-bench directory not found: {terminal_bench_dir}")
        sys.exit(1)
    
    # Build the command
    cmd = [
        "uv", "run", "terminal-bench", "run",
        "--agent", agent,
        "--model", f"hosted_vllm/{llm_name}",
        "--agent-kwarg", f"api_base={ray_endpoint}/v1",
        "--agent-kwarg", f"temperature={temperature}",
        "--agent-kwarg", f"seed={seed}",
        "--dataset-name", dataset_name,
        "--dataset-version", dataset_version,
        "--global-test-timeout-sec", str(global_test_timeout_sec),
        "--n-concurrent", str(max_replicas)
    ]
    
    # Add task ID if specified
    if task_id:
        cmd.extend(["--task-id", task_id])
    
    # Add log level debug if enabled
    if log_level_debug:
        cmd.extend(["--log-level", "debug"])
        print("Debug log level enabled")
    
    # Add LiteLLM debug if enabled
    if litellm_debug:
        cmd.append("--litellm-debug")
        print("LiteLLM debug enabled")
    
    # Add cleanup flag if enabled
    if cleanup:
        cmd.append("--cleanup")
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {terminal_bench_dir}")
    
    try:
        # Set up environment for Podman Docker emulation
        import os
        env = os.environ.copy()
        
        # Enable LiteLLM debug logging
        env["LITELLM_LOG"] = "DEBUG"
        
        # Check if we're using Podman
        try:
            podman_check = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if "podman" in podman_check.stdout.lower() or "podman" in podman_check.stderr.lower():
                print("Detected Podman Docker emulation, setting DOCKER_HOST...")
                user_id = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
                env["DOCKER_HOST"] = f"unix:///run/user/{user_id}/podman/podman.sock"
        except:
            pass
        
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=terminal_bench_dir,
            env=env,
            check=True,
            text=True
        )
        
        print("Benchmark complete!")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"Error running terminal-bench: {e}")
        print(f"Return code: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: 'uv' command not found. Please ensure uv is installed and in PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run terminal-bench with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--llm-name", 
        type=str, 
        required=True,
        help="Name of the LLM model to use"
    )
    parser.add_argument(
        "--ray-endpoint", 
        type=str, 
        required=True,
        help="Ray endpoint URL (e.g., http://localhost:8080)"
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        required=True,
        help="Dataset name to run"
    )
    parser.add_argument(
        "--dataset-version", 
        type=str, 
        required=True,
        help="Dataset version to use"
    )
    parser.add_argument(
        "--max-replicas", 
        type=int, 
        required=True,
        help="Maximum number of concurrent replicas"
    )
    
    # Optional arguments
    parser.add_argument(
        "--terminal-bench-dir",
        type=str,
        default="/home/benjamin/terminal-bench",
        help="Path to terminal-bench directory"
    )
    parser.add_argument(
        "--global-test-timeout-sec",
        type=int,
        default=1000,
        help="Global test timeout in seconds"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="terminus",
        help="Agent type to use"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable cleanup after benchmark"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM sampling (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--log-level-debug",
        action="store_true",
        help="Enable debug log level"
    )
    parser.add_argument(
        "--litellm-debug",
        action="store_true",
        help="Enable LiteLLM debug logging"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Specific task ID to run (optional)"
    )
    
    args = parser.parse_args()
    
    # Display configuration
    print("Configuration:")
    print(f"  LLM Name: {args.llm_name}")
    print(f"  Ray Endpoint: {args.ray_endpoint}")
    print(f"  Dataset: {args.dataset_name} v{args.dataset_version}")
    print(f"  Max Replicas: {args.max_replicas}")
    print(f"  Agent: {args.agent}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Seed: {args.seed}")
    print(f"  Task ID: {args.task_id if args.task_id else 'All tasks'}")
    print(f"  Terminal-bench Directory: {args.terminal_bench_dir}")
    print(f"  Global Test Timeout: {args.global_test_timeout_sec}s")
    print(f"  Cleanup: {not args.no_cleanup}")
    print(f"  Log Level Debug: {args.log_level_debug}")
    print(f"  LiteLLM Debug: {args.litellm_debug}")
    print()
    
    # Run terminal-bench
    return run_terminal_bench(
        llm_name=args.llm_name,
        ray_endpoint=args.ray_endpoint,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        max_replicas=args.max_replicas,
        terminal_bench_dir=args.terminal_bench_dir,
        global_test_timeout_sec=args.global_test_timeout_sec,
        cleanup=not args.no_cleanup,
        agent=args.agent,
        temperature=args.temperature,
        seed=args.seed,
        log_level_debug=args.log_level_debug,
        litellm_debug=args.litellm_debug,
        task_id=args.task_id
    )


if __name__ == "__main__":
    sys.exit(main())