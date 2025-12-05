#!/usr/bin/env python3
"""
Configurable parallel Daytona sandbox retry test script.
This script allows you to configure concurrency levels, task limits, and retry settings
via command-line arguments for optimal performance testing.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromImageParams,
    Image,
    Resources,
)
from daytona.common.errors import DaytonaError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

class ParallelDaytonaTester:
    def __init__(
        self, 
        dataset_path: str, 
        max_retries: int = 1, 
        max_concurrent: int = 10,
        max_tasks: int = None,
        task_filter: str = None
    ):
        self.dataset_path = Path(dataset_path)
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.max_tasks = max_tasks
        self.task_filter = task_filter
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.successful_tasks = []
        self.failed_tasks = []
        self.results = {
            'start_time': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'max_retries': max_retries,
            'max_concurrent': max_concurrent,
            'max_tasks': max_tasks,
            'task_filter': task_filter,
            'successful_tasks': [],
            'failed_tasks': [],
            'summary': {}
        }
        
    def get_task_directories(self) -> List[Path]:
        """Get task directories from the dataset with optional filtering."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        task_dirs = []
        for item in self.dataset_path.iterdir():
            if item.is_dir() and (item / "environment" / "Dockerfile").exists():
                # Apply task filter if specified
                if self.task_filter and self.task_filter.lower() not in item.name.lower():
                    continue
                task_dirs.append(item)
        
        # Sort and limit to max_tasks if specified
        task_dirs = sorted(task_dirs)
        if self.max_tasks:
            task_dirs = task_dirs[:self.max_tasks]
        
        return task_dirs
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((DaytonaError, Exception))
    )
    async def create_sandbox_for_task(self, task_dir: Path) -> AsyncSandbox:
        """Create a Daytona sandbox for a specific task with retry logic."""
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        
        resources = Resources(
            cpu=2,
            memory=4,
            disk=10,
            gpu=0,
        )
        
        params = CreateSandboxFromImageParams(
            image=Image.from_dockerfile(dockerfile_path),
            ephemeral=True,
            resources=resources,
        )
        
        daytona = AsyncDaytona()
        sandbox = await daytona.create(params=params)
        sandbox._daytona_instance = daytona
        
        return sandbox
    
    async def test_task(self, task_dir: Path) -> Dict[str, Any]:
        """Test a single task with retry logic and concurrency control."""
        async with self.semaphore:
            task_name = task_dir.name
            logger = logging.getLogger(__name__)
            logger.info(f"Testing task: {task_name}")
            
            result = {
                'task_name': task_name,
                'task_path': str(task_dir),
                'success': False,
                'attempts': 0,
                'error_messages': [],
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'duration_seconds': 0
            }
            
            start_time = datetime.now()
            sandbox = None
            daytona = None
            
            try:
                sandbox = await self.create_sandbox_for_task(task_dir)
                daytona = sandbox._daytona_instance
                
                result['success'] = True
                result['attempts'] = 1
                logger.info(f"✓ Successfully created sandbox for task: {task_name}")
                
            except Exception as e:
                error_msg = f"Failed to create sandbox for {task_name}: {str(e)}"
                logger.error(error_msg)
                result['error_messages'].append(error_msg)
                result['attempts'] = self.max_retries
                
            finally:
                if sandbox:
                    try:
                        await sandbox.stop()
                        await sandbox.delete()
                        logger.info(f"Cleaned up sandbox for task: {task_name}")
                    except Exception as cleanup_error:
                        logger.warning(f"Error during cleanup for {task_name}: {cleanup_error}")
                
                if daytona:
                    try:
                        await daytona.close()
                    except Exception as cleanup_error:
                        logger.warning(f"Error closing daytona connection for {task_name}: {cleanup_error}")
            
            end_time = datetime.now()
            result['end_time'] = end_time.isoformat()
            result['duration_seconds'] = (end_time - start_time).total_seconds()
            
            return result
    
    async def run_tests(self):
        """Run tests for all tasks with parallel execution."""
        logger = logging.getLogger(__name__)
        logger.info("Starting parallel Daytona sandbox retry tests")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Max retries per task: {self.max_retries}")
        logger.info(f"Max concurrent tasks: {self.max_concurrent}")
        if self.max_tasks:
            logger.info(f"Max tasks to test: {self.max_tasks}")
        if self.task_filter:
            logger.info(f"Task filter: {self.task_filter}")
        
        task_dirs = self.get_task_directories()
        total_tasks = len(task_dirs)
        
        if total_tasks == 0:
            logger.warning("No tasks found matching the criteria")
            return self.results
        
        logger.info(f"Testing {total_tasks} tasks in parallel...")
        
        # Create tasks for parallel execution
        tasks = [self.test_task(task_dir) for task_dir in task_dirs]
        
        # Track completed tasks
        completed = 0
        
        # Execute tasks in parallel with progress tracking
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            
            if result['success']:
                self.successful_tasks.append(result)
                self.results['successful_tasks'].append(result)
                logger.info(f"✓ [{completed}/{total_tasks}] Task {result['task_name']} succeeded")
            else:
                self.failed_tasks.append(result)
                self.results['failed_tasks'].append(result)
                logger.warning(f"✗ [{completed}/{total_tasks}] Task {result['task_name']} failed after {self.max_retries} attempts")
        
        # Update summary
        self.results['end_time'] = datetime.now().isoformat()
        self.results['summary'] = {
            'total_tasks': total_tasks,
            'successful_tasks': len(self.successful_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': len(self.successful_tasks) / total_tasks if total_tasks > 0 else 0,
            'total_duration_seconds': (
                datetime.fromisoformat(self.results['end_time']) - 
                datetime.fromisoformat(self.results['start_time'])
            ).total_seconds()
        }
        
        logger.info("=" * 60)
        logger.info("PARALLEL TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tasks: {total_tasks}")
        logger.info(f"Successful: {len(self.successful_tasks)}")
        logger.info(f"Failed: {len(self.failed_tasks)}")
        logger.info(f"Success rate: {self.results['summary']['success_rate']:.2%}")
        logger.info(f"Max concurrent: {self.max_concurrent}")
        logger.info(f"Total duration: {self.results['summary']['total_duration_seconds']:.1f} seconds")
        logger.info("=" * 60)
        
        # Save results to JSON file
        results_file = f"daytona_parallel_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Print failed tasks for easy reference
        if self.failed_tasks:
            logger.info("\nFAILED TASKS:")
            for task in self.failed_tasks:
                logger.info(f"  - {task['task_name']}: {task['error_messages'][-1] if task['error_messages'] else 'Unknown error'}")
        
        return self.results

def filter_failed_tasks(dataset_path: str, num_concurrent: int, max_retries: int = 5, max_tasks: int = None, task_filter: str = None) -> List[str]:
    """
    Main function that takes a dataset path and number of concurrent processes,
    and returns a list of paths that need to be filtered out (failed tasks).
    
    Args:
        dataset_path (str): Path to the dataset directory
        num_concurrent (int): Maximum number of concurrent tasks
        max_retries (int): Maximum number of retries per task (default: 5)
        max_tasks (int): Maximum number of tasks to test (default: all)
        task_filter (str): Filter tasks by name (case-insensitive substring match)
    
    Returns:
        List[str]: List of task paths that failed and need to be filtered out
    """
    # Setup logging
    logger = setup_logging("INFO")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return []
    
    tester = ParallelDaytonaTester(
        dataset_path=dataset_path,
        max_retries=max_retries,
        max_concurrent=num_concurrent,
        max_tasks=max_tasks,
        task_filter=task_filter
    )
    
    try:
        results = asyncio.run(tester.run_tests())
        
        # Extract failed task paths
        failed_paths = [task['task_path'] for task in results['failed_tasks']]
        
        logger.info(f"Found {len(failed_paths)} tasks that need to be filtered out")
        return failed_paths
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Parallel Daytona sandbox retry test script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all tasks with default settings (10 concurrent)
  python3 daytona_parallel_retry.py

  # Test first 20 tasks with 5 concurrent
  python3 daytona_parallel_retry.py --max-tasks 20 --concurrent 5

  # Test tasks containing 'hello' in name with 3 concurrent
  python3 daytona_parallel_retry.py --filter hello --concurrent 3

  # Test with 20 concurrent and 3 retries
  python3 daytona_parallel_retry.py --concurrent 20 --retries 3

  # Debug mode with custom log file
  python3 daytona_parallel_retry.py --log-level DEBUG --log-file debug.log
        """
    )
    
    parser.add_argument(
        '--dataset-path',
        default="/p/project/laionize/marianna/terminal_bench/data_cache_dir/datasets--mlfoundations-dev--sandboxes-tasks/snapshots/6fdf67053a80836ab1d5007104baded9f3513733",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=10,
        help="Maximum number of concurrent tasks (default: 10)"
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=5,
        help="Maximum number of retries per task (default: 5)"
    )
    parser.add_argument(
        '--max-tasks',
        type=int,
        help="Maximum number of tasks to test (default: all)"
    )
    parser.add_argument(
        '--filter',
        help="Filter tasks by name (case-insensitive substring match)"
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        '--log-file',
        help="Log file path (default: stdout only)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)
    
    tester = ParallelDaytonaTester(
        dataset_path=args.dataset_path,
        max_retries=args.retries,
        max_concurrent=args.concurrent,
        max_tasks=args.max_tasks,
        task_filter=args.filter
    )
    
    try:
        results = asyncio.run(tester.run_tests())
        
        if results['summary']['failed_tasks'] > 0:
            logger.warning(f"Some tasks failed. Check the log and results file for details.")
            sys.exit(1)
        else:
            logger.info("All tasks succeeded!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
