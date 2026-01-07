#!/usr/bin/env python3
"""
Common utilities for dataset operations.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import yaml
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from scripts.harbor.tasks_parquet_converter import (
    convert_logs_to_parquet,
    convert_to_parquet,
)
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from database.unified_db.utils import register_hf_dataset
from data.gcs_cache import gcs_cache
from data.generation import (
    GenerationContext,
    GenerationError,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    InferenceEngine,
    InferenceFailure,
    InputValidationError,
    TraceFailure,
    UploadFailure,
    BaseDataGenerator,
    add_generation_args,
    create_engine_from_args,
    create_inference_engine,
    OpenAIEngine,
    AnthropicEngine,
    GenericOpenAIEngine,
)
from tqdm.asyncio import tqdm_asyncio
import pyarrow as pa
from data.gcs_cache import gcs_cache
from harbor.models.trial.config import TrialConfig, TaskConfig, EnvironmentConfig    
import asyncio
from harbor.models.environment_type import EnvironmentType
from harbor.trial.trial import Trial
import traceback
import json
from rapidfuzz import fuzz
from transformers import AutoTokenizer
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources

CONSOLE = Console()
PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TextColumn("{task.completed}/{task.total}"),
    TextColumn("[green]✓ {task.fields[success]}"),
    TextColumn("[red]✗ {task.fields[failed]}"),
    TextColumn("Rem: {task.fields[remaining]}"),
    TextColumn("{task.fields[rate]:.2f} task/s"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


def finalize_dataset_output(
    source_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]],
) -> Path:
    """
    Move or copy a temporary dataset directory into the requested output directory.

    Args:
        source_dir: Path to the directory produced during generation.
        output_dir: Final destination directory provided by the caller.

    Returns:
        Path: Directory containing the final dataset.

    Raises:
        RuntimeError: If the destination directory exists and is not empty.
    """
    source_path = Path(source_dir)
    if output_dir is None:
        return source_path

    target_path = Path(output_dir)
    source_path = source_path.resolve()
    target_path = target_path.resolve()

    if source_path == target_path:
        return target_path

    target_path.mkdir(parents=True, exist_ok=True)

    existing_entries = list(target_path.iterdir())
    if existing_entries:
        raise RuntimeError(
            f"Output directory '{target_path}' is not empty. "
            "Please remove existing contents or provide a different --output-dir."
        )

    for item in source_path.iterdir():
        shutil.move(str(item), str(target_path / item.name))

    shutil.rmtree(source_path, ignore_errors=True)
    return target_path



@gcs_cache()
def create_task_from_dockerfiles_and_questions(
    dockerfiles: List[str],
    questions: List[str],
) -> str:
    """
    Create tasks from dockerfiles and questions.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="dockerfile_tasks_"))
    print(f"Generating tasks with dockerfiles in: {temp_dir}")

    for i, (dockerfile, question) in tqdm(enumerate(zip(dockerfiles, questions))):
        create_task_directory_unified(
            output_dir=temp_dir,
            task_id=i,
            instruction_content=question if isinstance(question, str) else question[0],
            dataset_prefix="dockerfile",
            metadata=question[1] if isinstance(question, tuple) and len(question) > 1 else None,
            solution_content=question[2] if isinstance(question, tuple) and len(question) > 2 else None,
            test_sh_content=question[3] if isinstance(question, tuple) and len(question) > 3 else None,
            test_py_content=question[4] if isinstance(question, tuple) and len(question) > 4 else None
        )
        
        # Write the dockerfile to the environment directory
        task_dir = temp_dir / f"dockerfile-{i:04d}"
        env_dir = task_dir / "environment"
        dockerfile_path = env_dir / "Dockerfile"
        # Remove existing dockerfile first
        if dockerfile_path.exists():
            dockerfile_path.unlink()
        dockerfile_path.write_text(dockerfile)
        
    print(f"Generated {len(dockerfiles)} tasks with dockerfiles successfully!")
    return str(temp_dir)

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def upload_tasks_to_hf(
    dataset_path: str, 
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: str = "Upload dataset",
    large_folders_mode: bool = False,
    marker: str = "instruction.md"
) -> str:
    """
    Upload a dataset to Hugging Face Hub.
    
    Args:
        dataset_path (str): Path to the local dataset directory containing tasks
        repo_id (str): Hugging Face repository ID (e.g., 'username/dataset-name')
        private (bool): Whether to create a private repository (default: False)
        token (str, optional): Hugging Face token for authentication
        commit_message (str): Commit message for the upload
    
    Returns:
        str: URL of the uploaded dataset repository
        
    Raises:
        ValueError: If dataset_path is invalid or repository ID is malformed
        FileNotFoundError: If dataset_path doesn't exist
        Exception: If upload fails
    """
    logger = setup_logging()    
    
    if not dataset_path or not isinstance(dataset_path, str):
        raise ValueError("dataset_path must be a non-empty string")
    
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string")
    
    # Validate dataset path
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path must be a directory: {dataset_path}")
    
    # Validate repository ID format
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise ValueError("repo_id must be in format 'username/dataset-name'")
    
    logger.info(f"Uploading dataset to {repo_id}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Private repository: {private}")
    
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        token=token,
        exist_ok=True
    )
    logger.info(f"Repository {repo_id} created or already exists")
    
    # Upload the entire dataset directory
    logger.info("Starting dataset upload...")
    parquet_dir = convert_to_parquet(dataset_path)
    # Use upload_folder to upload the entire directory
    result = api.upload_folder(
        folder_path=str(parquet_dir),
        repo_id=repo_id,
        delete_patterns="*",
        repo_type="dataset"
    )
    
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    
    logger.info(f"Successfully uploaded dataset to: {repo_url}")
    logger.info(f"Upload result: {result}")
    
    return repo_url


def upload_logs_to_hf(
    dataset_path: str, 
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: str = "Upload logs",
    large_folders_mode: bool = False
) -> str:
    """
    Upload a dataset to Hugging Face Hub.
    
    Args:
        dataset_path (str): Path to the local dataset directory containing tasks
        repo_id (str): Hugging Face repository ID (e.g., 'username/dataset-name')
        private (bool): Whether to create a private repository (default: False)
        token (str, optional): Hugging Face token for authentication
        commit_message (str): Commit message for the upload
    
    Returns:
        str: URL of the uploaded dataset repository
        
    Raises:
        ValueError: If dataset_path is invalid or repository ID is malformed
        FileNotFoundError: If dataset_path doesn't exist
        Exception: If upload fails
    """
    logger = setup_logging()    
    
    if not dataset_path or not isinstance(dataset_path, str):
        raise ValueError("dataset_path must be a non-empty string")
    
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string")
    
    # Validate dataset path
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path must be a directory: {dataset_path}")
    
    # Validate repository ID format
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise ValueError("repo_id must be in format 'username/dataset-name'")
    
    logger.info(f"Uploading dataset to {repo_id}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Private repository: {private}")
    
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        token=token,
        exist_ok=True
    )
    logger.info(f"Repository {repo_id} created or already exists")
    
    # Upload the entire dataset directory
    logger.info("Starting dataset upload...")
    parquet_dir = convert_logs_to_parquet(dataset_path)
    # Use upload_folder to upload the entire directory
    result = api.upload_folder(
        folder_path=str(parquet_dir),
        repo_id=repo_id,
        delete_patterns="*",
        repo_type="dataset"
    )
    
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    
    logger.info(f"Successfully uploaded dataset to: {repo_url}")
    logger.info(f"Upload result: {result}")
    
    return repo_url

def download_hf_dataset(
    repo_id: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    use_snapshot: bool = True,
    local_files_only: bool = False
) -> str:
    """
    Download a Hugging Face dataset and return the path to the datasets.
    
    Args:
        repo_id (str): Hugging Face dataset repository ID (e.g., 'mlfoundations-dev/sandboxes-tasks')
        cache_dir (str, optional): Directory to cache the dataset. If None, uses default HF cache.
        revision (str, optional): Specific revision/commit hash to download. If None, downloads latest.
        use_snapshot (bool): If True, uses snapshot_download for full repo. If False, uses load_dataset.
        local_files_only (bool): If True, only use local files (no internet download).
    
    Returns:
        str: Path to the downloaded dataset directory
        
    Raises:
        ValueError: If repo_id is invalid or dataset cannot be downloaded
        FileNotFoundError: If local_files_only=True and dataset not found locally
    """
    logger = setup_logging()
    
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string")

    cache_path = os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
    logger.info(f"Downloading dataset: {repo_id}")
    logger.info(f"Cache directory: {cache_path}")
    logger.info(f"Revision: {revision or 'latest'}")
    logger.info(f"Use snapshot: {use_snapshot}")
    
    if use_snapshot:
        # Use snapshot_download for full repository download
        dataset_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_path),
            revision=revision,
            local_files_only=local_files_only,
            repo_type="dataset"
        )
        logger.info(f"Dataset downloaded via snapshot to: {dataset_path}")
        
    else:
        # Use load_dataset for dataset-specific download
        # Load the dataset to trigger download
        dataset = load_dataset(
            repo_id,
            cache_dir=str(cache_path),
            revision=revision,
            download_mode="reuse_dataset_if_exists" if local_files_only else "reuse_cache_if_exists"
        )
        
        # Find the actual dataset directory in the cache
        # The dataset is typically stored in a subdirectory of the cache
        dataset_name = repo_id.replace("/", "___")
        potential_paths = [
            cache_path / "downloads" / "extracted" / dataset_name,
            cache_path / "downloads" / "extracted" / f"{dataset_name}-{revision}" if revision else None,
            cache_path / "datasets" / dataset_name,
        ]
        
        dataset_path = None
        for path in potential_paths:
            if path and path.exists():
                dataset_path = str(path)
                break
        
        if dataset_path is None:
            # Fallback: look for any directory containing the dataset name
            for root, dirs, files in os.walk(cache_path):
                if dataset_name in root or any(dataset_name in d for d in dirs):
                    dataset_path = root
                    break
        
        if dataset_path is None:
            raise FileNotFoundError(f"Could not locate downloaded dataset directory for {repo_id}")
        
        logger.info(f"Dataset downloaded via load_dataset to: {dataset_path}")
    
    # Verify the dataset path exists and contains data
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Downloaded dataset path does not exist: {dataset_path}")
    
    logger.info(f"Successfully downloaded dataset {repo_id} to: {dataset_path}")
    return str(dataset_path)



def create_standard_task_toml() -> str:

    return """
    version = "1.0"

    [agent]
    timeout_sec = 900.0

    [metadata]
    author_name = "Sandboxes"
    author_email = "sandboxes@sandboxes.com"
    difficulty = "medium"
    category = "sandbox"
    tags = [
        "sandbox",
    ]

    [verifier]
    restart_environment = false
    timeout_sec = 720.0
"""


def create_standard_dockerfile() -> str:
    """
    Create a standard Python-based Dockerfile for all datasets.

    Returns:
        str: Dockerfile content
    """
    return """FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
"""


def create_question_from_data(
    data_item: Dict[str, Any],
    question_template: str,
    **template_kwargs
) -> str:
    """
    Create a standardized question/instruction from data using a template.

    Args:
        data_item: Dictionary containing data fields
        question_template: Template string with placeholders for data fields
        **template_kwargs: Additional template variables

    Returns:
        str: Formatted question/instruction text
    """
    # Merge data_item with template_kwargs, giving priority to template_kwargs
    template_data = {**data_item, **template_kwargs}

    try:
        return question_template.format(**template_data)
    except KeyError as e:
        raise ValueError(f"Template missing required field: {e}")


def create_task_directory_unified(
    output_dir: Path,
    task_id: int,
    instruction_content: str,
    dataset_prefix: str,
    metadata: Optional[Union[str, Dict[str, Any]]] = None,
    solution_content: Optional[str] = None,
    test_sh_content: Optional[str] = None,
    test_py_content: Optional[str] = None,
    task_toml_content: Optional[str] = None,
    dockerfile_content: Optional[str] = None,
) -> Path:
    """
    Create a standardized task directory structure.

    Args:
        output_dir: Parent directory for tasks
        task_id: Unique task identifier
        instruction_content: Content for instruction.md file
        dataset_prefix: Prefix for task directory name (e.g., 'gsm8k', 'stackexchange')

    Returns:
        Path: Path to created task directory
    """
    task_dir = output_dir / f"{dataset_prefix}-{task_id:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)

    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(instruction_content, encoding="utf-8")

    dockerfile_path = env_dir / "Dockerfile"
    dockerfile_path.write_text(
        (dockerfile_content or create_standard_dockerfile()),
        encoding="utf-8",
    )

    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(
        (task_toml_content or create_standard_task_toml()),
        encoding="utf-8",
    )


    if metadata is not None:
        metadata_path = task_dir / "metadata.json"
        if isinstance(metadata, (dict, list)):
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            metadata_path.write_text(str(metadata), encoding="utf-8")

    if solution_content is not None:
        solution_dir = task_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        solve_path = solution_dir / "solve.sh"
        solve_path.write_text(solution_content, encoding="utf-8")
        os.chmod(solve_path, 0o755)

    if test_sh_content is not None or test_py_content is not None:
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        if test_sh_content is not None:
            test_sh_path = tests_dir / "test.sh"
            test_sh_path.write_text(test_sh_content, encoding="utf-8")
            os.chmod(test_sh_path, 0o755)

        if test_py_content is not None:
            test_py_path = tests_dir / "test_state.py"
            test_py_path.write_text(test_py_content, encoding="utf-8")

    return task_dir

def upload_traces_to_hf(
    dataset: Dataset,
    repo_id: str,
    dataset_type: str,
) -> None:
    """
    Upload traces to Hugging Face Hub.
    """
    dataset = clean_empty_structs(dataset)
    dataset.push_to_hub(repo_id)
    _sync_hf_dataset_card_metadata(dataset, repo_id)
    register_hf_dataset(
        repo_name=repo_id,
        dataset_type=dataset_type
    )
   

@gcs_cache()
def generate_tasks_from_questions(
    questions: List[str],
    dataset_prefix: str = "task",
) -> str:
    """Generate task directories from dataset components"""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_tasks_"))
    print(f"Generating tasks in: {temp_dir}")

    for i, dataset in tqdm(enumerate(questions)):
        create_task_directory_unified(
            output_dir=temp_dir,
            task_id=i,
            instruction_content=dataset if isinstance(dataset, str) else dataset[0],
            dataset_prefix=dataset_prefix,
            metadata=dataset[1] if isinstance(dataset, tuple) and len(dataset) > 1 else None,
            solution_content=dataset[2] if isinstance(dataset, tuple) and len(dataset) > 2 else None,
            test_sh_content=dataset[3] if isinstance(dataset, tuple) and len(dataset) > 3 else None,
            test_py_content=dataset[4] if isinstance(dataset, tuple) and len(dataset) > 4 else None
        )

    print(f"Generated {len(questions)} tasks successfully!")
    return str(temp_dir)

@gcs_cache()
def select_top_n_by_score(
    items: List[Any],
    scores: List[float],
    n: Optional[int] = None,
    top_percentage: Optional[float] = None,
    descending: bool = True,
    return_scores: bool = False
) -> Union[List[Any], tuple[List[Any], List[float]]]:
    """
    Select top N items based on scores.
    
    Args:
        items: List of items to select from (e.g., questions, conversations)
        scores: List of scores corresponding to each item
        n: Number of items to select (if None, uses top_percentage)
        top_percentage: Percentage of items to select (0.0 to 1.0)
        descending: If True, higher scores are better; if False, lower scores are better
        return_scores: If True, also returns the scores of selected items
    
    Returns:
        Selected items, or (selected_items, selected_scores) if return_scores=True
    
    Raises:
        ValueError: If neither n nor top_percentage is specified, or if lengths don't match
    """
    if len(items) != len(scores):
        raise ValueError(f"Items and scores must have same length: {len(items)} != {len(scores)}")
    
    if n is None and top_percentage is None:
        raise ValueError("Must specify either n or top_percentage")
    
    if n is None:
        n = int(len(items) * top_percentage)
    
    n = min(n, len(items))  # Don't select more than available
    
    # Create (item, score) pairs and sort by score
    item_score_pairs = list(zip(items, scores))
    item_score_pairs.sort(key=lambda x: x[1], reverse=descending)
    
    # Select top n
    selected_pairs = item_score_pairs[:n]
    selected_items = [item for item, _ in selected_pairs]
    
    if return_scores:
        selected_scores = [score for _, score in selected_pairs]
        return selected_items, selected_scores
    
    # Print statistics
    if n > 0:
        selected_scores = [s for _, s in selected_pairs]
        remaining_scores = [s for _, s in item_score_pairs[n:]] if n < len(item_score_pairs) else []
        
        print(f"Selected {n}/{len(items)} items")
        print(f"  Selected scores: mean={np.mean(selected_scores):.2f}, "
              f"min={min(selected_scores):.2f}, max={max(selected_scores):.2f}")
        if remaining_scores:
            print(f"  Remaining scores: mean={np.mean(remaining_scores):.2f}, "
                  f"min={min(remaining_scores):.2f}, max={max(remaining_scores):.2f}")
    
    return selected_items

@gcs_cache()
def subsample_tasks_directory(
    source_dir: str,
    num_samples: int,
    dataset_prefix: str = "subsampled"
) -> str:
    """
    Subsample tasks from a directory and create a new directory with randomly sampled tasks.

    Args:
        source_dir (str): Path to the source directory containing tasks
        num_samples (int): Number of tasks to randomly sample without replacement
        dataset_prefix (str): Prefix for the output directory name

    Returns:
        str: Path to the new directory containing subsampled tasks

    Raises:
        ValueError: If num_samples is greater than available tasks
        FileNotFoundError: If source directory doesn't exist
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_path.is_dir():
        raise ValueError(f"Source path must be a directory: {source_dir}")

    # Get all task directories (assuming they follow the pattern of being directories)
    task_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if len(task_dirs) == 0:
        raise ValueError(f"No task directories found in: {source_dir}")


    # Create temporary directory for subsampled tasks
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_"))
    print(f"Creating subsampled tasks in: {temp_dir}")

    # Randomly sample tasks without replacement
    if len(task_dirs) > num_samples:
        sampled_tasks = random.sample(task_dirs, num_samples)
    else:
        sampled_tasks = task_dirs

    # Copy sampled tasks to new directory
    for i, task_dir in enumerate(sampled_tasks):
        dest_path = temp_dir / task_dir.name
        shutil.copytree(task_dir, dest_path)

        if (i + 1) % 100 == 0:
            print(f"Copied {i + 1} tasks...")

    print(f"Successfully subsampled {num_samples} tasks from {len(task_dirs)} total tasks!")
    return str(temp_dir)

def upsample_tasks_directory(
    source_dir: str,
    num_samples: int,
    dataset_prefix: str = "upsampled"
) -> str:
    """
    Upsample tasks from a directory by repeating them to reach num_samples.
    If the number of existing tasks is already >= num_samples, returns the original path.

    Args:
        source_dir (str): Path to the source directory containing tasks
        num_samples (int): Target number of tasks to reach through upsampling
        dataset_prefix (str): Prefix for the output directory name

    Returns:
        str: Path to the new directory containing upsampled tasks (or original if no upsampling needed)

    Raises:
        FileNotFoundError: If source directory doesn't exist
        ValueError: If source directory is empty
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_path.is_dir():
        raise ValueError(f"Source path must be a directory: {source_dir}")

    # Get all task directories
    task_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if len(task_dirs) == 0:
        raise ValueError(f"No task directories found in: {source_dir}")

    # If we already have enough tasks, return the original directory
    if len(task_dirs) >= num_samples:
        print(f"Already have {len(task_dirs)} tasks (>= {num_samples}), no upsampling needed.")
        return str(source_path)

    # Create temporary directory for upsampled tasks
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_"))
    print(f"Creating upsampled tasks in: {temp_dir}")

    # Repeat tasks until we reach num_samples
    tasks_copied = 0
    while tasks_copied < num_samples:
        # Cycle through all tasks
        for task_dir in task_dirs:
            if tasks_copied >= num_samples:
                break

            dest_path = temp_dir / task_dir.name if tasks_copied < len(task_dirs) else temp_dir / f"{task_dir.name}_copy{tasks_copied // len(task_dirs)}"
            shutil.copytree(task_dir, dest_path)
            tasks_copied += 1

            if tasks_copied % 100 == 0:
                print(f"Copied {tasks_copied} tasks...")

    print(f"Successfully upsampled from {len(task_dirs)} to {tasks_copied} tasks!")
    return str(temp_dir)


def clean_empty_structs(dataset: Dataset) -> Dataset:
    """
    Automatically detect and remove empty struct columns from a HuggingFace dataset.
    Empty structs cause issues when writing to Parquet format.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Cleaned dataset with empty struct columns removed
    """
    def has_empty_struct(field_type) -> bool:
        """Recursively check if a type contains any empty structs."""
        if pa.types.is_struct(field_type):
            if field_type.num_fields == 0:
                return True
            # Check nested fields
            for i in range(field_type.num_fields):
                if has_empty_struct(field_type.field(i).type):
                    return True
        elif pa.types.is_list(field_type):
            return has_empty_struct(field_type.value_type)
        elif pa.types.is_map(field_type):
            return has_empty_struct(field_type.item_type)
        return False

    columns_to_remove = []

    # Check the actual data table schema (this is the one that matters for parquet)
    table = dataset.data
    for field in table.schema:
        if has_empty_struct(field.type):
            columns_to_remove.append(field.name)
            print(f"Found column with empty struct: '{field.name}' - will remove")

    # Remove the problematic columns
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
        print(f"Removed {len(columns_to_remove)} column(s) containing empty structs: {columns_to_remove}")
    else:
        print("No empty struct columns found")

    return dataset

def _split_readme_front_matter(content: str) -> tuple[str, str]:
    """Return (front_matter, body) from a README that may use YAML front matter."""
    if not content:
        return "", ""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
    return "", content


def _sync_hf_dataset_card_metadata(dataset: Dataset, repo_id: str) -> None:
    """Ensure README dataset_info sizes match actual uploaded artifact sizes."""
    api = HfApi()
    try:
        repo_entries = list(
            api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        )
    except Exception as exc:
        print(f"[trace-export] Warning: could not inspect {repo_id}: {exc}")
        return

    data_bytes = sum(
        getattr(entry, "size", 0) or 0
        for entry in repo_entries
        if getattr(entry, "path", "").startswith("data/")
    )
    if data_bytes <= 0:
        return

    try:
        readme_path = hf_hub_download(
            repo_id, "README.md", repo_type="dataset"
        )
        raw_readme = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        raw_readme = ""

    front_matter, body = _split_readme_front_matter(raw_readme)
    try:
        parsed = yaml.safe_load(front_matter) if front_matter.strip() else {}
    except Exception:
        parsed = {}
    if parsed is None:
        parsed = {}

    dataset_info = parsed.get("dataset_info") or {}
    splits = dataset_info.get("splits") or [{"name": "train"}]
    for split in splits:
        split_name = split.get("name", "train")
        if split_name == "train":
            split["num_examples"] = len(dataset)
            split["num_bytes"] = data_bytes
    dataset_info["splits"] = splits
    dataset_info["download_size"] = data_bytes
    dataset_info["dataset_size"] = data_bytes
    parsed["dataset_info"] = dataset_info
    if "configs" not in parsed or not parsed["configs"]:
        parsed["configs"] = [
            {
                "config_name": "default",
                "data_files": [{"split": "train", "path": "data/train-*"}],
            }
        ]

    new_front = yaml.safe_dump(parsed, sort_keys=False).strip()
    updated = f"---\n{new_front}\n---\n"
    if body:
        updated += body.lstrip("\n")

    try:
        api.upload_file(
            path_or_fileobj=updated.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    except Exception as exc:
        print(f"[trace-export] Warning: failed to update README for {repo_id}: {exc}")

async def can_daytona_start(task_path: str) -> bool:
    """
    Check if a task can be started by Daytona.
    """
    trial_config = TrialConfig(
        task=TaskConfig(path=task_path),
        environment=EnvironmentConfig(type=EnvironmentType.DAYTONA, force_build=True, delete=True),
    )
    trial = Trial(trial_config)
    try:
        await trial._start_environment_with_retry()
        await trial._environment.stop()
        if trial._environment._sandbox is not None:
            await trial._environment._sandbox.delete()
        return True
    except Exception as e:
        print(e)
        traceback.print_exc()
        try:
            if trial._environment._sandbox is not None:
                await trial._environment._sandbox.delete()
        except:
            pass
        return False

        

async def can_daytona_start_with_semaphore(task_path: str, semaphore: asyncio.Semaphore) -> bool:
    """
    Check if a task can be started by Daytona with semaphore concurrency control.
    """
    async with semaphore:
        return await can_daytona_start(task_path)

async def can_daytona_start_all_paths_async(task_paths: List[str], max_concurrent: int = 10) -> List[bool]:
    """
    Check if all tasks can be started by Docker (async version) with concurrency control.

    Args:
        task_paths: List of task paths to check
        max_concurrent: Maximum number of concurrent checks (default: 10)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = await tqdm_asyncio.gather(
        *[can_daytona_start_with_semaphore(task_path, semaphore) for task_path in task_paths],
        desc="Checking Docker start capability"
    )
    return results

def can_daytona_start_all_paths(task_paths: List[str], max_concurrent: int = 10) -> List[bool]:
    """
    Check if all tasks can be started by Docker with concurrency control.

    Args:
        task_paths: List of task paths to check
        max_concurrent: Maximum number of concurrent checks (default: 10)
    """
    return asyncio.run(can_daytona_start_all_paths_async(task_paths, max_concurrent))

@gcs_cache()
def filter_tasks_by_docker_start(parent_task_path: str, dataset_prefix: str = "docker_filtered", max_concurrent: int = 10) -> str:
    """
    Filter tasks by Docker start capability and create a new directory with tasks that can start.

    Args:
        parent_task_path (str): Path to the parent directory containing tasks
        dataset_prefix (str): Prefix for the output directory name
        max_concurrent (int): Maximum number of concurrent Docker start checks (default: 10)

    Returns:
        str: Path to the new directory containing filtered tasks that can start with Docker

    Raises:
        FileNotFoundError: If parent directory doesn't exist
        ValueError: If parent directory is empty or no tasks can start
    """
    parent_path = Path(parent_task_path)

    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_task_path}")

    if not parent_path.is_dir():
        raise ValueError(f"Parent path must be a directory: {parent_task_path}")

    # Get all task directories
    task_paths = [
        str(p)
        for p in parent_path.iterdir()
        if p.is_dir()
    ]

    if len(task_paths) == 0:
        raise ValueError(f"No task directories found in: {parent_task_path}")

    print(f"Checking {len(task_paths)} tasks for Docker start capability (max {max_concurrent} concurrent)...")

    # Check which tasks can start with Docker
    can_start = can_daytona_start_all_paths(task_paths, max_concurrent=max_concurrent)
    
    # Filter tasks that can start
    filtered_tasks = [path for path, can in zip(task_paths, can_start) if can]

    if len(filtered_tasks) == 0:
        raise ValueError(f"No tasks can start with Docker in: {parent_task_path}")

    # Create temporary directory for filtered tasks
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_"))
    print(f"Creating filtered tasks in: {temp_dir}")

    # Copy filtered tasks to new directory
    for i, task_path in enumerate(filtered_tasks):
        task_dir = Path(task_path)
        dest_path = temp_dir / task_dir.name
        shutil.copytree(task_dir, dest_path)

        if (i + 1) % 100 == 0:
            print(f"Copied {i + 1} tasks...")

    print(f"Successfully filtered {len(filtered_tasks)} tasks from {len(task_paths)} total tasks!")
    print(f"Filtered out {len(task_paths) - len(filtered_tasks)} tasks that cannot start with Docker.")
    return str(temp_dir)

async def _try_build_once(
    dockerfile: Path, *, cpu: int, memory_gb: int, disk_gb: int, gpu: int, timeout: int
) -> Tuple[bool, Optional[str]]:
    """Attempt a single Daytona build for a Dockerfile.

    Returns:
        (success, error_message) - error_message is None if successful

    Never raises exceptions - all failures return (False, error_msg).
    """
    resources = Resources(
        cpu=max(cpu, 1), memory=max(memory_gb, 1), disk=max(disk_gb, 1), gpu=max(gpu, 0)
    )
    params = CreateSandboxFromImageParams(
        image=Image.from_dockerfile(dockerfile),
        auto_delete_interval=0,
        resources=resources,
    )

    client = AsyncDaytona()
    sandbox = None
    try:
        sandbox = await client.create(params=params, timeout=timeout)
        # Successful build and start
        await sandbox.delete()
        return True, None
    except Exception as e:
        # Any failure returns (False, error_msg) - no exceptions propagated
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg
    finally:
        # Clean up sandbox if it exists
        if sandbox:
            try:
                await sandbox.delete()
            except Exception:
                pass
        # Don't close client here - let GC handle it to avoid race conditions
        # in concurrent execution where one client closing affects another's sandbox


async def validate_task_worker_async(
    task_dir: Path,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    gpu: int,
    timeout: int,
    progress_update: Optional[Callable[[str], None]] = None,
    num_retries: int = 0,
) -> Tuple[str, bool, Optional[str]]:
    """Async worker function for validation. Retry on any failure.

    Returns: (task_name, success, error_message)
    """
    name = task_dir.name
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        if progress_update:
            progress_update(f"{name}: no Dockerfile found")
        return name, False, "Dockerfile not found"

    last_error = None
    # Retry loop matching SWE-bench pattern
    for attempt in range(num_retries + 1):
        if attempt > 0:
            # Exponential backoff: min(2^attempt, 60) seconds
            delay = min(2**attempt, 60)
            if progress_update:
                progress_update(f"{name}: waiting {delay}s before retry...")
            await asyncio.sleep(delay)

        if progress_update:
            attempt_info = (
                f" (attempt {attempt + 1}/{num_retries + 1})" if num_retries > 0 else ""
            )
            progress_update(f"{name}: building{attempt_info}...")

        ok, error_msg = await _try_build_once(
            dockerfile,
            cpu=cpu,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            gpu=gpu,
            timeout=timeout,
        )

        if ok:
            # Success - break immediately
            if progress_update:
                progress_update(f"{name}: ✓ build successful")
            return name, True, None

        # Failed - store error message
        last_error = error_msg

        # Check if we should retry
        if attempt == num_retries:
            # Last attempt, accept failure
            if progress_update:
                progress_update(f"{name}: ✗ build failed")
            return name, False, last_error

        # Not the last attempt, will retry
        if progress_update:
            progress_update(f"{name}: ✗ build failed, will retry...")

    # Should not reach here
    return name, False, last_error


async def _run_single_validation(
    semaphore: asyncio.Semaphore,
    task_dir: Path,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    gpu: int,
    timeout: int,
    loading_progress: Progress,
    loading_progress_task,
    running_progress: Progress,
    name_to_path: Dict[str, Path],
    base_dir: Path,
    successes: List[Path],
    failures: List[Path],
    stats: Dict[str, int],
    start_time: float,
    num_retries: int,
    error_log: Path,
    cache_dir: Path,
) -> Tuple[str, bool]:
    """Run a single validation task with progress tracking."""
    async with semaphore:
        task_progress = running_progress.add_task(
            f"{task_dir.name}: starting...", total=None
        )

        def update_task_status(description: str):
            """Update the progress description for this task."""
            running_progress.update(task_progress, description=description)

        try:
            name, ok, error_msg = await validate_task_worker_async(
                task_dir,
                cpu,
                memory_gb,
                disk_gb,
                gpu,
                timeout,
                progress_update=update_task_status,
                num_retries=num_retries,
            )

            task_path = name_to_path.get(name, base_dir / name)
            if ok:
                successes.append(task_path)
                stats["success"] += 1
                # Copy to success cache immediately
                success_cache = cache_dir / "successes"
                success_cache.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    task_path, success_cache / task_path.name, dirs_exist_ok=True
                )
                # Keep the success message visible briefly
                await asyncio.sleep(0.5)
            else:
                failures.append(task_path)
                stats["failed"] += 1
                # Copy to failure cache immediately
                failure_cache = cache_dir / "failures"
                failure_cache.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    task_path, failure_cache / task_path.name, dirs_exist_ok=True
                )
                # Write error message to log file
                if error_msg:
                    import datetime

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(error_log, "a") as f:
                        f.write(f"[{timestamp}] ✗ {name} failed: {error_msg}\n")
                # Keep the failure message visible briefly
                await asyncio.sleep(0.5)

            loading_progress.update(
                loading_progress_task,
                advance=1,
                description=f"Stage 1: Daytona build validation (✓ {stats['success']} ✗ {stats['failed']})",
            )

            return name, ok
        finally:
            running_progress.remove_task(task_progress)


async def run_daytona_validation_async(
    task_dirs: List[Path],
    *,
    base_dir: Path,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    gpu: int,
    timeout: int,
    processes: int,
    num_retries: int = 3,
    cache_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[Path]]:
    """Async validation with Harbor-style concurrency and progress display."""
    if not task_dirs:
        return [], []

    # Create or use existing cache directory
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "daytona_validation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    success_cache = cache_dir / "successes"
    failure_cache = cache_dir / "failures"

    # Filter out already-validated tasks
    tasks_to_validate = []
    cached_successes = []
    cached_failures = []

    for task_dir in task_dirs:
        if (success_cache / task_dir.name).exists():
            cached_successes.append(task_dir)
        elif (failure_cache / task_dir.name).exists():
            cached_failures.append(task_dir)
        else:
            tasks_to_validate.append(task_dir)

    if cached_successes or cached_failures:
        CONSOLE.print(
            f"[dim]Skipping {len(cached_successes)} cached successes and {len(cached_failures)} cached failures[/dim]"
        )

    if not tasks_to_validate:
        return cached_successes, cached_failures

    name_to_path: Dict[str, Path] = {path.name: path for path in tasks_to_validate}
    semaphore = asyncio.Semaphore(max(1, processes))

    successes: List[Path] = list(cached_successes)
    failures: List[Path] = list(cached_failures)
    stats = {
        "success": len(cached_successes),
        "failed": len(cached_failures),
        "total": len(task_dirs),
    }
    start = time.perf_counter()

    # Create error log file immediately and print location
    error_log = cache_dir / "validation_errors.log"
    # Append to existing log if it exists
    mode = "a" if error_log.exists() else "w"
    with open(error_log, mode) as f:
        if mode == "w":
            f.write(f"# Validation errors log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            f.write(f"\n# Validation run - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.flush()
    CONSOLE.print(f"[dim]Cache dir: {cache_dir}[/dim]")
    CONSOLE.print(f"[dim]Error log: {error_log}[/dim]")

    loading_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    running_progress = Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}"),
    )

    with Live(
        Group(loading_progress, running_progress),
        console=CONSOLE,
        refresh_per_second=10,
    ):
        progress_task = loading_progress.add_task(
            "Stage 1: Daytona build validation",
            total=len(tasks_to_validate),
        )

        async with asyncio.TaskGroup() as tg:
            _ = [
                tg.create_task(
                    _run_single_validation(
                        semaphore,
                        task_dir,
                        cpu,
                        memory_gb,
                        disk_gb,
                        gpu,
                        timeout,
                        loading_progress,
                        progress_task,
                        running_progress,
                        name_to_path,
                        base_dir,
                        successes,
                        failures,
                        stats,
                        start,
                        num_retries,
                        error_log,
                        cache_dir,
                    )
                )
                for task_dir in tasks_to_validate
            ]

    return successes, failures


def run_daytona_validation(
    task_dirs: List[Path],
    *,
    base_dir: Path,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    gpu: int,
    timeout: int,
    processes: int,
    num_retries: int = 3,
    cache_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[Path]]:
    """Synchronous wrapper for async validation."""
    return asyncio.run(
        run_daytona_validation_async(
            task_dirs,
            base_dir=base_dir,
            cpu=cpu,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            gpu=gpu,
            timeout=timeout,
            processes=processes,
            num_retries=num_retries,
            cache_dir=cache_dir,
        )
    )


def ensure_output_dir_in_dockerfile(dockerfile: Path) -> None:
    """Ensure the staged Dockerfile creates /output for tasks expecting it."""

    if not dockerfile.exists():
        return

    content = dockerfile.read_text().splitlines()
    directive = "RUN mkdir -p /output && chmod 777 /output"
    if any(directive in line for line in content):
        return

    insert_idx = 0
    for idx, line in enumerate(content):
        if line.strip().upper().startswith("WORKDIR"):
            insert_idx = idx + 1
            break

    content.insert(insert_idx, directive)
    dockerfile.write_text("\n".join(content) + "\n")


def discover_tasks(extracted_root: Path) -> list[Path]:
    tasks: list[Path] = []
    for path in sorted(extracted_root.iterdir()):
        dockerfile = path / "environment" / "Dockerfile"
        if path.is_dir() and dockerfile.exists():
            ensure_output_dir_in_dockerfile(dockerfile)
            tasks.append(path)
    return tasks


def copy_successes(task_dirs: Iterable[Path], dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    for task in task_dirs:
        shutil.copytree(task, dest_root / task.name, dirs_exist_ok=True)
    return dest_root


@gcs_cache()
def filter_tasks_by_docker_start_working(
    base_dir: str,
    cpu: int = 1,
    memory_gb: int = 4,
    disk_gb: int = 10,
    gpu: int = 0,
    timeout: int = 600,
    processes: int = 64,
) -> str:
    base_path = Path(base_dir)
    # Use persistent cache directory
    cache_dir = Path(tempfile.gettempdir()) / "daytona_validation_cache"

    successes, failures = run_daytona_validation(
        discover_tasks(base_path),
        base_dir=base_path,
        cpu=cpu,
        memory_gb=memory_gb,
        disk_gb=disk_gb,
        gpu=gpu,
        timeout=timeout,
        processes=processes,
        cache_dir=cache_dir,
    )

    # Return the successes cache directory directly
    filtered_tasks_dir = cache_dir / "successes"
    print(f"Filtered out {len(failures)} tasks that cannot start with Docker.")
    print(f"Success tasks location: {filtered_tasks_dir}")
    return str(filtered_tasks_dir)


@gcs_cache()
def download_dcagent_dev_set_instructions() -> List[str]:
    """
    Download DCAgent/dev_set_71_tasks dataset from HuggingFace and extract instructions.
    
    Returns:
        List[str]: List of instruction texts from the dataset
    """
    from huggingface_hub import list_repo_files, hf_hub_download
    
    print("Downloading DCAgent/dev_set_71_tasks dataset instructions...")
    
    # List all files in the dataset
    files = list_repo_files("DCAgent/dev_set_71_tasks", repo_type="dataset")
    
    # Find all instruction.md files
    instruction_files = [f for f in files if "instruction.md" in f]
    
    instructions = []
    for file in instruction_files:
        try:
            # Download each instruction.md file
            local_file = hf_hub_download("DCAgent/dev_set_71_tasks", file, repo_type="dataset")
            with open(local_file, "r") as f:
                content = f.read()
                if content.strip():  # Only add non-empty instructions
                    instructions.append(content)
        except Exception as e:
            print(f"Warning: Could not download {file}: {e}")
            continue

    print(f"Extracted {len(instructions)} instructions from DCAgent dataset")
    return instructions

def convert_task_paths_to_hdf5(
    task_paths: List[str],
    output_path: Optional[str] = None,
    compression: str = "gzip",
    compression_opts: int = 4
) -> str:
    """
    Convert a list of task directory paths to a single HDF5 file.

    This function reads all files from each task directory and stores them
    as datasets in an HDF5 file. This enables fast subsampling, upsampling,
    and caching operations without copying many individual files.

    Args:
        task_paths: List of paths to task directories
        output_path: Path for output HDF5 file. If None, creates temp file.
        compression: HDF5 compression algorithm (gzip, lzf, or None)
        compression_opts: Compression level (0-9 for gzip)

    Returns:
        str: Path to created HDF5 file

    Example HDF5 structure:
        tasks.h5
        ├── task_0000/
        │   ├── instruction.md (dataset, dtype=string)
        │   ├── task.toml (dataset, dtype=string)
        │   ├── environment/Dockerfile (dataset, dtype=string)
        │   └── metadata (attributes: task_name, task_id, etc.)
        └── task_0001/
            └── ...
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 conversion. Install with: pip install h5py")

    if output_path is None:
        # Create temporary HDF5 file
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix="tasks_")
        os.close(fd)

    output_path = Path(output_path)
    print(f"Converting {len(task_paths)} tasks to HDF5: {output_path}")

    with h5py.File(output_path, 'w') as h5file:
        for i, task_path in enumerate(tqdm(task_paths, desc="Converting tasks to HDF5")):
            task_path = Path(task_path)

            if not task_path.exists() or not task_path.is_dir():
                print(f"Warning: Skipping invalid task path: {task_path}")
                continue

            # Create group for this task
            task_name = task_path.name
            task_group = h5file.create_group(f"task_{i:04d}")

            # Store task metadata as attributes
            task_group.attrs['task_name'] = task_name
            task_group.attrs['task_id'] = i
            task_group.attrs['original_path'] = str(task_path)

            # Walk through all files in task directory
            for file_path in task_path.rglob('*'):
                if file_path.is_file():
                    # Get relative path within task
                    rel_path = file_path.relative_to(task_path)
                    dataset_name = str(rel_path)

                    try:
                        # Try to read as text first
                        content = file_path.read_text(encoding='utf-8')
                        # Store as variable-length string dataset
                        task_group.create_dataset(
                            dataset_name,
                            data=content,
                            dtype=h5py.string_dtype(encoding='utf-8'),
                            compression=compression,
                            compression_opts=compression_opts if compression == 'gzip' else None
                        )
                    except (UnicodeDecodeError, ValueError):
                        # If text fails, store as binary
                        content = file_path.read_bytes()
                        task_group.create_dataset(
                            dataset_name,
                            data=np.frombuffer(content, dtype=np.uint8),
                            compression=compression,
                            compression_opts=compression_opts if compression == 'gzip' else None
                        )
                        task_group[dataset_name].attrs['binary'] = True

    print(f"Successfully created HDF5 file: {output_path}")
    return str(output_path)


def _extract_single_task(
    hdf5_path: str,
    task_group_name: str,
    output_path: Path
) -> str:
    """
    Extract a single task from HDF5 file. Helper function for parallel extraction.

    Args:
        hdf5_path: Path to HDF5 file
        task_group_name: Name of the task group to extract
        output_path: Base output directory

    Returns:
        str: Path to extracted task directory
    """
    import h5py

    with h5py.File(hdf5_path, 'r') as h5file:
        task_group = h5file[task_group_name]

        # Get task metadata - use original task_name from attrs
        original_task_name = task_group.attrs.get('task_name', task_group_name)
        dataset_prefix = task_group.attrs.get('dataset_prefix', 'task')
        task_id = task_group.attrs.get('task_id', 0)

        # Extract task index from group name (e.g., "task_0000" -> 0)
        # Use this to create a unique directory name even if task_name repeats
        task_index = int(task_group_name.split('_')[1])
        task_name = f"{dataset_prefix}-{task_index:04d}"

        # Create task directory
        task_dir = output_path / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # Extract all files from this task
        def extract_group(group, base_path):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # This is a file
                    file_path = base_path / key
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Check if binary
                    if item.attrs.get('binary', False):
                        # Write as binary
                        file_path.write_bytes(item[()].tobytes())
                    else:
                        # Write as text
                        content = item[()]
                        if isinstance(content, bytes):
                            content = content.decode('utf-8')
                        file_path.write_text(content, encoding='utf-8')
                elif isinstance(item, h5py.Group):
                    # This is a subdirectory
                    extract_group(item, base_path / key)

        extract_group(task_group, task_dir)
        return str(task_dir)


def extract_hdf5_to_task_paths(
    hdf5_path: str,
    output_dir: Optional[str] = None,
    task_indices: Optional[List[int]] = None,
    max_workers: Optional[int] = None
) -> List[str]:
    """
    Extract tasks from an HDF5 file back to individual task directories in parallel.

    Args:
        hdf5_path: Path to HDF5 file containing tasks
        output_dir: Directory to extract tasks to. If None, creates temp directory.
        task_indices: Optional list of task indices to extract. If None, extracts all.
        max_workers: Maximum number of parallel workers. If None, uses os.cpu_count().

    Returns:
        List[str]: Paths to extracted task directories
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 extraction. Install with: pip install h5py")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="extracted_tasks_")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get list of task groups to extract
    with h5py.File(hdf5_path, 'r') as h5file:
        task_groups = sorted([k for k in h5file.keys() if k.startswith('task_')])

        # Filter by indices if provided
        if task_indices is not None:
            task_groups = [f"task_{i:04d}" for i in task_indices if f"task_{i:04d}" in task_groups]

    print(f"Extracting {len(task_groups)} tasks from HDF5 to: {output_path}")
    print(f"Using {max_workers or os.cpu_count()} parallel workers")

    extracted_paths = []

    # Extract tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_extract_single_task, hdf5_path, task_group_name, output_path): task_group_name
            for task_group_name in task_groups
        }

        # Collect results as they complete with progress bar
        for future in tqdm(as_completed(future_to_task), total=len(task_groups), desc="Extracting tasks"):
            task_group_name = future_to_task[future]
            try:
                task_path = future.result()
                extracted_paths.append(task_path)
            except Exception as e:
                print(f"Error extracting {task_group_name}: {e}")

    print(f"Successfully extracted {len(extracted_paths)} tasks")
    return extracted_paths


def subsample_tasks_hdf5(
    hdf5_path: str,
    num_samples: int,
    output_path: Optional[str] = None,
    compression: str = "gzip",
    compression_opts: int = 4
) -> str:
    """
    Subsample tasks from an HDF5 file - much faster than directory copying.

    This function randomly samples tasks from an existing HDF5 file and creates
    a new HDF5 file with only the sampled tasks. This is orders of magnitude
    faster than copying individual task directories.

    Args:
        hdf5_path: Path to source HDF5 file containing tasks
        num_samples: Number of tasks to randomly sample
        output_path: Path for output HDF5 file. If None, creates temp file.
        compression: HDF5 compression algorithm (gzip, lzf, or None)
        compression_opts: Compression level (0-9 for gzip)

    Returns:
        str: Path to new HDF5 file with subsampled tasks
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 operations. Install with: pip install h5py")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix="subsampled_tasks_")
        os.close(fd)

    output_path = Path(output_path)

    with h5py.File(hdf5_path, 'r') as source_h5:
        # Get all task groups
        all_task_keys = sorted([k for k in source_h5.keys() if k.startswith('task_')])
        total_tasks = len(all_task_keys)

        if num_samples > total_tasks:
            print(f"Warning: Requested {num_samples} samples but only {total_tasks} available")
            num_samples = total_tasks

        # Randomly sample task indices
        sampled_indices = random.sample(range(total_tasks), num_samples)
        sampled_keys = [all_task_keys[i] for i in sorted(sampled_indices)]

        print(f"Subsampling {num_samples} tasks from {total_tasks} total tasks")
        print(f"Creating subsampled HDF5 file: {output_path}")

        with h5py.File(output_path, 'w') as dest_h5:
            for new_idx, old_key in enumerate(tqdm(sampled_keys, desc="Subsampling tasks")):
                new_key = f"task_{new_idx:04d}"
                # Copy entire group (all datasets and attributes)
                source_h5.copy(old_key, dest_h5, name=new_key)

    print(f"Successfully created subsampled HDF5 with {num_samples} tasks")
    return str(output_path)

@gcs_cache()
def upsample_tasks_hdf5(
    hdf5_path: str,
    num_samples: int,
    output_path: Optional[str] = None,
    compression: str = "gzip",
    compression_opts: int = 4
) -> str:
    """
    Upsample tasks from an HDF5 file by repeating them - much faster than directory copying.

    Args:
        hdf5_path: Path to source HDF5 file containing tasks
        num_samples: Target number of tasks after upsampling
        output_path: Path for output HDF5 file. If None, creates temp file.
        compression: HDF5 compression algorithm (gzip, lzf, or None)
        compression_opts: Compression level (0-9 for gzip)

    Returns:
        str: Path to new HDF5 file with upsampled tasks (or original if no upsampling needed)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 operations. Install with: pip install h5py")

    with h5py.File(hdf5_path, 'r') as source_h5:
        all_task_keys = sorted([k for k in source_h5.keys() if k.startswith('task_')])
        total_tasks = len(all_task_keys)

        if total_tasks >= num_samples:
            print(f"Already have {total_tasks} tasks (>= {num_samples}), no upsampling needed")
            return hdf5_path

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix="upsampled_tasks_")
        os.close(fd)

    output_path = Path(output_path)
    print(f"Upsampling from {total_tasks} to {num_samples} tasks")
    print(f"Creating upsampled HDF5 file: {output_path}")

    with h5py.File(hdf5_path, 'r') as source_h5:
        all_task_keys = sorted([k for k in source_h5.keys() if k.startswith('task_')])

        with h5py.File(output_path, 'w') as dest_h5:
            tasks_written = 0

            # Cycle through tasks until we reach target
            with tqdm(total=num_samples, desc="Upsampling tasks") as pbar:
                while tasks_written < num_samples:
                    for old_key in all_task_keys:
                        if tasks_written >= num_samples:
                            break

                        new_key = f"task_{tasks_written:04d}"
                        # Copy entire group
                        source_h5.copy(old_key, dest_h5, name=new_key)
                        tasks_written += 1
                        pbar.update(1)

    print(f"Successfully upsampled to {num_samples} tasks")
    return str(output_path)


def generate_tasks_from_questions_hdf5(
    questions: List[str],
    dataset_prefix: str = "task",
    output_path: Optional[str] = None,
    compression: str = "gzip",
    compression_opts: int = 4
) -> str:
    """
    Generate tasks from questions directly into an HDF5 file - avoids creating directories entirely.

    This function creates the HDF5 file directly without intermediate directory creation,
    making it much faster and more efficient.

    Args:
        questions: List of questions/instructions or tuples with (instruction, metadata, solution, test_sh, test_py)
        dataset_prefix: Prefix for task naming
        output_path: Path for output HDF5 file. If None, creates temp file.
        compression: HDF5 compression algorithm
        compression_opts: Compression level

    Returns:
        str: Path to created HDF5 file containing all tasks
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 operations. Install with: pip install h5py")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix=f"{dataset_prefix}_tasks_")
        os.close(fd)

    output_path = Path(output_path)
    print(f"Generating {len(questions)} tasks directly into HDF5: {output_path}")

    # Generate standard content
    standard_dockerfile = create_standard_dockerfile()
    standard_task_toml = create_standard_task_toml()

    with h5py.File(output_path, 'w') as h5file:
        for i, dataset in enumerate(tqdm(questions, desc="Generating tasks to HDF5")):
            # Parse dataset (could be string or tuple)
            if isinstance(dataset, str):
                instruction_content = dataset
                metadata = None
                solution_content = None
                test_sh_content = None
                test_py_content = None
            else:
                instruction_content = dataset[0]
                metadata = dataset[1] if len(dataset) > 1 else None
                solution_content = dataset[2] if len(dataset) > 2 else None
                test_sh_content = dataset[3] if len(dataset) > 3 else None
                test_py_content = dataset[4] if len(dataset) > 4 else None

            # Create group for this task
            task_name = f"{dataset_prefix}-{i:04d}"
            task_group = h5file.create_group(f"task_{i:04d}")

            # Store task metadata as attributes
            task_group.attrs['task_name'] = task_name
            task_group.attrs['task_id'] = i
            task_group.attrs['dataset_prefix'] = dataset_prefix

            # Create instruction.md
            task_group.create_dataset(
                "instruction.md",
                data=instruction_content,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # Create task.toml
            task_group.create_dataset(
                "task.toml",
                data=standard_task_toml,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # Create environment/Dockerfile
            task_group.create_dataset(
                "environment/Dockerfile",
                data=standard_dockerfile,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # Add metadata.json if provided
            if metadata is not None:
                if isinstance(metadata, (dict, list)):
                    metadata_str = json.dumps(metadata, indent=2)
                else:
                    metadata_str = str(metadata)

                task_group.create_dataset(
                    "metadata.json",
                    data=metadata_str,
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Add solution/solve.sh if provided
            if solution_content is not None:
                task_group.create_dataset(
                    "solution/solve.sh",
                    data=solution_content,
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Add tests/test.sh if provided
            if test_sh_content is not None:
                task_group.create_dataset(
                    "tests/test.sh",
                    data=test_sh_content,
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Add tests/test_state.py if provided
            if test_py_content is not None:
                task_group.create_dataset(
                    "tests/test_state.py",
                    data=test_py_content,
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

    print(f"Successfully generated {len(questions)} tasks in HDF5 format")
    return str(output_path)


@gcs_cache()
def generate_tasks_to_hdf5(
    instructions: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    solutions: Optional[List[str]] = None,
    test_sh: Optional[List[str]] = None,
    test_py: Optional[List[str]] = None,
    task_toml: Optional[List[str]] = None,
    dockerfiles: Optional[List[str]] = None,
    dataset_prefix: str = "task",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate tasks from separate lists directly into HDF5 format.

    Args:
        instructions: List of task instructions/problem statements (required)
        metadata: List of task metadata dicts (optional, same length as instructions)
        solutions: List of solution script contents (optional)
        test_sh: List of test.sh contents (optional)
        test_py: List of test_state.py contents (optional)
        task_toml: List of task.toml contents (optional)
        dockerfiles: List of Dockerfile contents (optional)
        dataset_prefix: Prefix for task naming
        output_path: Path for output HDF5 file. If None, creates temp file.

    Returns:
        str: Path to created HDF5 file
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    n = len(instructions)

    # Helper to get item from list or None
    def get_item(lst, idx):
        return lst[idx] if lst is not None and idx < len(lst) else None

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".h5", prefix=f"{dataset_prefix}_tasks_")
        os.close(fd)

    output_path = Path(output_path)
    print(f"Generating {n} tasks directly into HDF5: {output_path}")

    with h5py.File(output_path, 'w') as h5file:
        for i in tqdm(range(n), desc="Generating tasks to HDF5"):
            instruction_content = instructions[i]
            metadata_content = get_item(metadata, i)
            solution_content = get_item(solutions, i)
            test_sh_content = get_item(test_sh, i)
            test_py_content = get_item(test_py, i)
            task_toml_content = get_item(task_toml, i)
            dockerfile_content = get_item(dockerfiles, i)

            # Create group for this task
            task_name = f"{dataset_prefix}-{i:04d}"
            task_group = h5file.create_group(f"task_{i:04d}")

            # Store task metadata as attributes
            task_group.attrs['task_name'] = task_name
            task_group.attrs['task_id'] = i
            task_group.attrs['dataset_prefix'] = dataset_prefix

            # Helper function to create dataset
            def create_text_dataset(path, content):
                if content is not None:
                    task_group.create_dataset(
                        path,
                        data=content,
                        dtype=h5py.string_dtype(encoding='utf-8')
                    )

            # Add all files
            create_text_dataset("instruction.md", instruction_content)
            create_text_dataset("task.toml", task_toml_content)
            create_text_dataset("environment/Dockerfile", dockerfile_content)

            # Add metadata.json in workspace
            if metadata_content is not None:
                if isinstance(metadata_content, (dict, list)):
                    metadata_str = json.dumps(metadata_content, indent=2)
                else:
                    metadata_str = str(metadata_content)
                create_text_dataset("environment/workspace/metadata.json", metadata_str)

            # Add solution if provided
            if solution_content is not None:
                create_text_dataset("solution/solve.sh", solution_content)

            # Add tests if provided
            if test_sh_content is not None:
                create_text_dataset("tests/test.sh", test_sh_content)

            if test_py_content is not None:
                create_text_dataset("tests/test_state.py", test_py_content)

    print(f"Successfully generated {n} tasks in HDF5 format")
    return str(output_path)


def get_ngrams_autotokenizer(tokens: List[int], n: int) -> List[tuple]:
    """
    Extract n-grams from tokenized text.

    Args:
        tokens: List of token IDs from a tokenizer
        n: The size of n-grams to extract

    Returns:
        List of n-gram tuples
    """
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams


def decontaminate_ngram_overlap(
    values: List[str],
    benchmark_questions: List[str],
    ngram_size: int = 8,
    tokenizer_name: str = "bert-base-uncased",
) -> List[bool]:
    """
    Check for n-gram overlap between values and benchmark questions.

    Args:
        values: List of strings to check for contamination
        benchmark_questions: List of benchmark strings to decontaminate against
        ngram_size: Size of n-grams to use for overlap detection
        tokenizer_name: Name of the tokenizer to use for tokenization

    Returns:
        List of booleans where True indicates the value is contaminated
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    all_text_tokens = tokenizer(values)["input_ids"]
    all_benchmark_tokens = tokenizer(benchmark_questions)["input_ids"]

    # Process n-grams for benchmarks
    all_text_ngrams = []
    all_benchmark_ngrams = set()

    for benchmark_tokens in tqdm(all_benchmark_tokens, desc="Processing benchmark n-grams"):
        all_benchmark_ngrams.update(
            set(get_ngrams_autotokenizer(benchmark_tokens, ngram_size))
        )

    # Process n-grams for texts
    for text_tokens in tqdm(all_text_tokens, desc="Processing text n-grams"):
        text_ngrams = set(get_ngrams_autotokenizer(text_tokens, ngram_size))
        all_text_ngrams.append(text_ngrams)

    # Check each text for contamination
    contaminated_mask = []
    for text_ngram in tqdm(all_text_ngrams, desc="Checking contamination"):
        is_contaminated = bool(text_ngram & all_benchmark_ngrams)
        contaminated_mask.append(is_contaminated)

    return contaminated_mask


def _check_fuzzy_chunk(args):
    """Helper function for parallel fuzzy matching on a chunk of values."""
    values_chunk, benchmark_questions, similarity_threshold = args
    results = []
    for val in values_chunk:
        is_contaminated = False
        for bench in benchmark_questions:
            if fuzz.ratio(val, bench) / 100.0 >= similarity_threshold:
                is_contaminated = True
                break
            if fuzz.partial_ratio(val, bench) / 100.0 >= similarity_threshold:
                is_contaminated = True
                break
        results.append(is_contaminated)
    return results


def decontaminate_fuzzy_overlap(
    values: List[str],
    benchmark_questions: List[str],
    similarity_threshold: float = 0.8,
    n_jobs: int = -1,
    chunk_size: int = None,
) -> List[bool]:
    """
    Check for fuzzy string similarity between values and benchmark questions.
    Uses parallel processing for speed.

    Args:
        values: List of strings to check for contamination
        benchmark_questions: List of benchmark strings to decontaminate against
        similarity_threshold: Minimum similarity ratio (0.0 to 1.0) to consider contaminated
        n_jobs: Number of parallel workers. -1 means use all CPUs.
        chunk_size: Number of values per chunk. If None, auto-calculates based on n_jobs.

    Returns:
        List of booleans where True indicates the value is contaminated
    """
    from multiprocessing import Pool, cpu_count

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Auto-calculate chunk size to create ~n_jobs chunks
    if chunk_size is None:
        chunk_size = max(1, len(values) // n_jobs)

    # Split values into chunks
    chunks = [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]
    args = [(chunk, benchmark_questions, similarity_threshold) for chunk in chunks]

    print(f"Running fuzzy matching with {n_jobs} workers on {len(chunks)} chunks (chunk_size={chunk_size})...")

    with Pool(n_jobs) as pool:
        chunk_results = list(tqdm(
            pool.imap(_check_fuzzy_chunk, args),
            total=len(chunks),
            desc="Checking fuzzy similarity"
        ))

    # Flatten results
    contaminated_mask = [item for sublist in chunk_results for item in sublist]

    return contaminated_mask


def download_instructions_from_hf(repo_id: str) -> List[str]:
    """
    Download instructions from a HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repository ID

    Returns:
        List of instruction texts from the dataset

    Note:
        Uses HF_TOKEN environment variable for authentication.
    """
    print(f"Downloading {repo_id} dataset...")

    # Use download_hf_dataset to get the dataset path
    dataset_path = download_hf_dataset(repo_id=repo_id, use_snapshot=True)
    dataset_path = Path(dataset_path)

    # Find all instruction.md files in the downloaded dataset
    instruction_files = list(dataset_path.rglob("instruction.md"))

    instructions = []
    for file in tqdm(instruction_files, desc="Reading instructions"):
        try:
            content = file.read_text(encoding="utf-8")
            if content.strip():  # Only add non-empty instructions
                instructions.append(content)
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")
            continue

    print(f"Extracted {len(instructions)} instructions from {repo_id}")
    return instructions


def decontaminate_questions(
    questions: List[str],
    repo_ids: Union[str, List[str]] = [
        "DCAgent2/swebench-verified-random-100-folders",
        "DCAgent/dev_set_71_tasks",
    ],
    ngram_size: int = 8,
    tokenizer_name: str = "bert-base-uncased",
    similarity_threshold: float = 0.8,
    return_mask: bool = False,
) -> Union[List[str], Tuple[List[str], List[bool]]]:
    """
    Decontaminate a list of questions by removing those that overlap with
    benchmark instructions using both n-gram and fuzzy matching.

    Args:
        questions: List of question strings to decontaminate
        repo_ids: HuggingFace dataset repository ID(s) for benchmark questions.
                  Can be a single string or list of strings for multiple test sets.
        ngram_size: Size of n-grams to use for overlap detection
        tokenizer_name: Name of the tokenizer to use for n-gram tokenization
        similarity_threshold: Minimum similarity ratio (0.0 to 1.0) for fuzzy matching
        return_mask: If True, also return the contamination mask

    Returns:
        If return_mask is False: List of decontaminated questions
        If return_mask is True: Tuple of (decontaminated questions, contamination mask)

    Note:
        A question is marked as contaminated if it fails EITHER check (n-gram OR fuzzy)
        against ANY of the benchmark test sets.
        Uses HF_TOKEN environment variable for authentication.
    """
    # Normalize to list
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]

    # Download benchmark instructions from all repos
    all_benchmark_instructions = []
    for repo_id in repo_ids:
        print(f"\nDownloading from: {repo_id}")
        instructions = download_instructions_from_hf(repo_id)
        all_benchmark_instructions.extend(instructions)
        print(f"  -> Added {len(instructions)} instructions (total: {len(all_benchmark_instructions)})")

    if not all_benchmark_instructions:
        print("Warning: No benchmark instructions found. Returning all questions.")
        if return_mask:
            return questions, [False] * len(questions)
        return questions

    print(f"\nDecontaminating {len(questions)} questions against {len(all_benchmark_instructions)} total benchmark instructions...")

    # N-gram overlap detection
    print("\n[1/2] Running n-gram overlap detection...")
    ngram_mask = decontaminate_ngram_overlap(
        values=questions,
        benchmark_questions=all_benchmark_instructions,
        ngram_size=ngram_size,
        tokenizer_name=tokenizer_name,
    )
    ngram_contaminated = sum(ngram_mask)
    print(f"N-gram found {ngram_contaminated} contaminated questions")

    # Fuzzy matching detection
    print("\n[2/2] Running fuzzy matching detection...")
    fuzzy_mask = decontaminate_fuzzy_overlap(
        values=questions,
        benchmark_questions=all_benchmark_instructions,
        similarity_threshold=similarity_threshold,
    )
    fuzzy_contaminated = sum(fuzzy_mask)
    print(f"Fuzzy matching found {fuzzy_contaminated} contaminated questions")

    # Combine with OR (contaminated if either method flags it)
    combined_mask = [n or f for n, f in zip(ngram_mask, fuzzy_mask)]

    # Filter out contaminated questions
    clean_questions = [q for q, is_contaminated in zip(questions, combined_mask) if not is_contaminated]

    num_removed = len(questions) - len(clean_questions)
    print(f"\n=== Summary ===")
    print(f"Total contaminated (n-gram OR fuzzy): {num_removed} ({num_removed/len(questions)*100:.1f}%)")
    print(f"Remaining clean questions: {len(clean_questions)}")

    if return_mask:
        return clean_questions, combined_mask
    return clean_questions
