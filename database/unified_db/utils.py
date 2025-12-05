#!/usr/bin/env python3
"""
Dataset Registration Utilities for OT-Agents

This module provides utility functions for dataset registration using Supabase
with support for both HuggingFace and local parquet file datasets.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import warnings
from harbor.utils.traces_utils import export_traces

from supabase import Client

from .config import get_default_client, get_admin_client
from .models import *

logger = logging.getLogger(__name__)

# Optional heavy imports for parquet conversion
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

# HuggingFace Hub imports
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    HfApi = None  # type: ignore[assignment]
    create_repo = None  # type: ignore[assignment]

# HuggingFace datasets import for trace export
try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore[assignment]

# ==================== SUPABASE CLIENT ====================

def get_supabase_client(use_admin: bool = False) -> Client:
    """Get Supabase client for database operations."""
    if use_admin:
        return get_admin_client()
    return get_default_client()

def load_supabase_keys() -> bool:
    """Load Supabase credentials from DC_AGENT_SECRET_ENV (or legacy KEYS)."""
    keys_env = os.environ.get("DC_AGENT_SECRET_ENV") or os.environ.get("KEYS")
    if not keys_env:
        warnings.warn(
            "Supabase credentials not loaded: set DC_AGENT_SECRET_ENV to a secrets file "
            "to enable database registration."
        )
        return False

    keys_path = os.path.expandvars(keys_env)
    if not os.path.isfile(keys_path):
        warnings.warn(
            f"Supabase credentials file not found at '{keys_path}'. "
            "Model uploads will not be registered until DC_AGENT_SECRET_ENV points to a valid file."
        )
        return False

    try:
        with open(keys_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip('"').strip("'")
                os.environ[key] = os.path.expandvars(value)
    except Exception as exc:  # pragma: no cover - defensive parsing
        warnings.warn(
            f"Failed to load Supabase credentials from '{keys_path}': {exc!r}. "
            "Database registration will be skipped."
        )
        return False

    required = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        warnings.warn(
            "Missing Supabase settings "
            f"{', '.join(missing)} after loading secrets file. "
            "Model uploads will not be registered; ensure the secrets file exports these values."
        )
        return False

    return True

# ==================== DATASET UTILITIES ====================

def get_dataset_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a dataset from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('datasets').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving dataset by name {name}: {e}")
        return None

def get_dataset_by_id(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a dataset from the database by ID."""
    try:
        client = get_supabase_client()
        response = client.table('datasets').select('*').eq('id', dataset_id).execute()
        if not response.data:
            return None
        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving dataset with ID {dataset_id}: {e}")
        return None

def create_dataset(dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new dataset in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('datasets').insert(dataset_data).execute()

        if not response.data:
            raise ValueError("Failed to create dataset")

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


def update_dataset(dataset_id: str, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing dataset in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('datasets').update(dataset_data).eq('id', dataset_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update dataset with ID {dataset_id}")

        return clean_dataset_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating dataset {dataset_id}: {e}")
        raise


# ==================== DATASET REGISTRATION FUNCTIONS ====================

def register_hf_dataset(
    repo_name: str,
    dataset_type: str,
    name: Optional[str] = None,
    created_by: Optional[str] = None,
    data_generation_hash: Optional[str] = None,
    generation_start: Optional[datetime] = None,
    generation_end: Optional[datetime] = None,
    generation_parameters: Optional[Dict] = None,
    forced_update: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Register a HuggingFace dataset with comprehensive auto-filling.

    Auto-fills 12 fields from HF metadata:
    - id, creation_time, updated_at (system generated)
    - data_location, generation_status, creation_location (defaults)
    - generation_parameters, hf_fingerprint, hf_commit_hash, num_tasks (from HF API)
    - name, created_by (smart defaults with user override)

    Args:
        repo_name: HuggingFace repository name (e.g., "openai/gsm8k")
        dataset_type: Required - "SFT" or "RL"
        name: Optional override for dataset name (defaults to repo name)
        created_by: Optional override for creator (defaults to extracted from repo)
        data_generation_hash: Optional hash of code used to generate data
        generation_start: Optional generation start datetime
        generation_end: Optional generation end datetime
        generation_parameters: Optional additional generation parameters
        forced_update: If True, update existing dataset instead of returning early
        **kwargs: Additional overrides for any dataset fields

    Returns:
        Dictionary with dataset creation results {"success": bool, "dataset": dict, "error": str}
    """
    try:
        # Import HF libraries here to avoid dependency issues
        try:
            from datasets import load_dataset
            from huggingface_hub import dataset_info
        except ImportError:
            raise ImportError("datasets and huggingface_hub libraries required for HF dataset registration. Install with: pip install datasets huggingface_hub")

        logger.info(f"Registering HuggingFace dataset: {repo_name}")

        # Check if dataset already exists
        dataset_name = name or repo_name
        existing = get_dataset_by_name(dataset_name)
        if existing and not forced_update:
            logger.info(f"Dataset {dataset_name} already exists")
            return {"success": True, "dataset": existing, "exists": True}

        # Get HuggingFace dataset info
        try:
            hf_info = dataset_info(repo_name)
            logger.info(f"Retrieved HF metadata for {repo_name}")
        except Exception as e:
            logger.error(f"Failed to get HF dataset info for {repo_name}: {e}")
            return {"success": False, "error": f"Could not access HuggingFace dataset {repo_name}: {e}"}

        # Get dataset size (num_tasks)
        num_tasks = None
        try:
            dataset = load_dataset(repo_name)['train']
            # Handle different dataset structures
            if hasattr(dataset, '__len__'):
                num_tasks = len(dataset)
            elif hasattr(dataset, 'num_rows'):
                num_tasks = dataset.num_rows
            logger.info(f"Dataset size: {num_tasks} rows")
        except Exception as e:
            logger.warning(f"Could not determine dataset size for {repo_name}: {e}")
            num_tasks = None

        # Extract creator from repo name if not provided
        if not created_by:
            if '/' in repo_name:
                created_by = repo_name.split('/')[0]
            else:
                created_by = "hf-uploader"

        # Prepare auto-filled generation parameters
        auto_params = {
            "hf_repo": repo_name,
            "source": "huggingface_hub",
            "access_method": "datasets_library",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "hf_metadata": {
                "fingerprint": getattr(dataset, '_fingerprint', None) if 'dataset' in locals() else None,
                "commit_hash": getattr(hf_info, 'sha', None),
                "tags": getattr(hf_info, 'tags', []),
                "description": getattr(hf_info, 'description', ''),
            }
        }

        # Add user-provided generation parameters
        if generation_parameters:
            auto_params.update(generation_parameters)

        # Build dataset data with auto-filled fields
        now = datetime.now(timezone.utc)
        dataset_data = {
            # Auto-filled system fields
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),

            # Auto-filled from HF API
            "name": dataset_name,
            "created_by": created_by,
            "data_location": f"https://huggingface.co/datasets/{repo_name}",
            "creation_location": "HuggingFace",
            "generation_status": "completed",  # HF datasets are already generated
            "generation_parameters": auto_params,
            "generation_start": generation_start.isoformat() if generation_start else None,
            "generation_end": generation_end.isoformat() if generation_end else None,
            "hf_fingerprint": getattr(dataset, '_fingerprint', None) if 'dataset' in locals() else None,
            "hf_commit_hash": getattr(hf_info, 'sha', None),
            "num_tasks": num_tasks,

            # Required user field
            "dataset_type": dataset_type,

            # Optional user fields
            "data_generation_hash": data_generation_hash,
        }

        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if key != 'generation_parameters':  # Already handled above
                dataset_data[key] = value

        # Create or update dataset in database
        if existing and forced_update:
            logger.info(f"Updating existing dataset entry for {dataset_name}")
            result = update_dataset(existing['id'], dataset_data)
            logger.info(f"Successfully updated HF dataset: {dataset_name}")
            return {"success": True, "dataset": result, "updated": True}
        else:
            logger.info(f"Creating dataset entry for {dataset_name}")
            result = create_dataset(dataset_data)
            logger.info(f"Successfully registered HF dataset: {dataset_name}")
            return {"success": True, "dataset": result}

    except Exception as e:
        logger.error(f"Failed to register HF dataset {repo_name}: {e}")
        return {"success": False, "error": str(e)}


def register_local_parquet(
    file_path: str,
    name: str,
    created_by: str,
    dataset_type: str,
    data_generation_hash: Optional[str] = None,
    generation_start: Optional[datetime] = None,
    generation_end: Optional[datetime] = None,
    hf_fingerprint: Optional[str] = None,
    hf_commit_hash: Optional[str] = None,
    generation_parameters: Optional[Dict] = None,
    forced_update: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Register a local parquet file as a dataset with comprehensive auto-filling.

    Auto-fills 9 fields from file system:
    - id, creation_time, updated_at (system generated)
    - data_location, generation_status (from file path and status)
    - creation_location, generation_parameters (smart defaults)
    - num_tasks, last_modified (from file analysis)

    Args:
        file_path: Path to local parquet file
        name: Required - dataset name
        created_by: Required - creator/user name
        dataset_type: Required - "SFT" or "RL"
        data_generation_hash: Optional hash of code used to generate data
        generation_start: Optional datetime when generation started
        generation_end: Optional datetime when generation completed
        hf_fingerprint: Optional HF fingerprint if derived from HF dataset
        hf_commit_hash: Optional HF commit hash if derived from HF dataset
        generation_parameters: Optional additional generation parameters
        forced_update: If True, update existing dataset instead of returning early
        **kwargs: Additional overrides for any dataset fields

    Returns:
        Dictionary with dataset creation results {"success": bool, "dataset": dict, "error": str}
    """
    try:
        # Import pandas here to avoid dependency issues
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas library required for parquet file registration. Install with: pip install pandas")

        logger.info(f"Registering local parquet dataset: {file_path}")

        # Validate and get absolute path
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return {"success": False, "error": f"File does not exist: {abs_path}"}

        if not os.path.isfile(abs_path):
            return {"success": False, "error": f"Path is not a file: {abs_path}"}

        # Check if dataset already exists
        existing = get_dataset_by_name(name)
        if existing and not forced_update:
            logger.info(f"Dataset {name} already exists")
            return {"success": True, "dataset": existing, "exists": True}

        # Analyze parquet file
        try:
            # Read parquet file to get size and column info
            df = pd.read_parquet(abs_path)
            num_tasks = len(df)

            # Get file metadata
            file_stat = os.stat(abs_path)
            file_size = file_stat.st_size
            last_modified = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)

            logger.info(f"Parquet analysis: {num_tasks} rows, {len(df.columns)} columns, {file_size / 1024 / 1024:.2f} MB")

        except Exception as e:
            logger.error(f"Failed to analyze parquet file {abs_path}: {e}")
            return {"success": False, "error": f"Could not read parquet file: {e}"}

        # Prepare auto-filled generation parameters
        auto_params = {
            "source": "local_filesystem",
            "file_format": "parquet",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "file_metadata": {
                "absolute_path": abs_path,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / 1024 / 1024,
                "num_columns": len(df.columns),
                "column_names": list(df.columns),
                "column_dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            }
        }

        # Add user-provided generation parameters
        if generation_parameters:
            auto_params.update(generation_parameters)

        # Build dataset data with auto-filled fields
        now = datetime.now(timezone.utc)
        dataset_data = {
            # Auto-filled system fields
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),

            # Auto-filled from file system
            "data_location": abs_path,
            "generation_status": "completed",  # File exists so generation is complete
            "creation_location": "local",
            "generation_parameters": auto_params,
            "num_tasks": num_tasks,
            "last_modified": last_modified.isoformat(),

            # Required user fields
            "name": name,
            "created_by": created_by,
            "dataset_type": dataset_type,

            # Optional user fields
            "data_generation_hash": data_generation_hash,
            "generation_start": generation_start.isoformat() if generation_start else None,
            "generation_end": generation_end.isoformat() if generation_end else None,
            "hf_fingerprint": hf_fingerprint,
            "hf_commit_hash": hf_commit_hash,
        }

        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if key != 'generation_parameters':  # Already handled above
                dataset_data[key] = value

        # Create or update dataset in database
        if existing and forced_update:
            logger.info(f"Updating existing dataset entry for {name}")
            result = update_dataset(existing['id'], dataset_data)
            logger.info(f"Successfully updated local parquet dataset: {name}")
            return {"success": True, "dataset": result, "updated": True}
        else:
            logger.info(f"Creating dataset entry for {name}")
            result = create_dataset(dataset_data)
            logger.info(f"Successfully registered local parquet dataset: {name}")
            return {"success": True, "dataset": result}

    except Exception as e:
        logger.error(f"Failed to register local parquet dataset {file_path}: {e}")
        return {"success": False, "error": str(e)}


# ==================== MODEL UTILITIES ====================

def get_model_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a model from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('models').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving model by name {name}: {e}")
        return None


def create_model(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new model in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('models').insert(model_data).execute()

        if not response.data:
            raise ValueError("Failed to create model")

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


def update_model(model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing model in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('models').update(model_data).eq('id', model_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update model with ID {model_id}")

        return clean_model_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        raise


# ==================== MODEL REGISTRATION FUNCTIONS ====================

def register_hf_model(
    repo_name: str,
    agent_id: str,
    training_start: datetime,
    name: Optional[str] = None,
    created_by: Optional[str] = None,
    base_model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dataset_names: Optional[str] = None,
    training_end: Optional[datetime] = None,
    training_type: Optional[str] = None,
    training_parameters: Optional[Dict] = None,
    wandb_link: Optional[str] = None,
    traces_location_s3: Optional[str] = None,
    description: Optional[str] = None,
    multiple_datasets: bool = False,
    forced_update: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Register a HuggingFace model with comprehensive auto-filling.

    Auto-fills 10-12 fields from HF metadata:
    - id, creation_time, updated_at (system generated)
    - weights_location, creation_location, is_external (defaults)
    - created_by, name (smart extraction from repo name)
    - training_parameters, description (from model card/config)
    - training_status (inferred as "completed")
    - base_model info (extracted from config if available)

    Args:
        repo_name: HuggingFace repository name (e.g., "meta-llama/Llama-2-7b")
        agent_id: Required - UUID of the agent that trained the model
        training_start: Required - When training started
        name: Optional override for model name (defaults to repo name)
        created_by: Optional override for creator (defaults to extracted from repo)
        base_model_id: Optional UUID of base model
        dataset_id: Optional UUID of training dataset (backwards compatible)
        dataset_names: Optional comma-separated dataset names for multi-dataset support
        training_end: Optional when training completed
        training_type: Optional "SFT" or "RL"
        training_parameters: Optional additional training parameters
        wandb_link: Optional Weights & Biases link
        traces_location_s3: Optional S3 location of training traces
        description: Optional model description
        multiple_datasets: If True, parse dataset_names and validate each dataset exists
        forced_update: If True, update existing model instead of returning early
        **kwargs: Additional overrides for any model fields

    Returns:
        Dictionary with model creation results {"success": bool, "model": dict, "error": str}
    """
    try:
        # Import HF libraries here to avoid dependency issues
        try:
            from huggingface_hub import model_info, hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub library required for HF model registration. Install with: pip install huggingface_hub")

        logger.info(f"Registering HuggingFace model: {repo_name}")

        # Check if model already exists
        model_name = name or repo_name
        existing = get_model_by_name(model_name)
        if existing and not forced_update:
            logger.info(f"Model {model_name} already exists")
            return {"success": True, "model": existing['id'], "exists": True}

        # Handle multi-dataset validation
        final_dataset_id = None
        final_dataset_names = None

        if multiple_datasets and dataset_names:
            # Multi-dataset mode: validate all datasets exist by name
            datasets_to_validate = [d.strip() for d in dataset_names.split(',') if d.strip()]

            # Validate each dataset exists
            for ds_name in datasets_to_validate:
                existing_ds = get_dataset_by_name(ds_name)
                if not existing_ds:
                    logger.error(f"Dataset '{ds_name}' not found")
                    return {"success": False, "error": f"Dataset '{ds_name}' not found"}

            final_dataset_id = None  # Don't use single dataset_id in multi mode
            final_dataset_names = dataset_names
            logger.info(f"Validated {len(datasets_to_validate)} datasets for multi-dataset model")

        elif dataset_id:
            # Single dataset mode (backwards compat): validate single UUID
            existing_ds = get_dataset_by_id(dataset_id)
            if not existing_ds:
                logger.error(f"Dataset {dataset_id} not found")
                return {"success": False, "error": f"Dataset {dataset_id} not found"}

            final_dataset_id = dataset_id
            final_dataset_names = None

        # Get HuggingFace model info
        try:
            hf_info = model_info(repo_name)
            logger.info(f"Retrieved HF metadata for {repo_name}")
        except Exception as e:
            logger.error(f"Failed to get HF model info for {repo_name}: {e}")
            return {"success": False, "error": f"Could not access HuggingFace model {repo_name}: {e}"}

        # Try to get model config
        model_config = {}
        try:
            # Try to download and read config.json
            config_path = hf_hub_download(repo_id=repo_name, filename="config.json")
            import json
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            logger.info(f"Retrieved model config for {repo_name}")
        except Exception as e:
            logger.warning(f"Could not retrieve config.json for {repo_name}: {e}")

        # Extract creator from repo name if not provided
        if not created_by:
            if '/' in repo_name:
                created_by = repo_name.split('/')[0]
            else:
                created_by = "hf-uploader"

        # Extract description from model card if not provided
        if not description:
            description = getattr(hf_info, 'cardData', {}).get('description') or getattr(hf_info, 'description', '')

        # Prepare auto-filled training parameters
        auto_params = {
            "hf_repo": repo_name,
            "source": "huggingface_hub",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "hf_metadata": {
                "model_id": getattr(hf_info, 'modelId', None),
                "sha": getattr(hf_info, 'sha', None),
                "tags": getattr(hf_info, 'tags', []),
                "pipeline_tag": getattr(hf_info, 'pipeline_tag', None),
                "library_name": getattr(hf_info, 'library_name', None),
                "downloads": getattr(hf_info, 'downloads', None),
                "likes": getattr(hf_info, 'likes', None),
            },
            "model_config": model_config,
        }

        # Add user-provided training parameters
        if training_parameters:
            auto_params.update(training_parameters)

        # Build model data with auto-filled fields
        now = datetime.now(timezone.utc)
        model_data = {
            # Auto-filled system fields
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),

            # Auto-filled from HF
            "name": model_name,
            "created_by": created_by,
            "weights_location": f"https://huggingface.co/{repo_name}",
            "creation_location": "HuggingFace",
            "is_external": True,
            "training_status": "completed",  # HF models are already trained
            "training_parameters": auto_params,
            "description": description,

            # Required user fields
            "agent_id": agent_id,
            "training_start": training_start.isoformat(),

            # Optional user fields
            "base_model_id": base_model_id,
            "dataset_id": final_dataset_id,
            "dataset_names": final_dataset_names,
            "training_end": training_end.isoformat() if training_end else now.isoformat(),
            "training_type": training_type,
            "wandb_link": wandb_link,
            "traces_location_s3": traces_location_s3,
        }

        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if key != 'training_parameters':  # Already handled above
                model_data[key] = value

        # Create or update model in database
        if existing and forced_update:
            logger.info(f"Updating existing model entry for {model_name}")
            result = update_model(existing['id'], model_data)
            logger.info(f"Successfully updated HF model: {model_name}")
            return {"success": True, "model": result['id'], "updated": True}
        else:
            logger.info(f"Creating model entry for {model_name}")
            result = create_model(model_data)
            logger.info(f"Successfully registered HF model: {model_name}")
            return {"success": True, "model": result['id']}

    except Exception as e:
        logger.error(f"Failed to register HF model {repo_name}: {e}")
        return {"success": False, "error": str(e)}


def register_local_model(
    model_path: str,
    name: str,
    created_by: str,
    agent_id: str,
    training_start: datetime,
    base_model_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    dataset_names: Optional[str] = None,
    training_end: Optional[datetime] = None,
    training_type: Optional[str] = None,
    training_parameters: Optional[Dict] = None,
    wandb_link: Optional[str] = None,
    traces_location_s3: Optional[str] = None,
    description: Optional[str] = None,
    multiple_datasets: bool = False,
    forced_update: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Register a local model with comprehensive auto-filling.

    Auto-fills 8-10 fields from filesystem:
    - id, creation_time, updated_at (system generated)
    - weights_location (absolute path)
    - creation_location, is_external (defaults)
    - training_parameters (from config files)
    - training_status (inferred from files)
    - description (from README if present)
    - Model framework detection

    Args:
        model_path: Path to local model directory
        name: Required - model name
        created_by: Required - creator/user name
        agent_id: Required - UUID of the agent that trained the model
        training_start: Required - When training started
        base_model_id: Optional UUID of base model
        dataset_id: Optional UUID of training dataset (backwards compatible)
        dataset_names: Optional comma-separated dataset names for multi-dataset support
        training_end: Optional when training completed
        training_type: Optional "SFT" or "RL"
        training_parameters: Optional additional training parameters
        wandb_link: Optional Weights & Biases link
        traces_location_s3: Optional S3 location of training traces
        description: Optional model description
        multiple_datasets: If True, parse dataset_names and validate each dataset exists
        forced_update: If True, update existing model instead of returning early
        **kwargs: Additional overrides for any model fields

    Returns:
        Dictionary with model creation results {"success": bool, "model": dict, "error": str}
    """
    try:
        logger.info(f"Registering local model: {model_path}")

        # Validate and get absolute path
        abs_path = os.path.abspath(model_path)
        if not os.path.exists(abs_path):
            return {"success": False, "error": f"Model path does not exist: {abs_path}"}

        if not os.path.isdir(abs_path):
            return {"success": False, "error": f"Model path is not a directory: {abs_path}"}

        # Check if model already exists
        existing = get_model_by_name(name)
        if existing and not forced_update:
            logger.info(f"Model {name} already exists")
            return {"success": True, "model": existing, "exists": True}

        # Handle multi-dataset validation
        final_dataset_id = None
        final_dataset_names = None

        if multiple_datasets and dataset_names:
            # Multi-dataset mode: validate all datasets exist by name
            datasets_to_validate = [d.strip() for d in dataset_names.split(',') if d.strip()]

            # Validate each dataset exists
            for ds_name in datasets_to_validate:
                existing_ds = get_dataset_by_name(ds_name)
                if not existing_ds:
                    logger.error(f"Dataset '{ds_name}' not found")
                    return {"success": False, "error": f"Dataset '{ds_name}' not found"}

            final_dataset_id = None  # Don't use single dataset_id in multi mode
            final_dataset_names = dataset_names
            logger.info(f"Validated {len(datasets_to_validate)} datasets for multi-dataset model")

        elif dataset_id:
            # Single dataset mode (backwards compat): validate single UUID
            existing_ds = get_dataset_by_id(dataset_id)
            if not existing_ds:
                logger.error(f"Dataset {dataset_id} not found")
                return {"success": False, "error": f"Dataset {dataset_id} not found"}

            final_dataset_id = dataset_id
            final_dataset_names = None

        # Analyze model directory
        model_files = os.listdir(abs_path)
        model_info = {
            "files": model_files,
            "total_size_bytes": sum(os.path.getsize(os.path.join(abs_path, f))
                                   for f in model_files if os.path.isfile(os.path.join(abs_path, f))),
        }

        # Try to load config.json if it exists
        config_data = {}
        config_path = os.path.join(abs_path, "config.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"Loaded config.json from {abs_path}")
            except Exception as e:
                logger.warning(f"Could not load config.json: {e}")

        # Try to extract description from README if not provided
        if not description:
            readme_path = os.path.join(abs_path, "README.md")
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r') as f:
                        description = f.read()[:500]  # First 500 chars
                    logger.info(f"Extracted description from README.md")
                except Exception as e:
                    logger.warning(f"Could not read README.md: {e}")

        # Detect model framework
        framework = "unknown"
        if "pytorch_model.bin" in model_files or "model.safetensors" in model_files:
            framework = "pytorch"
        elif "saved_model.pb" in model_files:
            framework = "tensorflow"
        elif "model.onnx" in model_files:
            framework = "onnx"
        elif config_data.get("_name_or_path"):
            framework = "transformers"

        # Determine training status
        training_status = "completed" if training_end else "in_progress"

        # Prepare auto-filled training parameters
        auto_params = {
            "source": "local_filesystem",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_framework": framework,
            "model_metadata": {
                "absolute_path": abs_path,
                "total_size_bytes": model_info["total_size_bytes"],
                "total_size_mb": model_info["total_size_bytes"] / 1024 / 1024,
                "num_files": len(model_files),
                "file_list": model_files[:20],  # First 20 files
            },
            "config": config_data,
        }

        # Add user-provided training parameters
        if training_parameters:
            auto_params.update(training_parameters)

        # Build model data with auto-filled fields
        now = datetime.now(timezone.utc)
        model_data = {
            # Auto-filled system fields
            "creation_time": now.isoformat(),
            "updated_at": now.isoformat(),

            # Auto-filled from filesystem
            "weights_location": abs_path,
            "creation_location": "local",
            "is_external": False,
            "training_status": training_status,
            "training_parameters": auto_params,
            "description": description,

            # Required user fields
            "name": name,
            "created_by": created_by,
            "agent_id": agent_id,
            "training_start": training_start.isoformat(),

            # Optional user fields
            "base_model_id": base_model_id,
            "dataset_id": final_dataset_id,
            "dataset_names": final_dataset_names,
            "training_end": training_end.isoformat() if training_end else None,
            "training_type": training_type,
            "wandb_link": wandb_link,
            "traces_location_s3": traces_location_s3,
        }

        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if key != 'training_parameters':  # Already handled above
                model_data[key] = value

        # Create or update model in database
        if existing and forced_update:
            logger.info(f"Updating existing model entry for {name}")
            result = update_model(existing['id'], model_data)
            logger.info(f"Successfully updated local model: {name}")
            return {"success": True, "model": result, "updated": True}
        else:
            logger.info(f"Creating model entry for {name}")
            result = create_model(model_data)
            logger.info(f"Successfully registered local model: {name}")
            return {"success": True, "model": result}

    except Exception as e:
        logger.error(f"Failed to register local model {model_path}: {e}")
        return {"success": False, "error": str(e)}


# ==================== BASE MODEL CONSTANTS ====================

# Unix epoch sentinel value for external/base models that don't have real training start times
BASE_MODEL_TRAINING_START_SENTINEL = "1970-01-01T00:00:00Z"


# ==================== TRAINED MODEL UTILITIES ====================


# ---- Helper: parse HF model name from URLs ----
import re
HF_RE = re.compile(r'https?://(?:www\.)?huggingface\.co/([^/\s]+)/([^/\s#?]+)')

def parse_hf_model_name(val: Any) -> Optional[str]:
    """
    Accepts a string or dict; tries to find an HF URL and return 'org/repo'.
    Handles URLs like:
      - https://huggingface.co/ORG/REPO
      - https://huggingface.co/ORG/REPO/tree/main
      - https://huggingface.co/ORG/REPO/blob/main/config.json
    """
    if isinstance(val, dict):
        # Common places to look
        for k in ("training_parameters", "raw", "weights_location", "url", "hf_url"):
            v = val.get(k)
            if isinstance(v, str):
                m = HF_RE.search(v)
                if m:
                    return f"{m.group(1)}/{m.group(2)}"
        # Also scan all string values conservatively
        for v in val.values():
            if isinstance(v, str):
                m = HF_RE.search(v)
                if m:
                    return f"{m.group(1)}/{m.group(2)}"
        return None
    if isinstance(val, str):
        m = HF_RE.search(val)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    return None

def register_trained_model(
    training_record: Dict[str, Any],
    forced_update: bool = False
) -> Dict[str, Any]:
    """
    Register a newly trained model (SFT/RL)
    
    Expected training_record keys:
      - agent_name (str): trainer agent name
      - training_start (str|datetime)
      - training_end (str|datetime|None)
      - created_by (str|None)
      - base_model_name (str)
      - dataset_name (str)  # HF dataset id
      - training_type (str): 'SFT' | 'RL'
      - training_parameters (str|dict): JSON or JSON-serializable
      - wandb_link (str|None)
      - traces_location_s3 (str|None)
    """
    try:
        # ---- Validate required fields ----
        agent_name = training_record.get('agent_name')
        base_model_name = training_record.get('base_model_name')
        dataset_name = training_record.get('dataset_name')
        training_type = training_record.get('training_type')
        if not agent_name:
            return {"success": False, "error": "agent_name is required"}
        if not base_model_name:
            return {"success": False, "error": "base_model_name is required"}
        if not dataset_name:
            return {"success": False, "error": "dataset_name is required"}
        if training_type not in ('SFT', 'RL'):
            return {"success": False, "error": "training_type must be 'SFT' or 'RL'"}

        # ---- Parse timestamps ----
        def _parse_ts(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace('Z', '+00:00')) if val.endswith('Z') else datetime.fromisoformat(val)
            raise ValueError("timestamp must be datetime or ISO string")

        raw_start = training_record.get('training_start')
        if not raw_start:
            return {"success": False, "error": "training_start is required"}
        training_start_dt = _parse_ts(raw_start)
        training_end_dt = _parse_ts(training_record.get('training_end'))

        # ---- Normalize training_parameters ----
        params = training_record.get('training_parameters')
        if params is None:
            training_params = {}
        elif isinstance(params, dict):
            training_params = params
        elif isinstance(params, str):
            try:
                training_params = json.loads(params)
            except Exception:
                training_params = {"raw": params}
        else:
            training_params = {"raw": params}

        created_by = training_record.get('created_by') or ''
        wandb_link = training_record.get('wandb_link') or ''
        traces_location_s3 = training_record.get('traces_location_s3') or ''
        explicit_name = (training_record.get('model_name') or '').strip()

        # ---- Ensure agent exists ----
        agent_res = register_agent(name=agent_name)
        if not agent_res.get('success'):
            return agent_res
        agent = agent_res['agent']
        agent_id = agent['id']

        # ---- Ensure dataset(s) exist (HF) ----
        def _normalize_dataset_list(raw: Any) -> List[str]:
            if raw is None:
                return []
            if isinstance(raw, str):
                parts = raw.split(',')
            elif isinstance(raw, (list, tuple, set)):
                parts = list(raw)
            else:
                parts = [raw]
            norm: List[str] = []
            for part in parts:
                name = str(part).strip()
                if name and name not in norm:
                    norm.append(name)
            return norm

        dataset_names_raw = training_record.get('dataset_names')
        dataset_list = _normalize_dataset_list(dataset_names_raw) or _normalize_dataset_list(dataset_name)
        if not dataset_list:
            return {"success": False, "error": "No valid dataset_name(s) provided"}

        dataset_id: Optional[str] = None
        dataset_names_csv: Optional[str] = None

        if len(dataset_list) == 1:
            dataset_name_single = dataset_list[0]
            ds = get_dataset_by_name(dataset_name_single)
            if not ds:
                ds_res = register_hf_dataset(
                    repo_name=dataset_name_single,
                    dataset_type=training_type,
                    name=dataset_name_single,
                    created_by=created_by,
                )
                if not ds_res.get('success'):
                    return {"success": False, "error": ds_res.get('error', 'Dataset registration failed')}
                ds = ds_res['dataset']
            dataset_id = ds['id']
        else:
            dataset_names_csv = ",".join(dataset_list)
            for name in dataset_list:
                ds = get_dataset_by_name(name)
                if not ds:
                    ds_res = register_hf_dataset(
                        repo_name=name,
                        dataset_type=training_type,
                        name=name,
                        created_by=created_by,
                    )
                    if not ds_res.get('success'):
                        return {"success": False, "error": ds_res.get('error', 'Dataset registration failed')}

        # ---- Ensure base model exists WITH training_start sentinel ----
        base_m = get_model_by_name(base_model_name)
        if not base_m:
            now_ts = datetime.now(timezone.utc).isoformat()
            base_model_training_start = BASE_MODEL_TRAINING_START_SENTINEL  # e.g. "1970-01-01T00:00:00+00:00"
            base_payload = {
                "name": base_model_name,
                "created_by": (created_by or (base_model_name.split('/')[0] if '/' in base_model_name else "hf-uploader")),
                "creation_location": "HuggingFace",
                "creation_time": now_ts,
                "updated_at": now_ts,
                "is_external": True,
                "weights_location": f"https://huggingface.co/{base_model_name}",
                "training_status": "completed",
                "training_start": base_model_training_start,
                "agent_id": "6047d4e4-05de-4d33-867d-c4946ecfbd65",
                "training_parameters": {
                    "source": "huggingface_hub",
                    "registered_at": now_ts,
                    "base_model": True,
                },
            }
            base_m = create_model(base_payload)
        base_model_id = base_m['id']

        # ---- Decide trained model name (prefer HF repo if present) ----
        def looks_like_org_repo(name: str) -> bool:
            return '/' in name and not name.endswith('/')

        hf_from_explicit = explicit_name if looks_like_org_repo(explicit_name) else None
        hf_from_params = parse_hf_model_name(params) or parse_hf_model_name(training_params)

        if hf_from_explicit:
            model_name = hf_from_explicit
        elif hf_from_params:
            model_name = hf_from_params
        else: # Must provide the exact HF model repo name
            model_name = training_record.get('hf_model_repo_name')
            if not model_name:
                return {"success": False, "error": "model_name or hf_model_repo_name required"}

        # ---- Build model row ----
        existing = get_model_by_name(model_name)
        now_ts = datetime.now(timezone.utc).isoformat()
        # If we detected an HF repo name, link directly to it; otherwise keep legacy path.
        weights_location = f"https://huggingface.co/{model_name}"
        is_external = True

        training_status = 'completed' if training_end_dt else 'in_progress'

        model_data = {
            "name": model_name,
            "created_by": created_by,
            "creation_location": "HuggingFace",
            "creation_time": now_ts,
            "updated_at": now_ts,
            "is_external": is_external,
            "weights_location": weights_location,
            "training_status": training_status,
            "training_parameters": training_params,
            "description": None,

            # FKs / metadata
            "agent_id": agent_id,
            "base_model_id": base_model_id,
            "dataset_id": dataset_id,
            "dataset_names": dataset_names_csv,
            "training_type": training_type,

            # Times
            "training_start": training_start_dt.isoformat(),
            "training_end": training_end_dt.isoformat() if training_end_dt else None,

            # Optional links
            "wandb_link": wandb_link,
            "traces_location_s3": traces_location_s3,
        }

        # ---- Create or update ----
        if existing and not forced_update:
            return {"success": True, "model": existing, "exists": True}
        if existing and forced_update:
            updated = update_model(existing['id'], model_data)
            return {"success": True, "model": updated, "updated": True}
        created = create_model(model_data)
        return {"success": True, "model": created}

    except Exception as e:
        logger.error(f"Failed to register trained model: {e}")
        return {"success": False, "error": str(e)}

      
# ==================== AGENT UTILITIES ====================

def get_agent_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve an agent from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('agents').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving agent by name {name}: {e}")
        return None


def create_agent(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new agent in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('agents').insert(agent_data).execute()

        if not response.data:
            raise ValueError("Failed to create agent")

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


def update_agent(agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing agent in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('agents').update(agent_data).eq('id', agent_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update agent with ID {agent_id}")

        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating agent {agent_id}: {e}")
        raise


# ==================== AGENT REGISTRATION FUNCTIONS ====================

def register_agent(
    name: str,
    agent_version_hash: Optional[str] = None,
    description: Optional[str] = None,
    forced_update: bool = True
) -> Dict[str, Any]:
    """
    Register an evaluation agent with minimal auto-filling.

    Only auto-fills system fields (id and updated_at).
    All other fields must be provided manually.

    Args:
        name: Required - name of the agent
        agent_version_hash: Optional - SHA-256 hash of the agent version (64 characters)
        description: Optional - description of the agent and its capabilities
        forced_update: If True, update existing agent instead of returning early

    Returns:
        Dictionary with agent creation results {"success": bool, "agent": dict, "error": str}
    """
    try:
        logger.info(f"Registering agent: {name}")

        # Validate agent_version_hash length if provided
        if agent_version_hash and len(agent_version_hash) != 64:
            return {"success": False, "error": "agent_version_hash must be exactly 64 characters (SHA-256 hash)"}

        # Check if agent already exists
        existing = get_agent_by_name(name)
        if existing and not forced_update:
            logger.info(f"Agent {name} already exists")
            return {"success": True, "agent": existing, "exists": True}

        # Build agent data - only auto-fill system fields
        now = datetime.now(timezone.utc)
        agent_data = {
            # Auto-filled system fields
            "updated_at": now.isoformat(),

            # Manual user fields
            "name": name,
            "agent_version_hash": agent_version_hash,
            "description": description
        }

        # Create or update agent in database
        if existing and forced_update:
            logger.info(f"Updating existing agent entry for {name}")
            result = update_agent(existing['id'], agent_data)
            logger.info(f"Successfully updated agent: {name}")
            return {"success": True, "agent": result, "updated": True}
        else:
            logger.info(f"Creating agent entry for {name}")
            result = create_agent(agent_data)
            logger.info(f"Successfully registered agent: {name}")
            return {"success": True, "agent": result}

    except Exception as e:
        logger.error(f"Failed to register agent {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== BENCHMARK UTILITIES ====================

def get_benchmark_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a benchmark from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('benchmarks').select('*').eq('name', name).execute()

        if not response.data:
            return None

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving benchmark by name {name}: {e}")
        return None


def create_benchmark(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new benchmark in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('benchmarks').insert(benchmark_data).execute()

        if not response.data:
            raise ValueError("Failed to create benchmark")

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating benchmark: {e}")
        raise


def update_benchmark(benchmark_id: str, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing benchmark in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('benchmarks').update(benchmark_data).eq('id', benchmark_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update benchmark with ID {benchmark_id}")

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating benchmark {benchmark_id}: {e}")
        raise


# ==================== BENCHMARK REGISTRATION FUNCTIONS ====================

def register_benchmark(
    name: str,
    benchmark_version_hash: Optional[str] = None,
    is_external: bool = False,
    external_link: Optional[str] = None,
    description: Optional[str] = None,
    forced_update: bool = True
) -> Dict[str, Any]:
    """
    Register an evaluation benchmark with minimal auto-filling.

    Only auto-fills system fields (id and updated_at).
    All other fields must be provided manually.

    Args:
        name: Required - name of the benchmark
        benchmark_version_hash: Optional - SHA-256 hash of the benchmark version (64 characters)
        is_external: Optional - whether the benchmark is external (default False)
        external_link: Optional - link to external benchmark if applicable
        description: Optional - description of the benchmark and its purpose
        forced_update: If True, update existing benchmark instead of returning early

    Returns:
        Dictionary with benchmark creation results {"success": bool, "benchmark": dict, "error": str}
    """
    try:
        logger.info(f"Registering benchmark: {name}")

        # Validate benchmark_version_hash length if provided
        if benchmark_version_hash and len(benchmark_version_hash) != 64:
            return {"success": False, "error": "benchmark_version_hash must be exactly 64 characters (SHA-256 hash)"}

        # Check if benchmark already exists
        existing = get_benchmark_by_name(name)
        if existing and not forced_update:
            logger.info(f"Benchmark {name} already exists")
            return {"success": True, "benchmark": existing, "exists": True}

        # Build benchmark data - only auto-fill system fields
        now = datetime.now(timezone.utc)
        benchmark_data = {
            # Auto-filled system fields
            "updated_at": now.isoformat(),

            # Manual user fields
            "name": name,
            "benchmark_version_hash": benchmark_version_hash,
            "is_external": is_external,
            "external_link": external_link,
            "description": description
        }

        # Create or update benchmark in database
        if existing and forced_update:
            logger.info(f"Updating existing benchmark entry for {name}")
            result = update_benchmark(existing['id'], benchmark_data)
            logger.info(f"Successfully updated benchmark: {name}")
            return {"success": True, "benchmark": result, "updated": True}
        else:
            logger.info(f"Creating benchmark entry for {name}")
            result = create_benchmark(benchmark_data)
            logger.info(f"Successfully registered benchmark: {name}")
            return {"success": True, "benchmark": result}

    except Exception as e:
        logger.error(f"Failed to register benchmark {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== DELETE UTILITIES ====================

def delete_dataset_by_id(dataset_id: str) -> Dict[str, Any]:
    """
    Delete dataset from database by ID.

    Args:
        dataset_id: The ID of the dataset to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        client = get_supabase_client(use_admin=True)

        # Check if dataset exists and get its name for logging
        existing = client.table('datasets').select('name').eq('id', dataset_id).execute()
        if not existing.data:
            return {"success": False, "error": f"Dataset with ID {dataset_id} not found"}

        dataset_name = existing.data[0]['name']

        # Check for foreign key constraints (models referencing this dataset)
        models_using_dataset = client.table('models').select('id, name').eq('dataset_id', dataset_id).execute()
        if models_using_dataset.data:
            model_names = [model['name'] for model in models_using_dataset.data]
            return {"success": False, "error": f"Cannot delete dataset '{dataset_name}': referenced by models: {', '.join(model_names)}"}

        # Delete the dataset
        response = client.table('datasets').delete().eq('id', dataset_id).execute()

        if response.data:
            logger.info(f"Successfully deleted dataset: {dataset_name} (ID: {dataset_id})")
            return {"success": True, "message": f"Dataset '{dataset_name}' deleted successfully", "deleted_id": dataset_id}
        else:
            return {"success": False, "error": f"Failed to delete dataset with ID {dataset_id}"}

    except Exception as e:
        logger.error(f"Error deleting dataset by ID {dataset_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_dataset_by_name(name: str) -> Dict[str, Any]:
    """
    Delete dataset from database by name.

    Args:
        name: The name of the dataset to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        # First get the dataset ID
        dataset = get_dataset_by_name(name)
        if not dataset:
            return {"success": False, "error": f"Dataset with name '{name}' not found"}

        # Use delete by ID
        return delete_dataset_by_id(dataset['id'])

    except Exception as e:
        logger.error(f"Error deleting dataset by name {name}: {e}")
        return {"success": False, "error": str(e)}


def delete_agent_by_id(agent_id: str) -> Dict[str, Any]:
    """
    Delete agent from database by ID.

    Args:
        agent_id: The ID of the agent to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        client = get_supabase_client(use_admin=True)

        # Check if agent exists and get its name for logging
        existing = client.table('agents').select('name').eq('id', agent_id).execute()
        if not existing.data:
            return {"success": False, "error": f"Agent with ID {agent_id} not found"}

        agent_name = existing.data[0]['name']

        # Check for foreign key constraints (models referencing this agent)
        models_using_agent = client.table('models').select('id, name').eq('agent_id', agent_id).execute()
        if models_using_agent.data:
            model_names = [model['name'] for model in models_using_agent.data]
            return {"success": False, "error": f"Cannot delete agent '{agent_name}': referenced by models: {', '.join(model_names)}"}

        # Delete the agent
        response = client.table('agents').delete().eq('id', agent_id).execute()

        if response.data:
            logger.info(f"Successfully deleted agent: {agent_name} (ID: {agent_id})")
            return {"success": True, "message": f"Agent '{agent_name}' deleted successfully", "deleted_id": agent_id}
        else:
            return {"success": False, "error": f"Failed to delete agent with ID {agent_id}"}

    except Exception as e:
        logger.error(f"Error deleting agent by ID {agent_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_agent_by_name(name: str) -> Dict[str, Any]:
    """
    Delete agent from database by name.

    Args:
        name: The name of the agent to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        # First get the agent ID
        agent = get_agent_by_name(name)
        if not agent:
            return {"success": False, "error": f"Agent with name '{name}' not found"}

        # Use delete by ID
        return delete_agent_by_id(agent['id'])

    except Exception as e:
        logger.error(f"Error deleting agent by name {name}: {e}")
        return {"success": False, "error": str(e)}


def delete_model_by_id(model_id: str) -> Dict[str, Any]:
    """
    Delete model from database by ID.

    Args:
        model_id: The ID of the model to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        client = get_supabase_client(use_admin=True)

        # Check if model exists and get its name for logging
        existing = client.table('models').select('name').eq('id', model_id).execute()
        if not existing.data:
            return {"success": False, "error": f"Model with ID {model_id} not found"}

        model_name = existing.data[0]['name']

        # Check for foreign key constraints (models referencing this model as base_model)
        models_using_base = client.table('models').select('id, name').eq('base_model_id', model_id).execute()
        if models_using_base.data:
            dependent_names = [model['name'] for model in models_using_base.data]
            return {"success": False, "error": f"Cannot delete model '{model_name}': used as base model by: {', '.join(dependent_names)}"}

        # Delete the model
        response = client.table('models').delete().eq('id', model_id).execute()

        if response.data:
            logger.info(f"Successfully deleted model: {model_name} (ID: {model_id})")
            return {"success": True, "message": f"Model '{model_name}' deleted successfully", "deleted_id": model_id}
        else:
            return {"success": False, "error": f"Failed to delete model with ID {model_id}"}

    except Exception as e:
        logger.error(f"Error deleting model by ID {model_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_model_by_name(name: str) -> Dict[str, Any]:
    """
    Delete model from database by name.

    Args:
        name: The name of the model to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        # First get the model ID
        model = get_model_by_name(name)
        if not model:
            return {"success": False, "error": f"Model with name '{name}' not found"}

        # Use delete by ID
        return delete_model_by_id(model['id'])

    except Exception as e:
        logger.error(f"Error deleting model by name {name}: {e}")
        return {"success": False, "error": str(e)}


def delete_benchmark_by_id(benchmark_id: str) -> Dict[str, Any]:
    """
    Delete benchmark from database by ID.

    Args:
        benchmark_id: The ID of the benchmark to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        client = get_supabase_client(use_admin=True)

        # Check if benchmark exists and get its name for logging
        existing = client.table('benchmarks').select('name').eq('id', benchmark_id).execute()
        if not existing.data:
            return {"success": False, "error": f"Benchmark with ID {benchmark_id} not found"}

        benchmark_name = existing.data[0]['name']

        # Benchmarks currently have no foreign key constraints, so we can delete directly
        # Note: If benchmark results tables are added later, check those constraints here

        # Delete the benchmark
        response = client.table('benchmarks').delete().eq('id', benchmark_id).execute()

        if response.data:
            logger.info(f"Successfully deleted benchmark: {benchmark_name} (ID: {benchmark_id})")
            return {"success": True, "message": f"Benchmark '{benchmark_name}' deleted successfully", "deleted_id": benchmark_id}
        else:
            return {"success": False, "error": f"Failed to delete benchmark with ID {benchmark_id}"}

    except Exception as e:
        logger.error(f"Error deleting benchmark by ID {benchmark_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_benchmark_by_name(name: str) -> Dict[str, Any]:
    """
    Delete benchmark from database by name.

    Args:
        name: The name of the benchmark to delete

    Returns:
        {"success": bool, "message": str, "deleted_id": str} or
        {"success": bool, "error": str}
    """
    try:
        # First get the benchmark ID
        benchmark = get_benchmark_by_name(name)
        if not benchmark:
            return {"success": False, "error": f"Benchmark with name '{name}' not found"}

        # Use delete by ID
        return delete_benchmark_by_id(benchmark['id'])

    except Exception as e:
        logger.error(f"Error deleting benchmark by name {name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== SANDBOX TASK UTILITIES ====================

def get_sandbox_task_by_checksum(checksum: str) -> Optional[Dict[str, Any]]:
    """Retrieve a sandbox task from the database by checksum (PK)."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_tasks').select('*').eq('checksum', checksum).execute()

        if not response.data:
            return None

        return clean_sandbox_task_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving sandbox task by checksum {checksum}: {e}")
        return None


def get_sandbox_task_by_name(source: str, name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a sandbox task from the database by source and name."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_tasks').select('*').eq('source', source).eq('name', name).execute()

        if not response.data:
            return None

        return clean_sandbox_task_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving sandbox task by name {source}/{name}: {e}")
        return None


def create_sandbox_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new sandbox task in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_tasks').insert(task_data).execute()

        if not response.data:
            raise ValueError("Failed to create sandbox task")

        return clean_sandbox_task_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating sandbox task: {e}")
        raise


def update_sandbox_task(checksum: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing sandbox task in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_tasks').update(task_data).eq('checksum', checksum).execute()

        if not response.data:
            raise ValueError(f"Failed to update sandbox task with checksum {checksum}")

        return clean_sandbox_task_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating sandbox task {checksum}: {e}")
        raise


def register_sandbox_task(
    checksum: str,
    name: str,
    path: str,
    agent_timeout_sec: float,
    verifier_timeout_sec: float,
    source: Optional[str] = None,
    instruction: str = "",
    git_url: Optional[str] = None,
    git_commit_id: Optional[str] = None,
    forced_update: bool = True
) -> Dict[str, Any]:
    """
    Register a sandbox task with minimal auto-filling.

    Only auto-fills: created_at
    All other fields must be provided manually.
    """
    try:
        logger.info(f"Registering sandbox task: {checksum}")

        # Check if task already exists
        existing = get_sandbox_task_by_checksum(checksum)
        if existing and not forced_update:
            logger.info(f"Sandbox task {checksum} already exists")
            return {"success": True, "task": existing, "exists": True}

        # Build task data with minimal auto-filling
        now = datetime.now(timezone.utc)
        task_data = {
            "checksum": checksum,
            "created_at": now.isoformat(),
            "source": source,
            "name": name,
            "instruction": instruction,
            "agent_timeout_sec": agent_timeout_sec,
            "verifier_timeout_sec": verifier_timeout_sec,
            "git_url": git_url,
            "git_commit_id": git_commit_id,
            "path": path
        }

        # Create or update task
        if existing and forced_update:
            logger.info(f"Updating existing sandbox task: {checksum}")
            result = update_sandbox_task(checksum, task_data)
            logger.info(f"Successfully updated sandbox task: {checksum}")
            return {"success": True, "task": result, "updated": True}
        else:
            logger.info(f"Creating sandbox task: {checksum}")
            result = create_sandbox_task(task_data)
            logger.info(f"Successfully registered sandbox task: {checksum}")
            return {"success": True, "task": result}

    except Exception as e:
        logger.error(f"Failed to register sandbox task {checksum}: {e}")
        return {"success": False, "error": str(e)}


def delete_sandbox_task_by_checksum(checksum: str) -> Dict[str, Any]:
    """Delete sandbox task from database by checksum."""
    try:
        client = get_supabase_client(use_admin=True)

        # Check if task exists
        existing = client.table('sandbox_tasks').select('name').eq('checksum', checksum).execute()
        if not existing.data:
            return {"success": False, "error": f"Sandbox task with checksum {checksum} not found"}

        task_name = existing.data[0]['name']

        # Check for FK constraints (benchmark_tasks, trials)
        benchmark_links = client.table('sandbox_benchmark_tasks').select('benchmark_id').eq('task_checksum', checksum).execute()
        if benchmark_links.data:
            return {"success": False, "error": f"Cannot delete task '{task_name}': linked to {len(benchmark_links.data)} benchmarks"}

        trials = client.table('sandbox_trials').select('id').eq('task_checksum', checksum).execute()
        if trials.data:
            return {"success": False, "error": f"Cannot delete task '{task_name}': referenced by {len(trials.data)} trials"}

        # Delete the task
        response = client.table('sandbox_tasks').delete().eq('checksum', checksum).execute()

        if response.data:
            logger.info(f"Successfully deleted sandbox task: {task_name} (checksum: {checksum})")
            return {"success": True, "message": f"Task '{task_name}' deleted successfully", "deleted_checksum": checksum}
        else:
            return {"success": False, "error": f"Failed to delete task with checksum {checksum}"}

    except Exception as e:
        logger.error(f"Error deleting sandbox task by checksum {checksum}: {e}")
        return {"success": False, "error": str(e)}


def delete_sandbox_task_by_name(source: str, name: str) -> Dict[str, Any]:
    """Delete sandbox task from database by source and name."""
    try:
        task = get_sandbox_task_by_name(source, name)
        if not task:
            return {"success": False, "error": f"Sandbox task with source '{source}' and name '{name}' not found"}

        return delete_sandbox_task_by_checksum(task['checksum'])

    except Exception as e:
        logger.error(f"Error deleting sandbox task by name {source}/{name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== SANDBOX BENCHMARK TASK UTILITIES ====================

def link_benchmark_to_task(
    benchmark_id: str,
    task_checksum: str,
    benchmark_name: str,
    benchmark_version_hash: str
) -> Dict[str, Any]:
    """Create a link between a benchmark and a task."""
    try:
        client = get_supabase_client(use_admin=True)

        # Check if link already exists
        existing = client.table('sandbox_benchmark_tasks')\
            .select('*')\
            .eq('benchmark_id', benchmark_id)\
            .eq('task_checksum', task_checksum)\
            .execute()

        if existing.data:
            result = clean_sandbox_benchmark_task_metadata(existing.data[0])
            logger.info(f"Benchmark-task link already exists: {benchmark_id} -> {task_checksum}")
            return {"success": True, "link": result, "exists": True}

        link_data = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_name,
            "benchmark_version_hash": benchmark_version_hash,
            "task_checksum": task_checksum
        }

        response = client.table('sandbox_benchmark_tasks').insert(link_data).execute()

        if not response.data:
            raise ValueError("Failed to create benchmark-task link")

        result = clean_sandbox_benchmark_task_metadata(response.data[0])
        logger.info(f"Successfully linked benchmark {benchmark_id} to task {task_checksum}")
        return {"success": True, "link": result}

    except Exception as e:
        logger.error(f"Error linking benchmark to task: {e}")
        return {"success": False, "error": str(e)}


def unlink_benchmark_from_task(benchmark_id: str, task_checksum: str) -> Dict[str, Any]:
    """Remove the link between a benchmark and a task."""
    try:
        client = get_supabase_client(use_admin=True)

        response = client.table('sandbox_benchmark_tasks').delete().eq('benchmark_id', benchmark_id).eq('task_checksum', task_checksum).execute()

        if response.data:
            logger.info(f"Successfully unlinked benchmark {benchmark_id} from task {task_checksum}")
            return {"success": True, "message": "Benchmark-task link removed successfully"}
        else:
            return {"success": False, "error": "Link not found or already deleted"}

    except Exception as e:
        logger.error(f"Error unlinking benchmark from task: {e}")
        return {"success": False, "error": str(e)}


def delete_all_benchmark_task_links(benchmark_id: str) -> Dict[str, Any]:
    """Delete all task links for a benchmark (cleanup helper)."""
    try:
        client = get_supabase_client(use_admin=True)

        response = client.table('sandbox_benchmark_tasks').delete().eq('benchmark_id', benchmark_id).execute()

        count = len(response.data) if response.data else 0
        logger.info(f"Deleted {count} task links for benchmark {benchmark_id}")
        return {"success": True, "deleted_count": count}

    except Exception as e:
        logger.error(f"Error deleting benchmark task links: {e}")
        return {"success": False, "error": str(e)}


# ==================== SANDBOX JOB UTILITIES ====================

def get_sandbox_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a sandbox job from the database by ID."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_jobs').select('*').eq('id', job_id).execute()

        if not response.data:
            return None

        return clean_sandbox_job_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving sandbox job by ID {job_id}: {e}")
        return None


def get_sandbox_job_by_name(job_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a sandbox job from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_jobs').select('*').eq('job_name', job_name).execute()

        if not response.data:
            return None

        return clean_sandbox_job_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving sandbox job by name {job_name}: {e}")
        return None


def create_sandbox_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new sandbox job in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_jobs').insert(job_data).execute()

        if not response.data:
            raise ValueError("Failed to create sandbox job")

        return clean_sandbox_job_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating sandbox job: {e}")
        raise


def update_sandbox_job(job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing sandbox job in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_jobs').update(job_data).eq('id', job_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update sandbox job with ID {job_id}")

        return clean_sandbox_job_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating sandbox job {job_id}: {e}")
        raise


def register_sandbox_job(
    job_name: str,
    username: str,
    agent_id: str,
    model_id: str,
    benchmark_id: str,
    n_trials: int,
    n_rep_eval: int,
    config: Dict[str, Any],
    job_id: Optional[str] = None,
    git_commit_id: Optional[str] = None,
    package_version: Optional[str] = None,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
    metrics: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
    forced_update: bool = True,
    hf_traces_link: Optional[str] = None,
    job_status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Register a sandbox job with minimal auto-filling.

    Auto-fills: id (if not provided), created_at, started_at (defaults to now if not provided)
    Validates: git_commit_id OR package_version must be provided
    """
    try:
        logger.info(f"Registering sandbox job: {job_name}")

        # Validate version constraint
        if not git_commit_id and not package_version:
            return {"success": False, "error": "Either git_commit_id or package_version must be provided"}

        # Check if job already exists
        existing = get_sandbox_job_by_name(job_name)
        if existing and not forced_update:
            logger.info(f"Sandbox job {job_name} already exists")
            return {"success": True, "job": existing, "exists": True}

        # Build job data with minimal auto-filling
        now = datetime.now(timezone.utc)
        job_data = {
            "created_at": now.isoformat(),
            "job_name": job_name,
            "username": username,
            "started_at": (started_at or now).isoformat(),
            "ended_at": ended_at.isoformat() if ended_at else None,
            "git_commit_id": git_commit_id,
            "package_version": package_version,
            "n_trials": n_trials,
            "config": config,
            "metrics": metrics,
            "stats": stats,
            "agent_id": agent_id,
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "n_rep_eval": n_rep_eval,
            "hf_traces_link": hf_traces_link,
            "job_status": job_status
        }

        # Include job_id if provided (preserves local ID from result.json)
        if job_id:
            job_data["id"] = job_id

        # Create or update job
        if existing and forced_update:
            logger.info(f"Updating existing sandbox job: {job_name}")
            result = update_sandbox_job(existing['id'], job_data)
            logger.info(f"Successfully updated sandbox job: {job_name}")
            return {"success": True, "job": result, "updated": True}
        else:
            logger.info(f"Creating sandbox job: {job_name}")
            result = create_sandbox_job(job_data)
            logger.info(f"Successfully registered sandbox job: {job_name}")
            return {"success": True, "job": result}

    except Exception as e:
        logger.error(f"Failed to register sandbox job {job_name}: {e}")
        return {"success": False, "error": str(e)}


def delete_sandbox_job_by_id(job_id: str) -> Dict[str, Any]:
    """Delete sandbox job from database by ID."""
    try:
        client = get_supabase_client(use_admin=True)

        # Check if job exists
        existing = client.table('sandbox_jobs').select('job_name').eq('id', job_id).execute()
        if not existing.data:
            return {"success": False, "error": f"Sandbox job with ID {job_id} not found"}

        job_name = existing.data[0]['job_name']

        # Check for FK constraints (trials)
        trials = client.table('sandbox_trials').select('id').eq('job_id', job_id).execute()
        if trials.data:
            return {"success": False, "error": f"Cannot delete job '{job_name}': referenced by {len(trials.data)} trials"}

        # Delete the job
        response = client.table('sandbox_jobs').delete().eq('id', job_id).execute()

        if response.data:
            logger.info(f"Successfully deleted sandbox job: {job_name} (ID: {job_id})")
            return {"success": True, "message": f"Job '{job_name}' deleted successfully", "deleted_id": job_id}
        else:
            return {"success": False, "error": f"Failed to delete job with ID {job_id}"}

    except Exception as e:
        logger.error(f"Error deleting sandbox job by ID {job_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_sandbox_job_by_name(job_name: str) -> Dict[str, Any]:
    """Delete sandbox job from database by name."""
    try:
        job = get_sandbox_job_by_name(job_name)
        if not job:
            return {"success": False, "error": f"Sandbox job with name '{job_name}' not found"}

        return delete_sandbox_job_by_id(job['id'])

    except Exception as e:
        logger.error(f"Error deleting sandbox job by name {job_name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== SANDBOX TRIAL UTILITIES ====================

def get_sandbox_trial_by_id(trial_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a sandbox trial from the database by ID."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_trials').select('*').eq('id', trial_id).execute()

        if not response.data:
            return None

        return clean_sandbox_trial_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving sandbox trial by ID {trial_id}: {e}")
        return None


def get_sandbox_trial_by_name(trial_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve a sandbox trial from the database by name."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_trials').select('*').eq('trial_name', trial_name).execute()

        if not response.data:
            return None

        return clean_sandbox_trial_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving sandbox trial by name {trial_name}: {e}")
        return None


def create_sandbox_trial(trial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new sandbox trial in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_trials').insert(trial_data).execute()

        if not response.data:
            raise ValueError("Failed to create sandbox trial")

        return clean_sandbox_trial_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating sandbox trial: {e}")
        raise


def update_sandbox_trial(trial_id: str, trial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing sandbox trial in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_trials').update(trial_data).eq('id', trial_id).execute()

        if not response.data:
            raise ValueError(f"Failed to update sandbox trial with ID {trial_id}")

        return clean_sandbox_trial_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating sandbox trial {trial_id}: {e}")
        raise


def register_sandbox_trial(
    trial_name: str,
    trial_uri: str,
    task_checksum: str,
    config: Dict[str, Any],
    trial_id: Optional[str] = None,
    job_id: Optional[str] = None,
    reward: Optional[float] = None,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
    environment_setup_started_at: Optional[datetime] = None,
    environment_setup_ended_at: Optional[datetime] = None,
    agent_setup_started_at: Optional[datetime] = None,
    agent_setup_ended_at: Optional[datetime] = None,
    agent_execution_started_at: Optional[datetime] = None,
    agent_execution_ended_at: Optional[datetime] = None,
    verifier_started_at: Optional[datetime] = None,
    verifier_ended_at: Optional[datetime] = None,
    exception_info: Optional[Dict[str, Any]] = None,
    forced_update: bool = True
) -> Dict[str, Any]:
    """
    Register a sandbox trial with minimal auto-filling.

    Auto-fills: id (if not provided), created_at
    All timing and execution details are manual.
    """
    try:
        logger.info(f"Registering sandbox trial: {trial_name}")

        # Check if trial already exists
        existing = get_sandbox_trial_by_name(trial_name)
        if existing and not forced_update:
            logger.info(f"Sandbox trial {trial_name} already exists")
            return {"success": True, "trial": existing, "exists": True}

        # Build trial data with minimal auto-filling
        now = datetime.now(timezone.utc)
        trial_data = {
            "trial_name": trial_name,
            "trial_uri": trial_uri,
            "job_id": job_id,
            "task_checksum": task_checksum,
            "reward": reward,
            "started_at": started_at.isoformat() if started_at else None,
            "ended_at": ended_at.isoformat() if ended_at else None,
            "environment_setup_started_at": environment_setup_started_at.isoformat() if environment_setup_started_at else None,
            "environment_setup_ended_at": environment_setup_ended_at.isoformat() if environment_setup_ended_at else None,
            "agent_setup_started_at": agent_setup_started_at.isoformat() if agent_setup_started_at else None,
            "agent_setup_ended_at": agent_setup_ended_at.isoformat() if agent_setup_ended_at else None,
            "agent_execution_started_at": agent_execution_started_at.isoformat() if agent_execution_started_at else None,
            "agent_execution_ended_at": agent_execution_ended_at.isoformat() if agent_execution_ended_at else None,
            "verifier_started_at": verifier_started_at.isoformat() if verifier_started_at else None,
            "verifier_ended_at": verifier_ended_at.isoformat() if verifier_ended_at else None,
            "config": config,
            "exception_info": exception_info,
            "created_at": now.isoformat()
        }

        # Include trial_id if provided (preserves local ID from result.json)
        if trial_id:
            trial_data["id"] = trial_id

        # Create or update trial
        if existing and forced_update:
            logger.info(f"Found existing sandbox trial: {trial_name}, updating...")
            result = update_sandbox_trial(existing['id'], trial_data)
            logger.info(f"Successfully updated sandbox trial: {trial_name}")
            return {"success": True, "trial": result, "updated": True}
        else:
            logger.info(f"Creating sandbox trial: {trial_name}")
            result = create_sandbox_trial(trial_data)
            logger.info(f"Successfully registered sandbox trial: {trial_name}")
            return {"success": True, "trial": result}

    except Exception as e:
        logger.error(f"Failed to register sandbox trial {trial_name}: {e}")
        return {"success": False, "error": str(e)}


def delete_sandbox_trial_by_id(trial_id: str) -> Dict[str, Any]:
    """Delete sandbox trial from database by ID."""
    try:
        client = get_supabase_client(use_admin=True)

        # Check if trial exists
        existing = client.table('sandbox_trials').select('trial_name').eq('id', trial_id).execute()
        if not existing.data:
            return {"success": False, "error": f"Sandbox trial with ID {trial_id} not found"}

        trial_name = existing.data[0]['trial_name']

        # Check for FK constraints (model_usage)
        usage = client.table('sandbox_trial_model_usage').select('trial_id').eq('trial_id', trial_id).execute()
        if usage.data:
            return {"success": False, "error": f"Cannot delete trial '{trial_name}': has {len(usage.data)} model usage records"}

        # Delete the trial
        response = client.table('sandbox_trials').delete().eq('id', trial_id).execute()

        if response.data:
            logger.info(f"Successfully deleted sandbox trial: {trial_name} (ID: {trial_id})")
            return {"success": True, "message": f"Trial '{trial_name}' deleted successfully", "deleted_id": trial_id}
        else:
            return {"success": False, "error": f"Failed to delete trial with ID {trial_id}"}

    except Exception as e:
        logger.error(f"Error deleting sandbox trial by ID {trial_id}: {e}")
        return {"success": False, "error": str(e)}


def delete_sandbox_trial_by_name(trial_name: str) -> Dict[str, Any]:
    """Delete sandbox trial from database by name."""
    try:
        trial = get_sandbox_trial_by_name(trial_name)
        if not trial:
            return {"success": False, "error": f"Sandbox trial with name '{trial_name}' not found"}

        return delete_sandbox_trial_by_id(trial['id'])

    except Exception as e:
        logger.error(f"Error deleting sandbox trial by name {trial_name}: {e}")
        return {"success": False, "error": str(e)}


# ==================== SANDBOX TRIAL MODEL USAGE UTILITIES ====================

def get_trial_model_usage(trial_id: str, model_id: str, model_provider: str) -> Optional[Dict[str, Any]]:
    """Retrieve trial model usage by composite primary key."""
    try:
        client = get_supabase_client()
        response = client.table('sandbox_trial_model_usage').select('*')\
            .eq('trial_id', trial_id)\
            .eq('model_id', model_id)\
            .eq('model_provider', model_provider)\
            .execute()

        if not response.data:
            return None

        return clean_sandbox_trial_model_usage_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving trial model usage for trial {trial_id}, model {model_id}, provider {model_provider}: {e}")
        return None


def create_trial_model_usage(usage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new trial model usage record in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_trial_model_usage').insert(usage_data).execute()

        if not response.data:
            raise ValueError("Failed to create trial model usage")

        return clean_sandbox_trial_model_usage_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error creating trial model usage: {e}")
        raise


def update_trial_model_usage(trial_id: str, model_id: str, model_provider: str, usage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Update an existing trial model usage record in the database."""
    try:
        client = get_supabase_client(use_admin=True)
        response = client.table('sandbox_trial_model_usage').update(usage_data)\
            .eq('trial_id', trial_id)\
            .eq('model_id', model_id)\
            .eq('model_provider', model_provider)\
            .execute()

        if not response.data:
            raise ValueError(f"Failed to update trial model usage for trial {trial_id}, model {model_id}, provider {model_provider}")

        return clean_sandbox_trial_model_usage_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error updating trial model usage for trial {trial_id}, model {model_id}, provider {model_provider}: {e}")
        raise


def register_trial_model_usage(
    trial_id: str,
    model_id: str,
    model_provider: str,
    n_input_tokens: Optional[int] = None,
    n_output_tokens: Optional[int] = None,
    forced_update: bool = True
) -> Dict[str, Any]:
    """
    Register trial model usage with minimal auto-filling.

    Auto-fills: created_at
    All other fields are manual.
    """
    try:
        logger.info(f"Registering trial model usage for trial {trial_id}")

        # Check if usage record already exists
        existing = get_trial_model_usage(trial_id, model_id, model_provider)
        if existing and not forced_update:
            logger.info(f"Trial model usage already exists for trial {trial_id}, model {model_id}, provider {model_provider}")
            return {"success": True, "usage": existing, "exists": True}

        # Build usage data with minimal auto-filling
        now = datetime.now(timezone.utc)
        usage_data = {
            "trial_id": trial_id,
            "created_at": now.isoformat(),
            "model_id": model_id,
            "model_provider": model_provider,
            "n_input_tokens": n_input_tokens,
            "n_output_tokens": n_output_tokens
        }

        # Create or update usage record
        if existing and forced_update:
            logger.info(f"Updating existing trial model usage for trial {trial_id}, model {model_id}, provider {model_provider}")
            result = update_trial_model_usage(trial_id, model_id, model_provider, usage_data)
            logger.info(f"Successfully updated trial model usage")
            return {"success": True, "usage": result, "updated": True}
        else:
            logger.info(f"Creating new trial model usage for trial {trial_id}, model {model_id}, provider {model_provider}")
            result = create_trial_model_usage(usage_data)
            logger.info(f"Successfully registered trial model usage")
            return {"success": True, "usage": result}

    except Exception as e:
        logger.error(f"Failed to register trial model usage: {e}")
        return {"success": False, "error": str(e)}


def delete_trial_model_usage(trial_id: str, model_id: str, model_provider: str) -> Dict[str, Any]:
    """Delete trial model usage by composite PK."""
    try:
        client = get_supabase_client(use_admin=True)

        response = client.table('sandbox_trial_model_usage').delete()\
            .eq('trial_id', trial_id)\
            .eq('model_id', model_id)\
            .eq('model_provider', model_provider)\
            .execute()

        if response.data:
            logger.info(f"Successfully deleted trial model usage for trial {trial_id}")
            return {"success": True, "message": "Trial model usage deleted successfully"}
        else:
            return {"success": False, "error": "Usage record not found or already deleted"}

    except Exception as e:
        logger.error(f"Error deleting trial model usage: {e}")
        return {"success": False, "error": str(e)}


def delete_all_trial_model_usage(trial_id: str) -> Dict[str, Any]:
    """Delete all model usage records for a trial (cleanup helper)."""
    try:
        client = get_supabase_client(use_admin=True)

        response = client.table('sandbox_trial_model_usage').delete().eq('trial_id', trial_id).execute()

        count = len(response.data) if response.data else 0
        logger.info(f"Deleted {count} model usage records for trial {trial_id}")
        return {"success": True, "deleted_count": count}

    except Exception as e:
        logger.error(f"Error deleting trial model usage records: {e}")
        return {"success": False, "error": str(e)}


# ==================== S3 UTILITIES ====================

def _check_bucket_exists(client: Client, bucket_name: str) -> None:
    """
    Verify that S3 bucket exists.

    Args:
        client: Supabase client
        bucket_name: Name of the bucket to check

    Raises:
        ValueError: If bucket does not exist
    """
    try:
        client.storage.get_bucket(bucket_name)
        logger.info(f"Bucket '{bucket_name}' exists")
    except Exception as e:
        raise ValueError(
            f"Bucket '{bucket_name}' not found. "
            f"Please create the bucket before uploading results. "
            f"Error: {e}"
        )


def _upload_file_to_s3(
    client: Client,
    file_path: "Path",
    s3_path: str,
    bucket_name: str
) -> None:
    """
    Upload a single file to S3.

    Args:
        client: Supabase client
        file_path: Local file path
        s3_path: S3 path (e.g., "trial_id/folder/file.json")
        bucket_name: S3 bucket name

    Raises:
        Exception: If upload fails
    """
    try:
        with open(file_path, "rb") as f:
            client.storage.from_(bucket_name).upload(
                file=f,
                path=s3_path,
                file_options={"upsert": "false"}
            )
        logger.debug(f"Uploaded {file_path.name} to {s3_path}")
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to {s3_path}: {e}")
        raise


def _upload_trial_folder_to_s3(
    client: Client,
    trial_dir: "Path",
    trial_id: str,
    job_id: str,
    bucket_name: str
) -> None:
    """
    Upload entire trial directory to S3 recursively.

    Args:
        client: Supabase client
        trial_dir: Path to trial directory
        trial_id: Trial UUID (used as folder name in S3)
        job_id: Job UUID (used as root folder name in S3)
        bucket_name: S3 bucket name

    Raises:
        Exception: If upload fails
    """
    from pathlib import Path

    logger.info(f"Uploading trial folder {trial_dir.name} to S3 bucket {bucket_name}")

    # Walk through directory tree
    for root, dirs, files in os.walk(trial_dir):
        for file in files:
            file_path = Path(root) / file
            # Calculate relative path from trial_dir
            relative_path = file_path.relative_to(trial_dir)
            # Construct S3 path: job_id/trial_id/relative_path
            s3_path = f"{job_id}/{trial_id}/{relative_path}"

            _upload_file_to_s3(client, file_path, s3_path, bucket_name)

    logger.info(f"Successfully uploaded trial folder {trial_dir.name} to S3")


def _get_trial_s3_url(
    client: Client,
    trial_id: str,
    job_id: str,
    bucket_name: str
) -> str:
    """
    Get public URL for trial folder in S3.

    Args:
        client: Supabase client
        trial_id: Trial UUID
        job_id: Job UUID (root folder in S3)
        bucket_name: S3 bucket name

    Returns:
        Public URL to trial folder

    Note:
        This constructs a public URL. If the bucket is private,
        you may need to use create_signed_url instead.
    """
    from .config import supabase_config

    # Extract project ref from Supabase URL
    # URL format: https://<project_ref>.supabase.co
    supabase_url = supabase_config.supabase_url

    # Construct public URL: bucket/job_id/trial_id
    trial_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{job_id}/{trial_id}"

    logger.info(f"Generated S3 URL for trial {trial_id}")
    return trial_url


# ==================== EVAL RESULTS UPLOAD UTILITIES ====================

# Configuration constants
MAX_TRIAL_RETRIES = 3  # Number of retries per trial before giving up

def get_agent_by_name_and_version(
    name: str,
    version: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Retrieve an agent from the database by name and version.

    Args:
        name: Agent name
        version: Agent version (package_version)

    Returns:
        Cleaned agent metadata dict or None if not found
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("agents")
            .select("*")
            .eq("name", name)
            .execute()
        )

        if not response.data:
            return None

        # Package version matching is not currently enforced in the DB schema,
        # so we return the first record for this agent name.
        return clean_agent_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving agent by name {name} and version {version}: {e}")
        return None


def get_benchmark_by_name_and_version(
    name: str,
    version_hash: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a benchmark from the database by name and version hash.

    Args:
        name: Benchmark name
        version_hash: Benchmark version hash (64-char SHA-256)

    Returns:
        Cleaned benchmark metadata dict or None if not found
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("benchmarks")
            .select("*")
            .eq("name", name)
            .eq("benchmark_version_hash", version_hash)
            .execute()
        )

        if not response.data:
            return None

        return clean_benchmark_metadata(response.data[0])
    except Exception as e:
        logger.error(f"Error retrieving benchmark by name {name} and version {version_hash}: {e}")
        return None

def _assert_job_finished(job_dir: Path) -> None:
    """
    Assert that the job has finished execution.

    Args:
        job_dir: Path to job directory

    Raises:
        ValueError: If job is not finished (finished_at is null or missing)
        FileNotFoundError: If result.json doesn't exist
    """
    result_path = job_dir / "result.json"

    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found in {job_dir}")

    result = json.loads(result_path.read_text())

    if not result.get("finished_at"):
        raise ValueError(
            f"Job at {job_dir} is not finished. "
            f"The 'finished_at' field is missing or null. "
            f"Please wait for the job to complete before uploading results."
        )

    logger.info(f"Job {job_dir.name} is finished at {result['finished_at']}")


def _extract_job_metadata(
    job_dir: Path,
    username: str,
    git_commit_id: Optional[str],
    agent_id: UUID,
    model_id: UUID,
    benchmark_id: UUID
) -> Dict[str, Any]:
    """
    Extract job metadata from job directory.
    
    Args:
        job_dir: Path to job directory
        username: Username for job
        git_commit_id: Git commit ID (optional)
        agent_id: Agent UUID (foreign key)
        model_id: Model UUID (foreign key)
        benchmark_id: Benchmark UUID (foreign key)
    
    Returns:
        Dict with job metadata ready for register_sandbox_job()
    """
    config_path = job_dir / "config.json"
    result_path = job_dir / "result.json"
    
    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())
    
    # Extract package_version from first trial's agent_info
    trial_dirs = [d for d in job_dir.iterdir() if d.is_dir()]
    package_version = None
    if trial_dirs:
        first_trial_result = json.loads((trial_dirs[0] / "result.json").read_text())
        package_version = first_trial_result.get("agent_info", {}).get("version")
    
    # Extract metrics from the nested structure
    metrics = None
    if "stats" in result and "evals" in result["stats"]:
        # Get the first (and likely only) evaluation key
        evals = result["stats"]["evals"]
        if evals:
            # Get the first evaluation entry
            eval_key = list(evals.keys())[0]
            eval_data = evals[eval_key]
            
            # Extract metrics array
            metrics_array = eval_data.get("metrics", [])
            
            # If metrics exist, extract the mean value
            if metrics_array and len(metrics_array) > 0:
                # metrics is an array of dicts with 'mean' key
                metrics = {
                    "accuracy": metrics_array[0].get("mean"),
                }
    
    metrics_list = []
    if metrics:
        for key, value in metrics.items():
            metrics_list.append({"name": key, "value": value})
    

    job_metadata = {
        "job_id": result["id"],  # Preserve local job ID from result.json
        "job_name": config["job_name"],
        "username": username,
        "agent_id": str(agent_id),
        "model_id": str(model_id),
        "benchmark_id": str(benchmark_id),
        "n_trials": result["n_total_trials"],
        "n_rep_eval": config.get("n_attempts", 1),
        "config": config,
        "metrics": metrics_list,  # Now properly extracted from nested structure
        "stats": result.get("stats"),
        "git_commit_id": git_commit_id,
        "package_version": package_version,
        "started_at": _parse_datetime(result.get("started_at")),
        "ended_at": _parse_datetime(result.get("finished_at"))
    }
    
    return job_metadata


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO datetime string to datetime object.

    Args:
        dt_str: ISO format datetime string or None

    Returns:
        datetime object or None
    """
    if not dt_str:
        return None

    # Handle both with and without timezone info
    try:
        # Try parsing with timezone
        return datetime.fromisoformat(dt_str)
    except ValueError:
        try:
            # Try parsing without timezone, assume UTC
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            logger.warning(f"Failed to parse datetime: {dt_str}")
            return None


def _extract_trial_metadata(
    trial_dir: Path,
    trial_uri: str,
    trial_id: str,
    job_id: UUID,
    task_checksum: str
) -> Dict[str, Any]:
    """
    Extract trial metadata from trial directory.

    Args:
        trial_dir: Path to trial directory
        trial_uri: S3 URI for trial logs
        trial_id: Trial ID from local result.json (preserved)
        job_id: Job UUID (foreign key)
        task_checksum: Task checksum (foreign key)

    Returns:
        Dict with trial metadata ready for register_sandbox_trial()

    Raises:
        ValueError: If trial has incomplete execution data (no agent_execution or verifier_result)
                    and no exception_info explaining the early failure
    """
    config_path = trial_dir / "config.json"
    result_path = trial_dir / "result.json"

    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())

    # STRICT VALIDATION: Only accept fully completed trials
    # A valid trial MUST have BOTH agent_execution AND verifier_result
    # Exception info is NOT sufficient - we only accept complete runs
    agent_execution = result.get("agent_execution")
    verifier_result = result.get("verifier_result")

    if agent_execution is None:
        raise ValueError(
            f"Trial {result['trial_name']} is incomplete: missing agent_execution data. "
            f"Only fully completed trials are accepted."
        )

    if verifier_result is None:
        raise ValueError(
            f"Trial {result['trial_name']} is incomplete: missing verifier_result data. "
            f"Only fully completed trials are accepted."
        )

    exception_info = result.get("exception_info")

    trial_metadata = {
        "trial_id": trial_id,  # Preserve local trial ID from result.json
        "trial_name": result["trial_name"],
        "trial_uri": trial_uri,
        "task_checksum": task_checksum,
        "config": result["config"],
        "job_id": str(job_id),
        "reward": (verifier_result or {}).get("reward"),
        "exception_info": exception_info,
        "started_at": _parse_datetime(result.get("started_at")),
        "ended_at": _parse_datetime(result.get("finished_at")),
        "environment_setup_started_at": _parse_datetime((result.get("environment_setup") or {}).get("started_at")),
        "environment_setup_ended_at": _parse_datetime((result.get("environment_setup") or {}).get("finished_at")),
        "agent_setup_started_at": _parse_datetime((result.get("agent_setup") or {}).get("started_at")),
        "agent_setup_ended_at": _parse_datetime((result.get("agent_setup") or {}).get("finished_at")),
        "agent_execution_started_at": _parse_datetime((agent_execution or {}).get("started_at")),
        "agent_execution_ended_at": _parse_datetime((agent_execution or {}).get("finished_at")),
        "verifier_started_at": _parse_datetime((verifier_result or {}).get("started_at")),
        "verifier_ended_at": _parse_datetime((verifier_result or {}).get("finished_at")),
    }

    return trial_metadata


def _extract_model_usage(
    trial_dir: Path,
    trial_id: UUID,
    model_id: UUID
) -> Optional[Dict[str, Any]]:
    """
    Extract model usage from trial directory.

    Args:
        trial_dir: Path to trial directory
        trial_id: Trial UUID
        model_id: Model UUID (foreign key)

    Returns:
        Dict with model usage metadata ready for register_trial_model_usage(),
        or None if no agent_result exists (e.g., timeout)
    """
    result_path = trial_dir / "result.json"
    result = json.loads(result_path.read_text())

    agent_result = result.get("agent_result")
    if not agent_result:
        logger.debug(f"No agent_result for trial {trial_id}, skipping model usage")
        return None

    model_provider = result.get("agent_info", {}).get("model_info", {}).get("provider", "unknown")

    usage_metadata = {
        "trial_id": str(trial_id),
        "model_id": str(model_id),
        "model_provider": model_provider,
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_output_tokens": agent_result.get("n_output_tokens")
    }

    return usage_metadata


def _rollback_db_records(
    job_id: str,
    registered_trial_ids: List[str],
    registered_usage_records: List[tuple],
) -> None:
    """
    Rollback database-only changes from a failed upload (no S3 cleanup).

    This ensures atomic all-or-nothing behavior for DB records: if any trial fails,
    all registered data from this upload round is deleted.

    Args:
        job_id: Job ID to delete
        registered_trial_ids: List of trial IDs that were successfully registered in DB
        registered_usage_records: List of (trial_id, model_id, provider) tuples

    Returns:
        None (logs all cleanup steps)
    """
    logger.warning(f"Starting DB-only rollback for job {job_id}...")
    rollback_errors = []

    # Step 1: Delete all trial model usage records
    logger.info(f"Step 1: Deleting {len(registered_usage_records)} model usage records...")
    for trial_id, model_id, provider in registered_usage_records:
        try:
            result = delete_trial_model_usage(trial_id, model_id, provider)
            if not result.get("success"):
                rollback_errors.append(f"Failed to delete usage for trial {trial_id}: {result.get('error')}")
        except Exception as e:
            rollback_errors.append(f"Exception deleting usage for trial {trial_id}: {e}")

    # Step 2: Delete all registered trials
    logger.info(f"Step 2: Deleting {len(registered_trial_ids)} trials from database...")
    for trial_id in registered_trial_ids:
        try:
            result = delete_sandbox_trial_by_id(trial_id)
            if not result.get("success"):
                rollback_errors.append(f"Failed to delete trial {trial_id}: {result.get('error')}")
        except Exception as e:
            rollback_errors.append(f"Exception deleting trial {trial_id}: {e}")

    # Step 3: Delete the job
    logger.info(f"Step 3: Deleting job {job_id} from database...")
    try:
        result = delete_sandbox_job_by_id(job_id)
        if not result.get("success"):
            rollback_errors.append(f"Failed to delete job {job_id}: {result.get('error')}")
    except Exception as e:
        rollback_errors.append(f"Exception deleting job {job_id}: {e}")

    # Report rollback results
    if rollback_errors:
        logger.error(f"DB rollback completed with {len(rollback_errors)} errors:")
        for error in rollback_errors:
            logger.error(f"  - {error}")
    else:
        logger.info(f"DB rollback completed successfully. Deleted {len(registered_trial_ids)} trials and job {job_id}.")


def calculate_standard_error(
    job_dir: Path,
    n_attempts: int,
) -> Optional[float]:
    """
    Calculate standard error of accuracy using Bernoulli variance formula.

    This function treats each trial as a Bernoulli outcome and calculates standard error
    using the variance formula for binomial proportions.

    Args:
        job_dir: Path to job directory containing trial folders
        n_attempts: Deprecated parameter, kept for backward compatibility but not used

    Returns:
        Standard error of accuracy, or None if:
        - No trials found
        - Fewer than 2 valid task groups (tasks with k_i > 1)
        - All tasks have k_i = 1 (no variance can be calculated)

    Algorithm:
        Groups trials by task name and calculates variance using Bernoulli formula:
        1. Group trials by task name (extracted from trial_name__id format)
        2. For each task i:
           - Calculate p_i = mean(rewards for task i)
           - Calculate k_i = number of trials for task i
           - Skip if k_i = 1 (need at least 2 trials per task for variance)
           - Calculate contribution: p_i * (1 - p_i) / (k_i - 1)
        3. Return: (1 / n^2) * sum of all contributions
           where n = number of tasks with k_i > 1

    Note:
        - Trial names follow format: {task_name}__{unique_id}
        - Rewards are read from {trial_dir}/verifier/reward.txt
        - Trials with missing/unparseable rewards are treated as 0 reward
        - No artificial padding or synthetic trials needed
        - Handles unequal numbers of trials per task naturally
    """
    import numpy as np

    # Get all trial directories
    trial_dirs = [d for d in job_dir.iterdir() if d.is_dir() and d.name not in ["config.json", "result.json"]]
    if not trial_dirs:
        logger.warning("No trial directories found, cannot calculate standard error")
        return None

    logger.info(f"Calculating standard error from {len(trial_dirs)} trials using Bernoulli variance formula")

    # Extract rewards and group by task name
    task_trials = {}  # task_name -> list of rewards

    for trial_dir in trial_dirs:
        trial_name = trial_dir.name

        # Extract task name from trial_name (format: task_name__unique_id)
        if "__" in trial_name:
            task_name = trial_name.rsplit("__", 1)[0]
        else:
            # Fallback: use full trial name as task name
            task_name = trial_name

        reward_path = trial_dir / "verifier" / "reward.txt"

        try:
            if not reward_path.exists():
                logger.warning(f"Reward file not found for {trial_name}, using 0 reward")
                reward = 0.0
            else:
                reward_text = reward_path.read_text().strip()
                reward = float(reward_text)
        except Exception as e:
            logger.warning(f"Failed to parse reward for {trial_name}: {e}, using 0 reward")
            reward = 0.0

        if task_name not in task_trials:
            task_trials[task_name] = []

        task_trials[task_name].append(reward)

    logger.info(f"Found {len(task_trials)} unique tasks with {len(trial_dirs)} total trials")

    # Calculate variance contribution for each task
    variance_contributions = []
    skipped_tasks = []

    for task_name, rewards in task_trials.items():
        k_i = len(rewards)

        if k_i < 2:
            logger.info(f"Task '{task_name}': skipping (k_i={k_i}, need k_i >= 2 for variance calculation)")
            skipped_tasks.append((task_name, k_i))
            continue

        # Calculate mean accuracy for this task (Bernoulli parameter p_i)
        p_i = np.mean(rewards)

        # Calculate variance contribution: p_i * (1 - p_i) / (k_i - 1)
        # This is the sample variance of a binomial proportion
        variance_contribution = (p_i * (1.0 - p_i)) / (k_i - 1)
        variance_contributions.append(variance_contribution)

        # logger.info(f"Task '{task_name}': k_i={k_i}, p_i={p_i:.4f}, contribution={variance_contribution:.6f}")

    if skipped_tasks:
        logger.warning(f"Skipped {len(skipped_tasks)} tasks with fewer than 2 trials")
        for task_name, k_i in skipped_tasks:
            logger.info(f"  - '{task_name}': k_i={k_i}")

    if not variance_contributions:
        logger.warning("No valid tasks for variance calculation (all tasks have k_i < 2)")
        return None

    # Calculate standard error using Bernoulli formula
    # stderr = (1 / n^2) * sum(p_i * (1 - p_i) / (k_i - 1))
    # where n = number of valid tasks
    n = len(variance_contributions)
    sum_variance = sum(variance_contributions)
    variance = (1.0 / (n ** 2)) * sum_variance
    stderr = np.sqrt(variance)

    logger.info(f"Standard error calculation:")
    logger.info(f"  Number of valid tasks (n): {n}")
    logger.info(f"  Sum of variance contributions: {sum_variance:.6f}")
    logger.info(f"  variance = (1/{n}^2) * {sum_variance:.6f} = {variance:.6f}")
    logger.info(f"  stderr = sqrt(variance) = {stderr:.6f}")

    return float(stderr)


def _rollback_upload(
    job_id: str,
    uploaded_s3_trial_ids: List[str],
    registered_trial_ids: List[str],
    registered_usage_records: List[tuple],
    client: Client,
    bucket_name: str
) -> None:
    """
    Rollback all database and S3 changes from a failed upload.

    This ensures atomic all-or-nothing behavior: if any trial fails,
    all registered data from this upload round is deleted.

    Args:
        job_id: Job ID to delete
        uploaded_s3_trial_ids: List of trial IDs that were uploaded to S3 (may include trials that failed DB registration)
        registered_trial_ids: List of trial IDs that were successfully registered in DB
        registered_usage_records: List of (trial_id, model_id, provider) tuples
        client: Supabase admin client
        bucket_name: S3 bucket name for file cleanup

    Returns:
        None (logs all cleanup steps)
    """
    logger.warning(f"Starting rollback for job {job_id}...")
    rollback_errors = []

    # Step 1: Delete all trial model usage records
    logger.info(f"Step 1: Deleting {len(registered_usage_records)} model usage records...")
    for trial_id, model_id, provider in registered_usage_records:
        try:
            result = delete_trial_model_usage(trial_id, model_id, provider)
            if not result.get("success"):
                rollback_errors.append(f"Failed to delete usage for trial {trial_id}: {result.get('error')}")
        except Exception as e:
            rollback_errors.append(f"Exception deleting usage for trial {trial_id}: {e}")

    # Step 2: Delete all S3 files for uploaded trials (including trials that failed DB registration)
    logger.info(f"Step 2: Deleting S3 files for {len(uploaded_s3_trial_ids)} trials...")
    for trial_id in uploaded_s3_trial_ids:
        try:
            # Recursively collect all files in trial folder
            files_to_delete = []

            def collect_files(prefix):
                """Recursively collect all file paths under a prefix."""
                items = client.storage.from_(bucket_name).list(prefix)
                for item in items:
                    item_path = f"{prefix}/{item['name']}" if prefix else item['name']
                    # Check if it's a folder (Supabase returns folders with metadata)
                    if item.get('id') is None:  # Folder
                        collect_files(item_path)
                    else:  # File
                        files_to_delete.append(item_path)

            # Start collection from job_id/trial_id folder
            collect_files(f"{job_id}/{trial_id}")

            # Delete all collected files
            if files_to_delete:
                client.storage.from_(bucket_name).remove(files_to_delete)
                logger.debug(f"  Deleted {len(files_to_delete)} files for trial {trial_id}")
        except Exception as e:
            rollback_errors.append(f"Exception deleting S3 files for trial {trial_id}: {e}")

    # Step 3: Delete all registered trials
    logger.info(f"Step 3: Deleting {len(registered_trial_ids)} trials from database...")
    for trial_id in registered_trial_ids:
        try:
            result = delete_sandbox_trial_by_id(trial_id)
            if not result.get("success"):
                rollback_errors.append(f"Failed to delete trial {trial_id}: {result.get('error')}")
        except Exception as e:
            rollback_errors.append(f"Exception deleting trial {trial_id}: {e}")

    # Step 4: Delete the job
    logger.info(f"Step 4: Deleting job {job_id} from database...")
    try:
        result = delete_sandbox_job_by_id(job_id)
        if not result.get("success"):
            rollback_errors.append(f"Failed to delete job {job_id}: {result.get('error')}")
    except Exception as e:
        rollback_errors.append(f"Exception deleting job {job_id}: {e}")

    # Report rollback results
    if rollback_errors:
        logger.error(f"Rollback completed with {len(rollback_errors)} errors:")
        for error in rollback_errors:
            logger.error(f"  - {error}")
    else:
        logger.info(f"Rollback completed successfully. Deleted {len(uploaded_s3_trial_ids)} S3 folders, {len(registered_trial_ids)} DB trials, and job {job_id}.")


def register_benchmark_and_tasks_from_job(
    job_dir: Union[str, Path],
    benchmark_name: str,
    benchmark_version_hash: str
) -> Dict[str, Any]:
    """
    Register a benchmark and all its tasks from a job directory.

    This helper function:
    1. Registers the benchmark if it doesn't exist (using GET to check, not register with forced_update)
    2. Extracts task metadata from all trial result.json files
    3. Deduplicates tasks by checksum (important for n_attempts > 1)
    4. Registers tasks that don't exist
    5. Links all tasks to the benchmark

    Args:
        job_dir: Path to job directory containing trial folders
        benchmark_name: Name of the benchmark to register
        benchmark_version_hash: SHA-256 hash of the benchmark version

    Returns:
        Dict with registration summary:
        {
            "success": bool,
            "benchmark": dict,
            "benchmark_registered": bool,
            "tasks_total": int,
            "tasks_registered": int,
            "tasks_existing": int,
            "links_created": int,
            "links_existing": int
        }
    """
    try:
        job_dir = Path(job_dir)
        logger.info(f"Auto-registering benchmark and tasks from {job_dir}")

        # Step 1: Check if benchmark exists (use GET, not register)
        # benchmark = get_benchmark_by_name_and_version(benchmark_name, benchmark_version_hash)
        benchmark = get_benchmark_by_name(benchmark_name)
        benchmark_registered = False

        if not benchmark:
            logger.info(f"Benchmark '{benchmark_name}' not found, registering...")
            result = register_benchmark(
                name=benchmark_name,
                benchmark_version_hash=benchmark_version_hash,
                is_external=False,
                forced_update=False  # Don't update if exists
            )
            if not result.get("success"):
                return {"success": False, "error": f"Failed to register benchmark: {result.get('error')}"}

            benchmark = result["benchmark"]
            benchmark_registered = True
            logger.info(f"Successfully registered benchmark: {benchmark_name}")
        else:
            logger.info(f"Benchmark '{benchmark_name}' already exists")

        benchmark_id = benchmark["id"]

        # Step 2: Extract and deduplicate tasks from trial directories
        trial_dirs = [d for d in job_dir.iterdir() if d.is_dir()]
        if not trial_dirs:
            return {
                "success": True,
                "benchmark": benchmark,
                "benchmark_registered": benchmark_registered,
                "tasks_total": 0,
                "tasks_registered": 0,
                "tasks_existing": 0,
                "links_created": 0,
                "links_existing": 0
            }

        # Track processed task checksums to avoid duplicates (n_attempts > 1)
        processed_task_checksums = set()
        tasks_registered = 0
        tasks_existing = 0
        links_created = 0
        links_existing = 0

        for trial_dir in trial_dirs:
            try:
                result_path = trial_dir / "result.json"
                if not result_path.exists():
                    logger.warning(f"No result.json in {trial_dir.name}, skipping")
                    continue

                result = json.loads(result_path.read_text())
                task_checksum = result.get("task_checksum")

                if not task_checksum:
                    logger.warning(f"No task_checksum in {trial_dir.name}, skipping")
                    continue

                # Skip if we've already processed this task
                if task_checksum in processed_task_checksums:
                    continue

                processed_task_checksums.add(task_checksum)

                # Extract task metadata
                task_name = result.get("task_name", "")
                task_path = result.get("task_id", {}).get("path", "")

                if not task_name or not task_path:
                    logger.warning(f"Missing task_name or path in {trial_dir.name}, skipping")
                    continue

                # Check if task exists (use GET, not register)
                existing_task = get_sandbox_task_by_checksum(task_checksum)

                if not existing_task:
                    # Register new task with placeholder values for missing fields
                    logger.info(f"Registering task: {task_name} ({task_checksum[:8]}...)")
                    task_result = register_sandbox_task(
                        checksum=task_checksum,
                        name=task_name,
                        path=task_path,
                        agent_timeout_sec=3600,  # Default: 1 hour
                        verifier_timeout_sec=600,  # Default: 10 minutes
                        source=benchmark_name,  # Use benchmark name as source
                        instruction="",  # Empty instruction (has default in schema)
                        forced_update=False  # Don't update if exists
                    )
                    if not task_result.get("success"):
                        logger.error(f"Failed to register task {task_name}: {task_result.get('error')}")
                        continue

                    tasks_registered += 1
                else:
                    logger.debug(f"Task {task_name} already exists")
                    tasks_existing += 1

                # Link task to benchmark (idempotent)
                link_result = link_benchmark_to_task(
                    benchmark_id=benchmark_id,
                    task_checksum=task_checksum,
                    benchmark_name=benchmark_name,
                    benchmark_version_hash=benchmark_version_hash
                )

                if link_result.get("success"):
                    if link_result.get("exists"):
                        links_existing += 1
                    else:
                        links_created += 1
                else:
                    logger.error(f"Failed to link task {task_name} to benchmark: {link_result.get('error')}")

            except Exception as e:
                logger.error(f"Error processing trial {trial_dir.name}: {e}")
                continue

        tasks_total = len(processed_task_checksums)
        logger.info(
            f"Benchmark registration complete: {tasks_total} unique tasks "
            f"({tasks_registered} new, {tasks_existing} existing), "
            f"{links_created} new links, {links_existing} existing links"
        )

        return {
            "success": True,
            "benchmark": benchmark,
            "benchmark_registered": benchmark_registered,
            "tasks_total": tasks_total,
            "tasks_registered": tasks_registered,
            "tasks_existing": tasks_existing,
            "links_created": links_created,
            "links_existing": links_existing
        }

    except Exception as e:
        logger.error(f"Failed to register benchmark and tasks: {e}")
        return {"success": False, "error": str(e)}


def _hf_run_summary(model_name: str) -> Optional[Dict[str, Any]]:
    """Fetch run_summary.json from HF (model repo) and return dict with nulls -> None."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        logger.error(f"huggingface_hub not available: {e}")
        return None

    try:
        p = hf_hub_download(repo_id=model_name, filename="run_summary.json", repo_type="model")
    except Exception:
        try:
            p = hf_hub_download(repo_id=model_name, filename="run_summary.json", repo_type="model", revision="main")
        except Exception as e2:
            logger.error(f"Failed to download run_summary.json for {model_name}: {e2}")
            return None

    try:
        data = json.loads(Path(p).read_text())
    except Exception as e3:
        logger.error(f"Failed to parse run_summary.json for {model_name}: {e3}")
        return None

    # normalize null-ish values -> None
    def norm(v):
        if isinstance(v, dict):  return {k: norm(vv) for k, vv in v.items()}
        if isinstance(v, list):  return [norm(vv) for vv in v]
        if v is None:            return None
        if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
            return None
        return v

    data = norm(data)
    if not data.get("name"):
        data["name"] = model_name  # ensure DB model name set
    return data


def upload_job_and_trial_records(
    job_dir: Union[str, Path],
    username: str,
    agent_name: Optional[str] = None,
    agent_version: Optional[str] = None,
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    benchmark_version_hash: Optional[str] = None,
    git_commit_id: Optional[str] = None,
    error_mode: str = "rollback_on_error",
    register_benchmark: bool = False,
    hf_dataset_url: Optional[str] = None,
    forced_update: bool = False,
) -> Dict[str, Any]:
    """
    Upload job and trial records to database (with optional HF dataset URL for trials).

    This function handles DB registration in the correct FK dependency order:
    1. Look up foreign keys (agent, model, benchmark)
    2. Update existing job record OR create new one (if no existing job found)
    3. Register trial records (with HF dataset URL if provided)
    4. Register trial model usage records

    Args:
        job_dir: Path to job directory containing config.json, result.json, and trial folders
        username: Username for job registration
        agent_name: Agent name (auto-detected from trial if not provided)
        agent_version: Agent version (auto-detected from trial if not provided)
        model_name: Model name (auto-detected from trial if not provided)
        benchmark_name: Benchmark name (auto-detected from dataset path if not provided)
        benchmark_version_hash: Benchmark version hash (auto-detected from dataset path if not provided)
        git_commit_id: Git commit ID (optional, uses package_version if not provided)
        error_mode: Error handling mode:
            - "rollback_on_error": Delete all job/trial/usage records on any error (atomic)
            - "skip_on_error": Continue uploading even if individual trials fail (best-effort)
        register_benchmark: If True and benchmark not found, auto-register benchmark and tasks from job (default False)
        hf_dataset_url: HuggingFace dataset URL for trial traces (optional). If provided, trial_uri will use this URL.
        forced_update: If True, allow updating existing records (default: False)

    Returns:
        Dict with summary (fields depend on error_mode):

        rollback_on_error mode:
        {
            "success": bool,  # False if any trial fails
            "job_id": UUID or None,  # None if rolled back
            "n_trials_uploaded": int,  # 0 if rolled back
            "error": str,  # Error message if failed
            "failed_trial": Dict,  # Details of failed trial
            ...
        }

        skip_on_error mode:
        {
            "success": bool,  # True if ANY trials uploaded successfully
            "job_id": UUID,  # Job always kept in DB
            "n_trials_total": int,
            "n_trials_uploaded": int,
            "n_trials_failed": int,
            "failed_trials": List[Dict],  # List of failed trial details
            "job": Dict,
            "trials": List[Dict]  # Successfully uploaded trials
        }

    Raises:
        ValueError: If job not finished, required FKs not found, or invalid error_mode
        FileNotFoundError: If job directory or config files not found
    """
    # Validate error_mode
    if error_mode not in ("rollback_on_error", "skip_on_error"):
        raise ValueError(f"Invalid error_mode: {error_mode}. Must be 'rollback_on_error' or 'skip_on_error'")

    # Convert to Path
    job_dir = Path(job_dir)
    logger.info(f"Starting DB-only upload of evaluation results from {job_dir} (error_mode={error_mode})")

    # Step 1: Validate job is finished
    logger.info("Step 1: Validating job is finished")
    _assert_job_finished(job_dir)

    # Step 2: Load job metadata
    logger.info("Step 2: Loading job metadata")
    config_path = job_dir / "config.json"
    result_path = job_dir / "result.json"

    job_config = json.loads(config_path.read_text())
    job_result = json.loads(result_path.read_text())
    job_id = job_result["id"]

    # Step 3: Auto-detect and lookup foreign keys
    logger.info("Step 3: Auto-detecting and looking up foreign keys")

    # Get list of trial directories
    trial_dirs = [d for d in job_dir.iterdir() if d.is_dir()]
    if not trial_dirs:
        raise ValueError(f"No trial directories found in {job_dir}")

    # Read first trial to auto-detect metadata
    first_trial_result = json.loads((trial_dirs[0] / "result.json").read_text())
    first_trial_config = json.loads((trial_dirs[0] / "config.json").read_text())

    # Auto-detect agent name and version
    if not agent_name:
        agent_name = first_trial_result.get("agent_info", {}).get("name")
        if not agent_name:
            raise ValueError("agent_name not provided and could not be auto-detected from trial")
        logger.info(f"Auto-detected agent_name: {agent_name}")

    if not agent_version:
        agent_version = first_trial_result.get("agent_info", {}).get("version")
        if not agent_version:
            raise ValueError("agent_version not provided and could not be auto-detected from trial")
        logger.info(f"Auto-detected agent_version: {agent_version}")

    # Auto-detect model name
    if not model_name:
        model_name = first_trial_config.get("agent", {}).get("model_name")
        if not model_name:
            raise ValueError("model_name not provided and could not be auto-detected from trial config")
        logger.info(f"Auto-detected model_name: {model_name}")

    # Auto-detect benchmark name and version from dataset path
    if not benchmark_name or not benchmark_version_hash:
        dataset_path = job_config.get("datasets", [{}])[0].get("path", "")
        if dataset_path:
            # Extract benchmark name from path
            path_parts = dataset_path.split("/")
            for part in path_parts:
                if "datasets--" in part:
                    benchmark_name = part.split("--")[-1]
                    logger.info(f"Auto-detected benchmark_name: {benchmark_name}")
                    break

            # Extract version hash from snapshots path
            if "snapshots/" in dataset_path:
                snapshot_part = dataset_path.split("snapshots/")[1]
                raw_hash = snapshot_part.strip("/").split("/")[0]
                # If it's a 40-char git hash, convert to SHA-256 for consistency
                if len(raw_hash) == 40:
                    import hashlib
                    benchmark_version_hash = hashlib.sha256(raw_hash.encode()).hexdigest()
                    logger.info(f"Auto-detected git hash {raw_hash}, converted to SHA-256: {benchmark_version_hash}")
                else:
                    benchmark_version_hash = raw_hash
                    logger.info(f"Auto-detected benchmark_version_hash: {benchmark_version_hash}")

        if not benchmark_name:
            raise ValueError("benchmark_name not provided and could not be auto-detected from dataset path")
        if not benchmark_version_hash:
            raise ValueError("benchmark_version_hash not provided and could not be auto-detected from dataset path")

    # Lookup foreign keys in database
    logger.info("Looking up agent in database...")
    agent = get_agent_by_name(agent_name)
    if not agent:
        raise ValueError(f"Agent not found in database: {agent_name}. Please register the agent first.")
    agent_id = agent["id"]
    logger.info(f"Found agent: {agent_id}")

    logger.info("Looking up model in database...")
    model = get_model_by_name(model_name)

    if not model:
        # If it has the hosted_vllm prefix, try the stripped name first
        hf_name = model_name
        if model_name.startswith("hosted_vllm/"):
            hf_name = model_name.split("hosted_vllm/", 1)[1]
            logger.info(f"Model '{model_name}' not found, trying stripped name: {hf_name}")
            model = get_model_by_name(hf_name)

        # If still missing, attempt auto-register from HF run_summary.json
        if not model:
            summary = _hf_run_summary(hf_name)
            print("summary", summary, "hf_name", hf_name)
            if summary:
                uploaded_model = register_trained_model(summary, forced_update=forced_update)
                if not uploaded_model or uploaded_model['success'] == False:
                    raise ValueError(f"Could not register model in database: {uploaded_model}")
            # Re-lookup after attempted registration
            model = get_model_by_name(hf_name)
            if model:
                # Use the HF repo-style name going forward
                model_name = hf_name

    if not model:
        raise ValueError(f"Model not found in database: {model_name}. Please register the model first.")

    model_id = model["id"]
    logger.info(f"Found model: {model_id}")

    logger.info("Looking up benchmark in database...")
    benchmark = get_benchmark_by_name(benchmark_name)

    # If register_benchmark=True and benchmark not found, auto-register it
    if register_benchmark and not benchmark:
        logger.info(f"Benchmark not found. Auto-registering benchmark and tasks from job...")
        reg_result = register_benchmark_and_tasks_from_job(
            job_dir=job_dir,
            benchmark_name=benchmark_name,
            benchmark_version_hash=benchmark_version_hash
        )
        if not reg_result.get("success"):
            raise ValueError(f"Failed to register benchmark: {reg_result.get('error')}")

        logger.info(
            f"Successfully registered benchmark with {reg_result['tasks_total']} unique tasks "
            f"({reg_result['tasks_registered']} new, {reg_result['tasks_existing']} existing)"
        )

        # Re-lookup benchmark (should succeed now)
        benchmark = get_benchmark_by_name(benchmark_name)
        if not benchmark:
            raise ValueError(
                f"Benchmark registration succeeded but lookup failed. "
                f"benchmark_name: {benchmark_name}"
            )
    elif not benchmark:
        raise ValueError(
            f"Benchmark not found in database: {benchmark_name} (version {benchmark_version_hash}). "
            f"Please register the benchmark first or use register_benchmark=True."
        )

    benchmark_id = benchmark["id"]
    logger.info(f"Found benchmark: {benchmark_id}")

    # Step 4: Check for existing job (created by sbatch script)
    logger.info("Step 4: Checking for existing job entry")
    meta_env_path = job_dir / "meta.env"
    existing_job_id = None

    if meta_env_path.exists():
        # Read DB_JOB_ID from meta.env
        for line in meta_env_path.read_text().splitlines():
            if line.startswith("DB_JOB_ID="):
                existing_job_id = line.split("=", 1)[1].strip()
                logger.info(f"Found existing job ID in meta.env: {existing_job_id}")
                break

    # Extract job metadata for both update and create scenarios
    job_metadata = _extract_job_metadata(
        job_dir, username, git_commit_id, agent_id, model_id, benchmark_id
    )

    if existing_job_id:
        # UPDATE existing job row (created by sbatch)
        logger.info(f"Updating existing job {existing_job_id} with final results")
        
        try:
            client = get_supabase_client()
            
            # Prepare update payload with final job data
            update_data = {
                "ended_at": job_metadata["ended_at"].isoformat() if job_metadata.get("ended_at") else None,
                "git_commit_id": job_metadata.get("git_commit_id"),
                "package_version": job_metadata.get("package_version"),
                "metrics": job_metadata.get("metrics"),
                "stats": job_metadata.get("stats"),
                "hf_traces_link": hf_dataset_url,
                "job_status": "Finished",
                "config": job_metadata.get("config"),
                "n_trials": job_metadata.get("n_trials"),
                "n_rep_eval": job_metadata.get("n_rep_eval"),
            }
            
            # Remove None values to avoid overwriting with null
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            resp = client.table('sandbox_jobs').update(update_data).eq('id', existing_job_id).execute()
            
            if not resp.data:
                raise Exception(f"Failed to update job {existing_job_id}")
            
            db_job_id = existing_job_id
            job_record = {"success": True, "job": resp.data[0], "updated": True}
            logger.info(f"Job {db_job_id} updated successfully")
            
        except Exception as e:
            raise Exception(f"Job update failed: {e}")
    
    else:
        # CREATE new job (fallback for manual/legacy uploads without sbatch pre-creation)
        logger.info("No existing job found in meta.env, creating new job (legacy mode)")
        
        job_metadata["hf_traces_link"] = hf_dataset_url
        job_metadata["job_status"] = "Finished"
        
        job_record = register_sandbox_job(**job_metadata, forced_update=forced_update)
        
        if not job_record.get("success"):
            raise Exception(f"Job registration failed: {job_record.get('error')}")
        
        # Get the database-generated job_id
        db_job_id = job_record.get("job", {}).get("id")
        if not db_job_id:
            raise Exception("Job registration succeeded but no job_id was returned")
        logger.info(f"Job created with ID: {db_job_id}")

    # Step 5: Process each trial (DB registration only, no S3)
    logger.info(f"Step 5: Processing {len(trial_dirs)} trials (DB records only)")
    trials_uploaded = []
    trials_failed = []
    registered_trial_ids = []
    registered_usage_records = []

    for i, trial_dir in enumerate(trial_dirs, 1):
        trial_name = trial_dir.name
        logger.info(f"Processing trial {i}/{len(trial_dirs)}: {trial_name}")

        try:
            # Load trial result to get trial_id and task_checksum
            trial_result = json.loads((trial_dir / "result.json").read_text())
            trial_id = trial_result["id"]
            task_checksum = trial_result["task_checksum"]

            # Determine trial URI based on whether HF dataset URL is provided
            if hf_dataset_url:
                # Use HF dataset URL as base for trial URI
                trial_uri = hf_dataset_url
            else:
                # Use placeholder URI for DB-only mode
                trial_uri = None

            # Register trial
            trial_metadata = _extract_trial_metadata(trial_dir, trial_uri, trial_id, db_job_id, task_checksum)
            trial_record = register_sandbox_trial(**trial_metadata, forced_update=forced_update)

            if not trial_record.get("success"):
                raise Exception(f"Trial registration failed: {trial_record.get('error')}")

            # Track registered trial for potential rollback
            registered_trial_ids.append(trial_id)

            # Register model usage (if available)
            usage_metadata = _extract_model_usage(trial_dir, trial_id, model_id)
            if usage_metadata:
                usage_record = register_trial_model_usage(**usage_metadata, forced_update=forced_update)
                if not usage_record.get("success"):
                    raise Exception(f"Model usage registration failed: {usage_record.get('error')}")

                # Track registered usage for potential rollback
                usage_key = (usage_metadata["trial_id"], usage_metadata["model_id"], usage_metadata["model_provider"])
                registered_usage_records.append(usage_key)

            trials_uploaded.append(trial_record.get("trial"))
            logger.info(f"  Trial {trial_name} completed successfully")

        except Exception as e:
            logger.error(f"  Failed to process trial {trial_name}: {e}")

            if error_mode == "skip_on_error":
                # Skip mode: Log error and continue to next trial
                logger.warning(f"  Skipping failed trial {trial_name} (error_mode=skip_on_error)")
                trials_failed.append({
                    "trial_name": trial_name,
                    "trial_number": i,
                    "error": str(e),
                })
            else:
                # Rollback mode: Trigger rollback and return immediately
                logger.error(f"Triggering immediate rollback due to trial failure...")
                _rollback_db_records(db_job_id, registered_trial_ids, registered_usage_records)

                return {
                    "success": False,
                    "error": f"Upload failed on trial {i}/{len(trial_dirs)}: {str(e)}",
                    "job_id": None,
                    "job_name": job_config["job_name"],
                    "n_trials_total": len(trial_dirs),
                    "n_trials_processed": i,
                    "n_trials_uploaded": 0,  # All rolled back
                    "failed_trial": {
                        "trial_name": trial_name,
                        "trial_number": i,
                        "error": str(e),
                    }
                }

    # Step 6: Calculate and upload standard error if n_attempts > 1
    n_attempts = job_config.get("n_attempts", 1)
    if n_attempts > 1 and len(trials_uploaded) > 0:
        logger.info(f"Step 6: Calculating standard error for {n_attempts} repeated evaluation runs")
        try:
            stderr = calculate_standard_error(job_dir, n_attempts)
            if stderr is not None:
                # Update job record with standard error in metrics
                logger.info(f"Updating job record with accuracy_stderr = {stderr:.6f}")

                # Get current metrics from job record
                current_metrics = job_record.get("job", {}).get("metrics", []) or []

                # Add standard error to metrics
                updated_metrics = current_metrics + [{"name": "accuracy_stderr", "value": stderr}]

                # Update job record in database
                admin_client = get_admin_client()
                update_result = admin_client.table("sandbox_jobs").update({
                    "metrics": updated_metrics
                }).eq("id", db_job_id).execute()

                if update_result:
                    logger.info(f"Successfully updated job record with standard error")
                    # Update the job record in our summary
                    job_record["job"]["metrics"] = updated_metrics
                else:
                    logger.warning(f"Failed to update job record with standard error")
        except Exception as e:
            logger.warning(f"Failed to calculate or upload standard error: {e}")
            # Don't fail the entire upload if stderr calculation fails

    # Step 7: Write result_with_std_error.json if upload succeeded
    try:
        result_path = job_dir / "result.json"
        result_with_stderr_path = job_dir / "result_with_std_error.json"

        if result_path.exists():
            # Read original result.json
            result_data = json.loads(result_path.read_text())

            # Get final metrics from job record (includes std error if calculated)
            final_metrics = job_record.get("job", {}).get("metrics", []) or []

            # Update metrics in result data
            result_data["metrics"] = final_metrics

            # Write to new file
            result_with_stderr_path.write_text(json.dumps(result_data, indent=2))
            logger.info(f"Created result_with_std_error.json with updated metrics")
    except Exception as e:
        logger.warning(f"Failed to create result_with_std_error.json: {e}")
        # Don't fail the entire upload if file creation fails

    # Step 8: Return success summary
    summary = {
        "success": True,
        "job_id": db_job_id,
        "job_name": job_config["job_name"],
        "job_updated": job_record.get("updated", False),  # True if updated existing, False if created new
        "n_trials_total": len(trial_dirs),
        "n_trials_uploaded": len(trials_uploaded),
        "n_trials_failed": len(trials_failed),
        "failed_trials": trials_failed,
        "job": job_record.get("job"),
        "trials": trials_uploaded
    }

    if trials_failed:
        logger.warning(f"Upload completed with failures: {len(trials_uploaded)}/{len(trial_dirs)} trials uploaded, {len(trials_failed)} failed")
        logger.warning(f"Failed trials: {[trial['trial_name'] for trial in trials_failed]}")
    else:
        logger.info(f"Upload completed successfully: {len(trials_uploaded)}/{len(trial_dirs)} trials uploaded")
    
    return summary


def upload_traces_to_hf(
    job_dir: Union[str, Path],
    hf_repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    episodes: str = "last",
    success_filter: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Upload job trial execution traces to HuggingFace Hub as a conversation dataset.

    This function extracts episode-level conversation data from trial directories and
    uploads them to HuggingFace Hub as a structured Dataset. Each row represents one
    episode with conversation messages, agent metadata, and task information.

    Dataset Schema (per row):
        - conversations: list[{"role": str, "content": str}] - OpenAI format messages
        - agent: str - Agent name
        - model: str - Model name
        - model_provider: str - Provider ID (e.g., "hosted_vllm", "openai")
        - date: str - ISO timestamp of trial start
        - task: str - Task name
        - episode: str - Episode directory name (e.g., "episode-0")
        - run_id: str - Job/run identifier
        - trial_name: str - Trial name

    Args:
        job_dir: Path to job directory containing trial subdirectories
        hf_repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        private: Whether to create a private repository (default: False)
        token: HuggingFace API token for authentication (default: None, uses HF_TOKEN env var)
        episodes: "all" or "last" - which episodes to export per trial (default: "all")
        success_filter: Filter trials by success ("success", "failure", or None)
        verbose: Enable verbose logging (default: False)

    Returns:
        str: HuggingFace dataset URL (e.g., 'https://huggingface.co/datasets/username/dataset-name')

    Raises:
        ValueError: If job_dir is invalid or hf_repo_id is malformed
        FileNotFoundError: If job_dir doesn't exist
        RuntimeError: If required dependencies are not available

    Example:
        >>> url = upload_traces_to_hf(
        ...     job_dir="jobs/2024-01-15_eval",
        ...     hf_repo_id="myorg/eval-traces-2024",
        ...     private=True,
        ...     token=os.getenv("HF_TOKEN"),
        ...     episodes="last",
        ...     success_filter="success"
        ... )
        >>> print(url)
        https://huggingface.co/datasets/myorg/eval-traces-2024
    """
    # Validate inputs
    if not job_dir or not isinstance(job_dir, (str, Path)):
        raise ValueError("job_dir must be a non-empty string or Path")

    if not hf_repo_id or not isinstance(hf_repo_id, str):
        raise ValueError("hf_repo_id must be a non-empty string")

    # Validate job directory
    job_dir = Path(job_dir)
    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")

    if not job_dir.is_dir():
        raise ValueError(f"Job path must be a directory: {job_dir}")

    # Validate repository ID format
    if "/" not in hf_repo_id or len(hf_repo_id.split("/")) != 2:
        raise ValueError("hf_repo_id must be in format 'username/dataset-name'")

    # Validate episodes parameter
    if episodes not in ("all", "last"):
        raise ValueError("episodes must be either 'all' or 'last'")

    # Validate success_filter parameter
    if success_filter is not None and success_filter not in ("success", "failure"):
        raise ValueError("success_filter must be None, 'success', or 'failure'")

    # Check dependencies
    if Dataset is None:
        raise RuntimeError(
            "datasets library is required for trace export. Please install it: pip install datasets"
        )

    if HfApi is None or create_repo is None:
        raise RuntimeError(
            "huggingface_hub is required for HF upload. Please install it: pip install huggingface_hub"
        )

    logger.info(f"Exporting trial traces to HuggingFace: {hf_repo_id}")
    logger.info(f"Job directory: {job_dir}")
    logger.info(f"Private repository: {private}")
    logger.info(f"Episodes: {episodes}, Success filter: {success_filter}")

    # Step 1: Export traces as HuggingFace Dataset
    logger.info("Extracting conversation traces from trial directories...")
    try:
        dataset = export_traces(
            root=job_dir,
            recursive=True,
            episodes=episodes,
            to_sharegpt=False,  # Keep OpenAI format by default
            repo_id=None,  # Don't push yet, we'll do it manually
            push=False,
            verbose=verbose,
            success_filter=success_filter,
        )
        logger.info(f"Extracted {len(dataset)} conversation rows from trials")
    except Exception as e:
        logger.error(f"Failed to extract traces: {e}")
        raise

    # Step 2: Create HuggingFace repository
    logger.info(f"Creating HuggingFace repository: {hf_repo_id}")
    try:
        create_repo(
            repo_id=hf_repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True
        )
        logger.info(f"Repository {hf_repo_id} created or already exists")
    except Exception as e:
        logger.error(f"Failed to create HF repository: {e}")
        raise

    # Step 3: Push dataset to HuggingFace Hub
    logger.info("Uploading dataset to HuggingFace Hub...")
    try:
        dataset.push_to_hub(hf_repo_id, token=token)
        logger.info(f"Upload complete")
    except Exception as e:
        logger.error(f"Failed to upload to HuggingFace: {e}")
        raise

    # Step 4: Return dataset URL
    repo_url = f"https://huggingface.co/datasets/{hf_repo_id}"
    logger.info(f"Successfully uploaded traces to: {repo_url}")

    return repo_url


def create_job_entry_started(
    model_hf_name: str,
    benchmark_hf_name: str,
    job_name: str,
    username: str,
    slurm_job_id: str,
    agent_name: str,
    config: Dict[str, Any],
    n_trials: int,
    n_rep_eval: int,
    harbor_package_version: Optional[str] = "-1.0.0"
) -> Dict[str, Any]:
    """
    Create initial sandbox_jobs entry with status='Started'.

    Returns:
        {"success": bool, "job": dict, "error": str (optional)}
    """
    try:
        # Lookup foreign keys
        agent = get_agent_by_name_and_version(agent_name, version=None)
        if not agent:
            # Get latest agent by name
            client = get_supabase_client()
            resp = client.table('agents').select('*').eq('name', agent_name).limit(1).execute()
            if not resp.data:
                return {"success": False, "error": f"Agent '{agent_name}' not found"}
            agent = resp.data[0]

        model = get_model_by_name(model_hf_name)
        if not model:
            return {"success": False, "error": f"Model '{model_hf_name}' not found"}

        # Extract benchmark name from dataset_hf (e.g., "DCAgent/dev_set_71_tasks" -> "dev_set_71_tasks")
        benchmark_name = benchmark_hf_name.split("/")[-1]
        benchmark = get_benchmark_by_name(benchmark_name)
        if not benchmark:
            return {"success": False, "error": f"Benchmark '{benchmark_name}' not found"}

        # Create job row
        client = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()

        job_data = {
            "job_name": job_name,
            "username": username,
            "started_at": now,
            "git_commit_id": None,  # Will be filled in by upload
            "package_version": harbor_package_version,  # Will be filled in by upload
            "n_trials": n_trials,
            "config": config,
            "metrics": None,
            "stats": None,
            "agent_id": agent["id"],
            "model_id": model["id"],
            "benchmark_id": benchmark["id"],
            "n_rep_eval": n_rep_eval,
            "hf_traces_link": None,  # Will be filled in by upload
            "job_status": "Started",
        }

        resp = client.table('sandbox_jobs').insert(job_data).execute()

        if not resp.data:
            return {"success": False, "error": "Failed to insert job row"}

        return {"success": True, "job": resp.data[0]}

    except Exception as e:
        logger.error(f"Failed to create job entry: {e}")
        return {"success": False, "error": str(e)}


def get_latest_job_for_model_benchmark(
    model_hf_name: str,
    benchmark_hf_name: str
) -> Optional[Dict]:
    """Get the most recent job for a model+benchmark combination."""
    try:
        model = get_model_by_name(model_hf_name)
        if not model:
            return None

        benchmark_name = benchmark_hf_name.split("/")[-1]
        benchmark = get_benchmark_by_name(benchmark_name)
        if not benchmark:
            return None

        client = get_supabase_client()
        resp = client.table('sandbox_jobs')\
            .select('*')\
            .eq('model_id', model["id"])\
            .eq('benchmark_id', benchmark["id"])\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()

        return resp.data[0] if resp.data else None

    except Exception as e:
        logger.error(f"Failed to get latest job: {e}")
        return None


def upload_eval_results(
    job_dir: Union[str, Path],
    username: str,
    error_mode: str,
    agent_name: Optional[str] = None,
    agent_version: Optional[str] = None,
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    benchmark_version_hash: Optional[str] = None,
    git_commit_id: Optional[str] = None,
    register_benchmark: bool = False,
    hf_repo_id: Optional[str] = None,
    hf_private: bool = False,
    hf_token: Optional[str] = None,
    hf_episodes: str = "last",
    hf_success_filter: Optional[str] = None,
    hf_verbose: bool = False,
    forced_update: bool = False,
) -> Dict[str, Any]:
    """
    Upload evaluation results from a job directory to HuggingFace and database.

    This function orchestrates the complete upload workflow:
    1. upload_traces_to_hf() - Upload trial traces to HuggingFace (if hf_repo_id provided)
    2. upload_job_and_trial_records() - Upload DB records with HF dataset URL

    Args:
        job_dir: Path to job directory containing config.json, result.json, and trial folders
        username: Username for job registration
        error_mode: Error handling mode (REQUIRED):
            - "rollback_on_error": Delete all job/trial/usage records on any error (atomic)
            - "skip_on_error": Continue uploading even if individual trials fail (best-effort)
        agent_name: Agent name (auto-detected from trial if not provided)
        agent_version: Agent version (auto-detected from trial if not provided)
        model_name: Model name (auto-detected from trial if not provided)
        benchmark_name: Benchmark name (auto-detected from dataset path if not provided)
        benchmark_version_hash: Benchmark version hash (auto-detected from dataset path if not provided)
        git_commit_id: Git commit ID (optional, uses package_version if not provided)
        register_benchmark: If True and benchmark not found, auto-register benchmark and tasks from job (default False)
        hf_repo_id: HuggingFace repository ID for traces upload (e.g., 'username/dataset-name'). If None, skips HF upload.
        hf_private: Whether to create a private HF repository (default: False)
        hf_token: HuggingFace API token for authentication (default: None, uses HF_TOKEN env var)
        hf_episodes: "all" or "last" - which episodes to export per trial (default: "last")
        hf_success_filter: Filter trials by success ("success", "failure", or None)
        hf_verbose: Enable verbose logging for HF upload (default: False)
        forced_update: If True, allow updating existing job records (default: False)

    Returns:
        Dict with summary (fields depend on error_mode):

        rollback_on_error mode:
        {
            "success": bool,  # False if any trial fails
            "job_id": UUID or None,  # None if rolled back
            "n_trials_uploaded": int,  # 0 if rolled back
            "error": str,  # Error message if failed
            "failed_trial": Dict,  # Details of failed trial
            "hf_dataset_url": str or None,  # HF dataset URL if uploaded
            "hf_upload_error": str or None,  # HF upload error if occurred
            ...
        }

        skip_on_error mode:
        {
            "success": bool,  # True if ANY trials uploaded successfully
            "job_id": UUID,  # Job always kept in DB
            "n_trials_total": int,
            "n_trials_uploaded": int,
            "n_trials_failed": int,
            "failed_trials": List[Dict],  # List of failed trial details
            "job": Dict,
            "trials": List[Dict],  # Successfully uploaded trials
            "hf_dataset_url": str or None,  # HF dataset URL if uploaded
            "hf_upload_error": str or None,  # HF upload error if occurred
        }

    Raises:
        ValueError: If job not finished, required FKs not found, or invalid error_mode
        FileNotFoundError: If job directory or config files not found
    """
    # Validate error_mode
    if error_mode not in ("rollback_on_error", "skip_on_error"):
        raise ValueError(f"Invalid error_mode: {error_mode}. Must be 'rollback_on_error' or 'skip_on_error'")

    logger.info(f"Starting upload of evaluation results from {job_dir}")
    logger.info(f"Error mode: {error_mode}")

    # Initialize result tracking
    hf_dataset_url = None
    hf_upload_error = None
    hf_upload_attempted = bool(hf_repo_id)

    # Step 1: Upload traces to HuggingFace (if hf_repo_id provided)
    if hf_repo_id:
        logger.info(f"Step 1: Uploading traces to HuggingFace: {hf_repo_id}")
        try:
            hf_dataset_url = upload_traces_to_hf(
                job_dir=job_dir,
                hf_repo_id=hf_repo_id,
                private=hf_private,
                token=hf_token,
                episodes=hf_episodes,
                success_filter=hf_success_filter,
                verbose=hf_verbose,
            )
            logger.info(f"HuggingFace upload successful: {hf_dataset_url}")

        except Exception as e:
            hf_upload_error = str(e)
            error_msg = f"Failed to upload traces to HuggingFace: {e}"
            logger.error(error_msg)

            # Handle based on error_mode
            if error_mode == "rollback_on_error":
                # For rollback mode, return early with error details
                logger.error("Rollback mode: Aborting entire upload process due to HF upload failure")
                return {
                    "success": False,
                    "job_id": None,
                    "n_trials_uploaded": 0,
                    "error": error_msg,
                    "hf_dataset_url": None,
                    "hf_upload_error": hf_upload_error,
                    "hf_upload_attempted": True,
                    "stage_failed": "hf_upload",
                }
            else:  # skip_on_error
                # For skip mode, log warning and continue with DB upload
                logger.warning(f"Skip mode: HF upload failed but continuing with DB upload. Error: {hf_upload_error}")
                # Continue to DB upload even though HF failed
    else:
        logger.info("Step 1: Skipping HuggingFace upload (no hf_repo_id provided)")

    # Step 2: Upload DB records (job, trials, model usage) with HF dataset URL
    logger.info("Step 2: Uploading database records")

    try:
        db_result = upload_job_and_trial_records(
            job_dir=job_dir,
            username=username,
            agent_name=agent_name,
            agent_version=agent_version,
            model_name=model_name,
            benchmark_name=benchmark_name,
            benchmark_version_hash=benchmark_version_hash,
            git_commit_id=git_commit_id,
            error_mode=error_mode,
            register_benchmark=register_benchmark,
            hf_dataset_url=hf_dataset_url,  # Will be None if HF upload failed
            forced_update=forced_update,
        )

        # Add HF-related information to result
        db_result["hf_dataset_url"] = hf_dataset_url
        db_result["hf_upload_attempted"] = hf_upload_attempted

        # If HF upload was attempted but failed, include error details
        if hf_upload_error:
            db_result["hf_upload_error"] = hf_upload_error
            db_result["hf_upload_success"] = False

            # For skip_on_error mode, adjust success flag to indicate partial success
            if error_mode == "skip_on_error" and db_result.get("success"):
                db_result["partial_success"] = True
                db_result["warnings"] = db_result.get("warnings", [])
                db_result["warnings"].append(f"HuggingFace upload failed: {hf_upload_error}")
                logger.warning("Upload completed with warnings: HF upload failed but DB records uploaded successfully")
        elif hf_upload_attempted:
            db_result["hf_upload_success"] = True

        return db_result

    except Exception as e:
        # DB upload failed
        error_msg = f"Failed to upload database records: {e}"
        logger.error(error_msg)

        # If we're in rollback_on_error mode, the DB function should have already rolled back
        # Return comprehensive error information
        result = {
            "success": False,
            "error": error_msg,
            "stage_failed": "db_upload",
            "hf_dataset_url": hf_dataset_url,  # Include if HF succeeded before DB failed
            "hf_upload_attempted": hf_upload_attempted,
        }

        # Add HF error info if it also failed
        if hf_upload_error:
            result["hf_upload_error"] = hf_upload_error
            result["multiple_failures"] = True

        # Re-raise the exception after logging
        raise


def register_base_model(
    base_model_name: str,
    created_by: Optional[str] = None,
    agent_id: str = "6047d4e4-05de-4d33-867d-c4946ecfbd65",
    extra_training_parameters: Optional[Dict[str, Any]] = None,
    forced_update: bool = False,
) -> Dict[str, Any]:
    """
    Ensure a 'base model' entry exists in the models table for a Hugging Face model.
    If it already exists:
      - returns it (idempotent), unless forced_update=True, in which case certain fields are refreshed.

    Returns:
        {"success": bool, "model": dict, "exists": True}  # if found and not updated
        {"success": bool, "model": dict, "updated": True} # if updated
        {"success": bool, "model": dict}                  # if created
        {"success": False, "error": str}                  # on error
    """
    try:
        if not base_model_name or not isinstance(base_model_name, str):
            return {"success": False, "error": "base_model_name must be a non-empty string"}

        # If the model already exists, return or update depending on forced_update
        existing = get_model_by_name(base_model_name)
        now_ts = datetime.now(timezone.utc).isoformat()

        # Derive created_by if not provided
        if not created_by:
            created_by = base_model_name.split('/')[0] if '/' in base_model_name else "hf-uploader"

        # Minimal training_parameters, with ability to extend
        tp = {
            "source": "huggingface_hub",
            "registered_at": now_ts,
            "base_model": True,
        }
        if extra_training_parameters and isinstance(extra_training_parameters, dict):
            tp.update(extra_training_parameters)

        base_payload = {
            "name": base_model_name,
            "created_by": created_by,
            "creation_location": "HuggingFace",
            "creation_time": now_ts,
            "updated_at": now_ts,
            "is_external": True,
            "weights_location": f"https://huggingface.co/{base_model_name}",
            "training_status": "completed",
            "training_start": BASE_MODEL_TRAINING_START_SENTINEL,  # e.g. "1970-01-01T00:00:00Z"
            "agent_id": agent_id,
            "training_parameters": tp,
        }

        if existing:
            if not forced_update:
                return {"success": True, "model": existing, "exists": True}
            # Only refresh a safe subset on update to avoid clobbering user-managed fields
            update_fields = {
                "updated_at": now_ts,
                "weights_location": base_payload["weights_location"],
                "is_external": True,
                "training_status": "completed",
                "training_start": BASE_MODEL_TRAINING_START_SENTINEL,
            }
            # Optionally refresh training_parameters (merge)
            merged_tp = dict(existing.get("training_parameters") or {})
            merged_tp.update(tp)
            update_fields["training_parameters"] = merged_tp

            updated = update_model(existing["id"], update_fields)
            return {"success": True, "model": updated, "updated": True}

        # Create if missing
        created = create_model(base_payload)
        return {"success": True, "model": created}

    except Exception as e:
        logger.error(f"Failed to register base model {base_model_name}: {e}")
        return {"success": False, "error": str(e)}
