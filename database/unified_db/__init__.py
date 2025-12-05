"""
OT-Agents Dataset and Model Registration Package

A comprehensive registration system for managing datasets and models
with support for HuggingFace and local files.
"""

# Import main functions for easy access
from .utils import (
    # Dataset functions
    register_hf_dataset,
    register_local_parquet,
    get_dataset_by_name,
    get_dataset_by_id,
    delete_dataset_by_id,
    delete_dataset_by_name,
    # Model functions
    register_hf_model,
    register_local_model,
    get_model_by_name,
    delete_model_by_id,
    delete_model_by_name,
    create_model,
    update_model,
    # Agent functions
    register_agent,
    get_agent_by_name,
    delete_agent_by_id,
    delete_agent_by_name,
    # Benchmark functions
    register_benchmark,
    get_benchmark_by_name,
    delete_benchmark_by_id,
    delete_benchmark_by_name,
    # Sandbox Task functions
    register_sandbox_task,
    get_sandbox_task_by_checksum,
    get_sandbox_task_by_name,
    delete_sandbox_task_by_checksum,
    delete_sandbox_task_by_name,
    # Sandbox Benchmark-Task link functions
    link_benchmark_to_task,
    unlink_benchmark_from_task,
    delete_all_benchmark_task_links,
    # Sandbox Job functions
    register_sandbox_job,
    get_sandbox_job_by_id,
    get_sandbox_job_by_name,
    delete_sandbox_job_by_id,
    delete_sandbox_job_by_name,
    # Sandbox Trial functions
    register_sandbox_trial,
    get_sandbox_trial_by_id,
    get_sandbox_trial_by_name,
    delete_sandbox_trial_by_id,
    delete_sandbox_trial_by_name,
    # Sandbox Trial Model Usage functions
    register_trial_model_usage,
    get_trial_model_usage,
    delete_trial_model_usage,
    delete_all_trial_model_usage,
    # Eval Results Upload functions
    upload_eval_results,
    upload_job_and_trial_records,
    upload_traces_to_hf,
    register_benchmark_and_tasks_from_job,
)
from .models import (
    DatasetModel,
    ModelModel,
    AgentModel,
    BenchmarkModel,
    SandboxTaskModel,
    SandboxBenchmarkTaskModel,
    SandboxJobModel,
    SandboxTrialModel,
    SandboxTrialModelUsageModel
)

# Package metadata
__version__ = "1.0.0"
__author__ = "OT-Agents Team"

# Export main functions
__all__ = [
    # Dataset exports
    "register_hf_dataset",
    "register_local_parquet",
    "get_dataset_by_name",
    "get_dataset_by_id",
    "delete_dataset_by_id",
    "delete_dataset_by_name",
    "DatasetModel",
    # Model exports
    "register_hf_model",
    "register_local_model",
    "get_model_by_name",
    "delete_model_by_id",
    "delete_model_by_name",
    "create_model",
    "update_model",
    "ModelModel",
    # Agent exports
    "register_agent",
    "get_agent_by_name",
    "delete_agent_by_id",
    "delete_agent_by_name",
    "AgentModel",
    # Benchmark exports
    "register_benchmark",
    "get_benchmark_by_name",
    "delete_benchmark_by_id",
    "delete_benchmark_by_name",
    "BenchmarkModel",
    # Sandbox Task exports
    "register_sandbox_task",
    "get_sandbox_task_by_checksum",
    "get_sandbox_task_by_name",
    "delete_sandbox_task_by_checksum",
    "delete_sandbox_task_by_name",
    "SandboxTaskModel",
    # Sandbox Benchmark-Task link exports
    "link_benchmark_to_task",
    "unlink_benchmark_from_task",
    "delete_all_benchmark_task_links",
    "SandboxBenchmarkTaskModel",
    # Sandbox Job exports
    "register_sandbox_job",
    "get_sandbox_job_by_id",
    "get_sandbox_job_by_name",
    "delete_sandbox_job_by_id",
    "delete_sandbox_job_by_name",
    "SandboxJobModel",
    # Sandbox Trial exports
    "register_sandbox_trial",
    "get_sandbox_trial_by_id",
    "get_sandbox_trial_by_name",
    "delete_sandbox_trial_by_id",
    "delete_sandbox_trial_by_name",
    "SandboxTrialModel",
    # Sandbox Trial Model Usage exports
    "register_trial_model_usage",
    "get_trial_model_usage",
    "delete_trial_model_usage",
    "delete_all_trial_model_usage",
    "SandboxTrialModelUsageModel",
    # Eval Results Upload exports
    "upload_eval_results",
    "upload_job_and_trial_records",
    "upload_traces_to_hf",
    "register_benchmark_and_tasks_from_job",
]