"""Schemas and dataclasses for data generation workflow."""

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING


class GenerationStage(str, Enum):
    PREPARE = "prepare"
    GENERATE = "generate"
    POSTPROCESS = "postprocess"


class GenerationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class GenerationRequest:
    """Normalized set of inputs provided to a generator."""

    engine_type: Optional[str] = None
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    input_dir: Optional[str] = None
    target_repo: Optional[str] = None
    output_dir: Optional[str] = None
    tasks_input: Optional[str] = None
    stage: str = "tasks"
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_args: Optional[argparse.Namespace] = None
    limit: Optional[int] = None


@dataclass
class GenerationResult:
    """Structured output emitted by generators."""

    dataset_path: str
    status: GenerationStatus = GenerationStatus.SUCCESS
    uploaded_repo: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


if TYPE_CHECKING:  # pragma: no cover
    from .engines import InferenceEngine


@dataclass
class GenerationContext:
    """Runtime information shared throughout a generation run."""

    args: argparse.Namespace
    request: GenerationRequest
    engine: Optional["InferenceEngine"]
    temp_dir: Path
    logger: logging.Logger
    services: Dict[str, Any] = field(default_factory=dict)
    stage: str = "tasks"


class GenerationError(Exception):
    """Base exception for generator failures."""

    def __init__(self, message: str, *, stage: GenerationStage, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.stage = stage
        self.cause = cause


class InputValidationError(GenerationError):
    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, stage=GenerationStage.PREPARE, cause=cause)


class InferenceFailure(GenerationError):
    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, stage=GenerationStage.GENERATE, cause=cause)


class UploadFailure(GenerationError):
    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, stage=GenerationStage.POSTPROCESS, cause=cause)


class TraceFailure(GenerationError):
    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, stage=GenerationStage.POSTPROCESS, cause=cause)
