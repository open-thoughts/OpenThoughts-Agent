"""Base generator class providing lifecycle orchestration."""

import argparse
import json
import os
import logging
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional
import shutil

from .engines import InferenceEngine, GenericOpenAIEngine, GeminiOpenAIEngine
from .schemas import (
    GenerationContext,
    GenerationError,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    GenerationStatus,
    InferenceFailure,
    InputValidationError,
)
from .utils import (
    RuntimeEngineSettings,
    add_generation_args,
    create_engine_from_args,
    load_datagen_config,
    resolve_engine_runtime,
)
from scripts.harbor.job_config_utils import (
    load_job_config,
    normalize_trajectory_kwargs,
    overwrite_agent_fields,
    set_job_metadata,
    set_local_dataset,
    update_agent_kwargs,
)


MODEL_INFO_DEFAULT_COST_PER_TOKEN = 1e-6


class BaseDataGenerator(ABC):
    """Abstract base class for data generation scripts."""

    HEALTHCHECK_MAX_ATTEMPTS: int = 20
    HEALTHCHECK_RETRY_DELAY: int = 30

    parser_description: str = "Data generation script"
    default_target_repo: Optional[str] = None
    default_input_dir: Optional[str] = None
    default_engine: Optional[str] = None

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.parser = argparse.ArgumentParser(
            description=self.parser_description,
            conflict_handler="resolve",
        )
        add_generation_args(
            self.parser,
            default_target_repo=self.default_target_repo,
            default_input_dir=self.default_input_dir,
            default_engine=self.default_engine,
        )
        self.parser.add_argument(
            "--stage",
            type=str,
            choices=["tasks", "traces", "both"],
            default="tasks",
            help="Generation stage to execute",
        )
        self.add_arguments(self.parser)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Hook for subclasses to extend the argument parser."""
        return None

    def parse_args(self, argv: Optional[Any] = None) -> argparse.Namespace:
        args = self.parser.parse_args(argv)

        if getattr(args, "target_repo", None) is None and self.default_target_repo:
            args.target_repo = self.default_target_repo
        if getattr(args, "input_dir", None) is None and self.default_input_dir:
            args.input_dir = self.default_input_dir

        config_path = getattr(args, "engine_config", None)
        if not config_path:
            env_path = os.environ.get("DATAGEN_CONFIG_PATH")
            if env_path:
                args.engine_config = env_path
                config_path = env_path
        if not config_path:
            self.parser.error(
                "No engine configuration provided. Set --engine-config or export DATAGEN_CONFIG_PATH."
            )

        loaded = load_datagen_config(config_path)
        runtime = resolve_engine_runtime(loaded.config)

        setattr(args, "_datagen_config", loaded.config)
        setattr(args, "_datagen_config_raw", loaded.raw)
        setattr(args, "_datagen_config_path", str(loaded.path))
        setattr(args, "_engine_runtime", runtime)
        setattr(args, "engine_request_params", dict(runtime.request_params))
        extra_agent_kwargs = dict(getattr(loaded.config, "extra_agent_kwargs", {}) or {})
        setattr(args, "_datagen_extra_agent_kwargs", extra_agent_kwargs)
        if getattr(args, "trace_agent_kwargs", None) is None and extra_agent_kwargs:
            setattr(args, "trace_agent_kwargs", extra_agent_kwargs)

        endpoint_override = getattr(args, "endpoint_json", None) or os.environ.get("TRACE_ENDPOINT_JSON")
        if runtime:
            existing_endpoint = runtime.engine_kwargs.get("endpoint_json")
            if endpoint_override:
                runtime.engine_kwargs["endpoint_json"] = endpoint_override
                setattr(args, "endpoint_json", endpoint_override)
            elif existing_endpoint:
                setattr(args, "endpoint_json", existing_endpoint)

        return args

    def build_request(self, args: argparse.Namespace) -> GenerationRequest:
        metadata = self._build_request_metadata(args)
        endpoint_json = getattr(args, "endpoint_json", None) or os.environ.get("TRACE_ENDPOINT_JSON")
        runtime: Optional[RuntimeEngineSettings] = getattr(args, "_engine_runtime", None)
        if not endpoint_json and runtime:
            endpoint_json = runtime.engine_kwargs.get("endpoint_json")
        if endpoint_json:
            metadata.setdefault("trace_endpoint_json", endpoint_json)

        limit_value = getattr(args, "limit", None)
        if limit_value is not None:
            metadata.setdefault("limit", limit_value)
        engine_value = runtime.type if runtime else None
        if engine_value and engine_value.lower() == "none":
            engine_value = None
            engine_kwargs: Dict[str, Any] = {}
        else:
            engine_kwargs = runtime.engine_kwargs if runtime else {}

        if runtime and runtime.request_params:
            metadata.setdefault("engine_request_params", dict(runtime.request_params))

        config_path = getattr(args, "_datagen_config_path", getattr(args, "engine_config", None))
        if config_path:
            metadata.setdefault("engine_config_path", config_path)
        request = GenerationRequest(
            engine_type=engine_value,
            engine_kwargs=engine_kwargs,
            input_dir=getattr(args, "input_dir", None),
            target_repo=getattr(args, "target_repo", None),
            output_dir=getattr(args, "output_dir", None),
            tasks_input=getattr(args, "tasks_input", None),
            metadata=metadata,
            raw_args=args,
            limit=limit_value,
        )
        request.stage = getattr(args, "stage", "tasks")

        if runtime and runtime.type != "none" and runtime.engine_kwargs.get("model"):
            metadata.setdefault("datagen_model", runtime.engine_kwargs["model"])
        if runtime and runtime.max_output_tokens is not None:
            metadata.setdefault("max_output_tokens", runtime.max_output_tokens)

        trace_metadata_map = {
            "trace_model": getattr(args, "trace_model", None),
            "trace_agent_name": getattr(args, "trace_agent_name", None),
            "trace_jobs_dir": getattr(args, "trace_jobs_dir", None),
            "trace_n_concurrent": getattr(args, "trace_n_concurrent", None),
            "trace_agent_kwargs": getattr(args, "trace_agent_kwargs", None),
            "trace_env": getattr(args, "trace_env", None),
            "trace_episodes": getattr(args, "trace_episodes", None),
            "trace_export_filter": getattr(args, "trace_export_filter", None),
            "trace_dataset_type": getattr(args, "trace_dataset_type", None),
        }

        for key, value in trace_metadata_map.items():
            if value is not None:
                metadata[key] = value

        for override_key in ("sandbox_cpu", "sandbox_memory_gb", "sandbox_disk_gb"):
            value = getattr(args, override_key, None)
            if value is not None:
                metadata[override_key] = value

        override_env_map = {
            "sandbox_cpu": "SANDBOX_CPU",
            "sandbox_memory_gb": "SANDBOX_MEMORY_GB",
            "sandbox_disk_gb": "SANDBOX_DISK_GB",
        }
        for key, env_var in override_env_map.items():
            if metadata.get(key) is not None:
                continue
            env_value = os.environ.get(env_var)
            if env_value in (None, ""):
                continue
            try:
                metadata[key] = int(env_value)
            except ValueError:
                self.logger.warning(
                    "[tasks] Ignoring invalid environment override %s=%s",
                    env_var,
                    env_value,
                )

        return request

    def _build_request_metadata(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Hook for subclasses to include additional metadata in the request."""

        return {}

    def _normalize_agent_kwargs_dict(
        self,
        source: Optional[dict[str, Any]],
        *,
        context: str,
    ) -> dict[str, Any]:
        """Normalise agent kwargs to the Harbor ``trajectory_config`` schema."""

        if source is None:
            return {}

        try:
            return normalize_trajectory_kwargs(source)
        except ValueError as exc:
            raise InputValidationError(f"{context}: {exc}", cause=exc) from exc

    def _inject_model_info_defaults(
        self,
        agent_kwargs: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Ensure LiteLLM receives explicit model metadata from datagen configs."""

        defaults = self._derive_model_info_defaults(args)
        if not defaults:
            return agent_kwargs

        merged = dict(agent_kwargs)
        model_info = dict(merged.get("model_info") or {})
        changed = False
        for key, value in defaults.items():
            if value is None or key in model_info:
                continue
            model_info[key] = value
            changed = True
        if changed:
            merged["model_info"] = model_info
        return merged

    def _derive_model_info_defaults(self, args: argparse.Namespace) -> dict[str, Any]:
        """Derive context window + token pricing hints for LiteLLM."""

        defaults: dict[str, Any] = {}
        context_window = self._derive_context_window(args)
        if context_window:
            defaults["max_input_tokens"] = context_window

        max_output = self._derive_max_output_tokens(args)
        if max_output:
            defaults["max_output_tokens"] = max_output

        token_costs = self._derive_token_costs(args)
        defaults.update(token_costs)
        return {k: v for k, v in defaults.items() if v is not None}

    def _derive_context_window(self, args: argparse.Namespace) -> Optional[int]:
        """Best-effort context window from datagen YAML (max_model_len, etc.)."""

        datagen_config = getattr(args, "_datagen_config", None)
        candidates: list[int] = []

        if datagen_config is not None:
            vllm_cfg = getattr(datagen_config, "vllm_server", None)
            if vllm_cfg and getattr(vllm_cfg, "max_model_len", None):
                value = self._safe_int(getattr(vllm_cfg, "max_model_len"))
                if value:
                    candidates.append(value)

            engine_cfg = getattr(datagen_config, "engine", None)
            if engine_cfg:
                for key in ("max_model_len", "max_input_tokens", "context_window"):
                    value = self._safe_int(getattr(engine_cfg, key, None))
                    if value:
                        candidates.append(value)
                if isinstance(engine_cfg.request_params, dict):
                    for key in ("max_input_tokens", "context_window", "max_context_tokens"):
                        value = self._safe_int(engine_cfg.request_params.get(key))
                        if value:
                            candidates.append(value)

        runtime = getattr(args, "_engine_runtime", None)
        if runtime and isinstance(runtime.request_params, dict):
            for key in ("max_input_tokens", "context_window", "max_context_tokens"):
                value = self._safe_int(runtime.request_params.get(key))
                if value:
                    candidates.append(value)

        if runtime and runtime.max_output_tokens:
            # Some configs only specify max output; assume 4x as context window fallback.
            candidates.append(int(runtime.max_output_tokens) * 4)

        return max(candidates) if candidates else None

    def _derive_max_output_tokens(self, args: argparse.Namespace) -> Optional[int]:
        runtime = getattr(args, "_engine_runtime", None)
        if runtime and runtime.max_output_tokens:
            return self._safe_int(runtime.max_output_tokens)

        datagen_config = getattr(args, "_datagen_config", None)
        if datagen_config and getattr(datagen_config, "engine", None):
            value = self._safe_int(getattr(datagen_config.engine, "max_output_tokens", None))
            if value:
                return value
        return None

    def _derive_token_costs(self, args: argparse.Namespace) -> dict[str, float]:
        """Return a non-zero invented cost so LiteLLM trusts the metadata."""

        cost = MODEL_INFO_DEFAULT_COST_PER_TOKEN
        return {
            "input_cost_per_token": cost,
            "output_cost_per_token": cost,
        }

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value in (None, "", False):
            return None
        try:
            intval = int(value)
        except (TypeError, ValueError):
            return None
        return intval if intval > 0 else None

    def _initialise_engine(
        self,
        args: argparse.Namespace,
        request: GenerationRequest,
    ) -> Optional[InferenceEngine]:
        engine_name = request.engine_type or ""
        engine_name_normalized = engine_name.lower() if engine_name else ""
        if not engine_name_normalized or engine_name_normalized == "none":
            return None

        try:
            engine = create_engine_from_args(args)
            if engine is None:
                return None
        except Exception as exc:  # pragma: no cover - defensive
            raise InputValidationError(
                f"Failed to initialize inference engine: {exc}",
                cause=exc,
            ) from exc

        if getattr(engine, "requires_initial_healthcheck", False):
            self.logger.info("Performing initial healthcheck for inference engine")
            attempts = getattr(self, 'HEALTHCHECK_MAX_ATTEMPTS', 20)
            delay = getattr(self, 'HEALTHCHECK_RETRY_DELAY', 30)
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    healthy = engine.healthcheck()
                except Exception as exc:  # pragma: no cover - defensive
                    healthy = False
                    last_exc = exc
                    self.logger.warning(
                        "Healthcheck attempt %d/%d raised %s",
                        attempt,
                        attempts,
                        exc.__class__.__name__,
                    )
                if healthy:
                    self.logger.info("Initial healthcheck succeeded")
                    break
                if attempt < attempts:
                    self.logger.warning(
                        "Healthcheck attempt %d/%d failed; retrying in %s seconds",
                        attempt,
                        attempts,
                        delay,
                    )
                    time.sleep(delay)
            else:
                if last_exc is not None:
                    raise InferenceFailure(
                        "Inference engine healthcheck raised an exception",
                        cause=last_exc,
                    )
                raise InferenceFailure("Initial inference engine healthcheck failed")

        return engine

    def build_context(
        self,
        args: argparse.Namespace,
        request: GenerationRequest,
        engine: Optional[InferenceEngine],
    ) -> GenerationContext:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.__class__.__name__.lower()}_"))
        return GenerationContext(
            args=args,
            request=request,
            engine=engine,
            temp_dir=temp_dir,
            logger=self.logger,
            stage=request.stage,
        )

    def requires_engine_for_stage(self, stage: str) -> bool:
        if stage in {"traces", "both"}:
            return True
        if stage == "tasks":
            return bool(getattr(self.run_task_generation, "_gpu_required", False))
        return False

    def run(self, argv: Optional[Any] = None) -> GenerationResult:
        args = self.parse_args(argv)
        request = self.build_request(args)
        stage = getattr(args, "stage", "tasks")

        engine: Optional[InferenceEngine] = None
        if self.requires_engine_for_stage(stage):
            engine = self._initialise_engine(args, request)

        context = self.build_context(args, request, engine)

        results: Dict[str, GenerationResult] = {}

        try:
            if stage in {"tasks", "both"}:
                task_result = self.run_task_generation(request, context)
                normalised_task_result = self._normalise_result(task_result)
                normalised_task_result = self._apply_task_limit(
                    request, normalised_task_result
                )
                self._apply_environment_overrides(request, normalised_task_result)
                results["tasks"] = normalised_task_result
                request.tasks_input = normalised_task_result.dataset_path

            if stage in {"traces", "both"}:
                task_result = results.get("tasks")
                if task_result:
                    request.tasks_input = task_result.dataset_path
                trace_result = self.run_trace_generation(
                    request, context, task_result
                )
                results["traces"] = self._normalise_result(trace_result)

            if stage == "traces":
                return results["traces"]
            if stage == "both":
                return results.get("traces") or results.get("tasks")
            return results["tasks"]
        except GenerationError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise GenerationError(
                f"Unhandled exception while running {self.__class__.__name__}",
                stage=GenerationStage.GENERATE,
                cause=exc,
            ) from exc

    def _normalise_result(self, result: Any) -> GenerationResult:
        if isinstance(result, GenerationResult):
            return result
        if isinstance(result, str):
            return GenerationResult(dataset_path=result)
        raise GenerationError(
            "run_generation must return a GenerationResult or dataset path",
            stage=GenerationStage.GENERATE,
        )

    def _resolve_limit(self, request: GenerationRequest) -> Optional[int]:
        limit = getattr(request, "limit", None)
        if isinstance(limit, int):
            return limit

        meta_limit = (request.metadata or {}).get("limit")
        if isinstance(meta_limit, int):
            return meta_limit
        if isinstance(meta_limit, str):
            try:
                return int(meta_limit)
            except ValueError:
                pass

        if request.raw_args is not None:
            raw_limit = getattr(request.raw_args, "limit", None)
            if isinstance(raw_limit, int):
                return raw_limit
            if isinstance(raw_limit, str):
                try:
                    return int(raw_limit)
                except ValueError:
                    pass
        return None

    def _apply_task_limit(
        self, request: GenerationRequest, result: GenerationResult
    ) -> GenerationResult:
        limit = self._resolve_limit(request)
        if not limit or limit <= 0:
            return result

        from .limits import limit_tasks_directory

        limited_path_str = limit_tasks_directory(
            result.dataset_path,
            limit,
            dataset_prefix=self.__class__.__name__.lower(),
        )
        limited_path = Path(limited_path_str)
        original_path = Path(result.dataset_path)
        limit_applied = limited_path.resolve() != original_path.resolve()

        final_path = limited_path
        output_dir_value = getattr(request, "output_dir", None)
        if output_dir_value:
            target_output = Path(output_dir_value).resolve()
            limited_resolved = limited_path.resolve()
            if limited_resolved != target_output:
                target_output.mkdir(parents=True, exist_ok=True)
                for existing in target_output.iterdir():
                    if existing.is_dir():
                        shutil.rmtree(existing)
                    else:
                        existing.unlink()
                for item in limited_resolved.iterdir():
                    shutil.move(str(item), str(target_output / item.name))
                shutil.rmtree(limited_resolved, ignore_errors=True)
                final_path = target_output
            else:
                final_path = target_output
        final_path_str = str(final_path)

        if limit_applied or Path(final_path_str) != Path(result.dataset_path):
            artifacts = dict(result.artifacts)
            artifacts.setdefault("original_dataset_path", result.dataset_path)
            artifacts["limit"] = limit
            result = replace(result, dataset_path=final_path_str, artifacts=artifacts)

        request.limit = limit
        request.metadata.setdefault("limit", limit)
        return result

    def _apply_environment_overrides(
        self, request: GenerationRequest, result: GenerationResult
    ) -> None:
        overrides = {}
        for key in ("sandbox_cpu", "sandbox_memory_gb", "sandbox_disk_gb"):
            value = request.metadata.get(key)
            if value is not None:
                overrides[key] = int(value)

        if not overrides:
            return

        dataset_path = Path(result.dataset_path)
        if not dataset_path.exists():
            self.logger.warning(
                "[tasks] Skipping sandbox overrides; dataset path missing: %s",
                dataset_path,
            )
            return

        try:
            from harbor.models.task.config import TaskConfig
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "Harbor is required to apply sandbox overrides during task generation. "
                "Install it via `pip install harbor`."
            ) from exc

        memory_override_mb = (
            overrides.get("sandbox_memory_gb") * 1024
            if overrides.get("sandbox_memory_gb") is not None
            else None
        )
        storage_override_mb = (
            overrides.get("sandbox_disk_gb") * 1024
            if overrides.get("sandbox_disk_gb") is not None
            else None
        )

        modified = 0
        for task_dir in sorted(dataset_path.iterdir()):
            if not task_dir.is_dir():
                continue
            config_path = task_dir / "task.toml"
            if not config_path.exists():
                continue
            try:
                config = TaskConfig.model_validate_toml(config_path.read_text())
            except Exception as exc:
                self.logger.warning(
                    "[tasks] Failed to load %s for sandbox overrides (%s)",
                    config_path,
                    exc,
                )
                continue

            env_cfg = config.environment
            if overrides.get("sandbox_cpu") is not None:
                env_cfg.cpus = int(overrides["sandbox_cpu"])
            if memory_override_mb is not None:
                env_cfg.memory_mb = int(memory_override_mb)
            if storage_override_mb is not None:
                env_cfg.storage_mb = int(storage_override_mb)

            config_path.write_text(config.model_dump_toml(), encoding="utf-8")
            modified += 1

        if modified:
            result.metadata.setdefault("environment_overrides_applied", {}).update(
                {k: overrides[k] for k in overrides}
            )
            self.logger.info(
                "[tasks] Applied sandbox overrides to %s tasks under %s",
                modified,
                dataset_path,
            )

    def _maybe_limit_tasks_path(
        self, request: GenerationRequest, tasks_path: Path, *, prefix: str
    ) -> Path:
        limit = self._resolve_limit(request)
        if not limit or limit <= 0:
            return tasks_path

        from .limits import limit_tasks_directory

        limited_path = limit_tasks_directory(tasks_path, limit, dataset_prefix=prefix)
        request.limit = limit
        request.metadata.setdefault("limit", limit)
        return Path(limited_path)

    @abstractmethod
    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        """Implement task generation logic (stages 1-3)."""

        raise NotImplementedError

    def run_trace_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
        task_result: Optional[GenerationResult],
    ) -> GenerationResult:
        """Default trace-generation flow (stages 4-5)."""

        from scripts.harbor.run_and_export_traces import run_dataset_to_traces
        from data.commons import upload_traces_to_hf
        from harbor.models.agent.name import AgentName
        from harbor.models.environment_type import EnvironmentType

        TRACE_MODEL_PLACEHOLDER = "placeholder/override-at-runtime"

        args = context.args
        metadata = request.metadata or {}
        engine = context.engine
        disable_verification = bool(getattr(args, "disable_verification", False))
        trace_backend_raw = metadata.get("trace_backend") or getattr(args, "trace_backend", None)
        trace_backend = (
            str(trace_backend_raw).strip().lower()
            if trace_backend_raw not in (None, "")
            else None
        )

        tasks_input = request.tasks_input or getattr(args, "tasks_input", None)
        if not tasks_input:
            raise InputValidationError(
                "Trace generation requires --tasks-input or a tasks_input in the request"
            )

        tasks_path = Path(tasks_input)
        tasks_path = self._maybe_limit_tasks_path(request, tasks_path, prefix="trace_tasks")
        request.tasks_input = str(tasks_path)
        if not tasks_path.exists():
            raise InputValidationError(
                f"Trace input tasks path does not exist: {tasks_path}"
            )

        trace_target_repo = request.target_repo
        if not trace_target_repo:
            raise InputValidationError(
                "Trace generation requires --trace-target-repo/--target-repo"
            )

        trace_output_dir = request.output_dir
        if trace_output_dir:
            trace_output_dir = Path(trace_output_dir)

        trace_config_path_raw = (
            metadata.get("trace_harbor_config")
            or getattr(args, "trace_harbor_config", None)
        )
        if not trace_config_path_raw:
            raise InputValidationError(
                "Trace generation requires a Harbor job config (--trace-harbor-config or metadata.trace_harbor_config)."
            )
        trace_config_path = Path(trace_config_path_raw).expanduser().resolve()
        if not trace_config_path.exists():
            raise InputValidationError(
                f"Trace Harbor config not found: {trace_config_path}"
            )
        trace_config_template = load_job_config(trace_config_path)

        trace_model: Optional[str] = None
        served_model: Optional[str] = None
        api_base: Optional[str] = None
        is_openai_like_engine = isinstance(
            engine, (GenericOpenAIEngine, GeminiOpenAIEngine)
        )
        requires_endpoint = isinstance(engine, GenericOpenAIEngine) and not isinstance(
            engine, GeminiOpenAIEngine
        )

        if is_openai_like_engine:
            print(
                "[traces] engine",
                {
                    "engine": engine.__class__.__name__,
                    "engine_model": getattr(engine, "model_name", None),
                    "engine_base_url": getattr(engine, "base_url", None),
                },
            )

        if requires_endpoint:
            trace_endpoint_path = (
                getattr(args, "endpoint_json", None)
                or metadata.get("trace_endpoint_json")
                or engine.kwargs.get("endpoint_json")
            )
            if not trace_endpoint_path:
                raise InputValidationError(
                    "vllm_local traces require an --endpoint-json pointing to the running cluster"
                )

            endpoint_path = Path(trace_endpoint_path)
            if not endpoint_path.exists():
                raise InputValidationError(
                    f"Trace endpoint JSON not found: {trace_endpoint_path}"
                )

            try:
                endpoint_data = json.loads(endpoint_path.read_text())
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    f"Trace endpoint JSON is not valid JSON: {trace_endpoint_path}",
                    cause=exc,
                )

            trace_model = endpoint_data.get("model_name")
            served_model = trace_model
            raw_url = (endpoint_data.get("endpoint_url") or "").rstrip("/")
            api_path = (endpoint_data.get("api_path") or "").strip()
            if api_path:
                api_base = f"{raw_url.rstrip('/')}/{api_path.lstrip('/')}"
            else:
                api_base = raw_url

            api_base = f"{api_base.rstrip('/')}/v1"

            print(
                "[traces] endpoint",
                {
                    "trace_endpoint_json": str(endpoint_path),
                    "trace_model": trace_model,
                    "trace_api_base": api_base,
                },
            )

            if not trace_model:
                raise InputValidationError(
                    f"Trace endpoint JSON missing 'model_name': {trace_endpoint_path}"
                )
            if not api_base:
                raise InputValidationError(
                    f"Trace endpoint JSON missing 'endpoint_url': {trace_endpoint_path}"
                )
        else:
            trace_model = (
                metadata.get("trace_model")
                or getattr(args, "trace_model", None)
                or metadata.get("datagen_model")
                or getattr(args, "datagen_model", None)
                or metadata.get("vllm_model_path")
                or getattr(args, "vllm_model_path", None)
                or getattr(args, "model", None)
            )
            if trace_model and str(trace_model).strip().lower() == TRACE_MODEL_PLACEHOLDER:
                trace_model = ""
            if not trace_model and isinstance(engine, (GenericOpenAIEngine, GeminiOpenAIEngine)):
                trace_model = getattr(engine, "model_name", None)

            if isinstance(engine, GenericOpenAIEngine):
                if served_model is None:
                    served_model = getattr(engine, "model_name", None)
                base_url = getattr(engine, "base_url", None)
                if base_url and not isinstance(engine, GeminiOpenAIEngine):
                    cleaned = base_url.rstrip("/")
                    if not cleaned.endswith("/v1"):
                        cleaned = f"{cleaned}/v1"
                    api_base = cleaned
            elif isinstance(engine, GeminiOpenAIEngine):
                if served_model is None:
                    served_model = getattr(engine, "model_name", None)

        if (
            not trace_model
            and trace_config_template.agents
            and trace_config_template.agents[0].model_name
        ):
            trace_model = trace_config_template.agents[0].model_name

        if trace_model and str(trace_model).strip().lower() == TRACE_MODEL_PLACEHOLDER:
            trace_model = ""

        if not trace_model:
            raise InputValidationError("Trace generation requires a model (e.g., --trace-model)")

        trace_model_for_dispatch = trace_model

        def _normalize_model_for_provider(model_value: str, provider: str | None) -> str:
            if not model_value:
                return ""
            candidate = str(model_value).strip()
            if provider and candidate.lower().startswith(f"{provider.lower()}/"):
                return candidate
            for prefix in (
                "openai/",
                "anthropic/",
                "gemini/",
                "google_gemini/",
            ):
                if candidate.lower().startswith(prefix):
                    candidate = candidate[len(prefix):]
                    break
            if candidate.lower().startswith("models/"):
                candidate = candidate[len("models/"):]
            candidate = candidate.lstrip("/")
            if provider:
                if not candidate:
                    return ""
                return f"{provider}/{candidate}"
            return candidate

        if requires_endpoint:
            trace_model_for_dispatch = f"hosted_vllm/{trace_model}"
            print(
                "[traces] hosted_vllm routing",
                {
                    "trace_model_raw": trace_model,
                    "trace_model_dispatch": trace_model_for_dispatch,
                },
            )
        elif is_openai_like_engine:
            provider_prefix = "openai"
            if isinstance(engine, GeminiOpenAIEngine):
                provider_prefix = "gemini"
                api_base = None
            trace_model_for_dispatch = _normalize_model_for_provider(trace_model, provider_prefix)
            if not trace_model_for_dispatch:
                raise InputValidationError(
                    "Unable to derive trace model for provider. Check trace_model/engine configuration."
                )
            if trace_model_for_dispatch.lower() == TRACE_MODEL_PLACEHOLDER:
                raise InputValidationError(
                    "Trace Harbor config still references placeholder/override-at-runtime. "
                    "Ensure the engine configuration provides a concrete model."
                )

        config_agent_fallback = None
        if trace_config_template.agents:
            cfg_agent = trace_config_template.agents[0]
            config_agent_fallback = cfg_agent.name or cfg_agent.import_path

        trace_agent_name = (
            metadata.get("trace_agent_name")
            or getattr(args, "trace_agent_name", None)
            or config_agent_fallback
            or "terminus-2"
        )

        jobs_dir_override_raw = metadata.get("trace_jobs_dir") or getattr(args, "trace_jobs_dir", None)
        if jobs_dir_override_raw:
            trace_jobs_dir = Path(jobs_dir_override_raw).expanduser().resolve()
        else:
            config_jobs_dir = Path(trace_config_template.jobs_dir)
            if config_jobs_dir.is_absolute():
                trace_jobs_dir = config_jobs_dir
            else:
                trace_jobs_dir = (trace_config_path.parent / config_jobs_dir).resolve()
        trace_jobs_dir.mkdir(parents=True, exist_ok=True)

        trace_eval_only_requested = bool(
            metadata.pop("trace_eval_only", False)
            or getattr(args, "trace_eval_only", False)
        )
        if trace_eval_only_requested:
            self.logger.warning(
                "trace_eval_only is deprecated. Launch eval workloads via "
                "`python -m hpc.launch --job_type eval` instead."
            )

        trace_n_concurrent = (
            metadata.get("trace_n_concurrent")
            or getattr(args, "trace_n_concurrent", None)
        )
        if trace_n_concurrent is None:
            trace_n_concurrent = trace_config_template.orchestrator.n_concurrent_trials
        if trace_n_concurrent is None:
            trace_n_concurrent = 8
            if isinstance(engine, GenericOpenAIEngine):
                print("[traces] defaulting trace_n_concurrent to 8 for vLLM local engine")

        raw_agent_kwargs = metadata.get("trace_agent_kwargs")
        if raw_agent_kwargs is None:
            raw_agent_kwargs = getattr(args, "trace_agent_kwargs", None)

        datagen_agent_defaults = self._normalize_agent_kwargs_dict(
            dict(getattr(args, "_datagen_extra_agent_kwargs", {}) or {}),
            context="datagen agent defaults",
        )
        parsed_agent_kwargs: dict[str, Any] = {}
        if raw_agent_kwargs not in (None, "", {}):
            if isinstance(raw_agent_kwargs, str):
                try:
                    parsed_agent_kwargs = json.loads(raw_agent_kwargs)
                except json.JSONDecodeError as exc:
                    raise InputValidationError("Invalid JSON for --trace-agent-kwargs", cause=exc)
            elif isinstance(raw_agent_kwargs, dict):
                parsed_agent_kwargs = raw_agent_kwargs
            else:
                raise InputValidationError("trace agent kwargs must be a JSON object")

            if not isinstance(parsed_agent_kwargs, dict):
                raise InputValidationError("trace agent kwargs must be a JSON object")

        agent_kwargs = dict(datagen_agent_defaults)
        if parsed_agent_kwargs:
            agent_kwargs.update(parsed_agent_kwargs)

        agent_kwargs = self._normalize_agent_kwargs_dict(
            agent_kwargs,
            context="trace agent kwargs",
        )
        agent_kwargs = self._inject_model_info_defaults(agent_kwargs, args)

        def _derive_metrics_endpoint_from_api_base(base: str) -> str:
            cleaned = base.rstrip("/")
            if cleaned.endswith("/v1"):
                cleaned = cleaned[: -len("/v1")].rstrip("/")
            return f"{cleaned}/metrics"

        agent_kwargs = dict(agent_kwargs)
        if requires_endpoint:
            # Endpoint JSON is authoritative for local vLLM jobs; discard stale overrides.
            for key in ("api_base", "metrics_endpoint"):
                agent_kwargs.pop(key, None)

        derived_metrics_endpoint: Optional[str] = None
        if api_base:
            agent_kwargs["api_base"] = api_base
            derived_metrics_endpoint = _derive_metrics_endpoint_from_api_base(api_base)

        if derived_metrics_endpoint:
            agent_kwargs["metrics_endpoint"] = derived_metrics_endpoint

        print(
            "[traces] dispatch",
            {
                "trace_model": trace_model_for_dispatch,
                "trace_agent": trace_agent_name,
                "trace_jobs_dir": str(trace_jobs_dir),
                "trace_api_base": agent_kwargs.get("api_base"),
                "trace_agent_kwargs": agent_kwargs,
            },
        )

        trace_env = (
            metadata.get("trace_env")
            or getattr(args, "trace_env", None)
            or trace_config_template.environment.type
        )
        trace_episodes = metadata.get("trace_episodes") or getattr(args, "trace_episodes", None) or "last"
        trace_export_filter = metadata.get("trace_export_filter") or getattr(args, "trace_export_filter", None) or "none"
        if trace_export_filter == "none":
            trace_export_filter = None

        trace_dataset_type = metadata.get("trace_dataset_type") or getattr(args, "trace_dataset_type", None) or "SFT"
        try:
            agent_enum = AgentName(trace_agent_name)
        except ValueError:
            try:
                agent_enum = AgentName(trace_agent_name.replace("-", "_").upper())
            except ValueError as exc:
                raise InputValidationError(f"Invalid trace agent name: {trace_agent_name}", cause=exc)

        if isinstance(trace_env, EnvironmentType):
            env_enum = trace_env
        else:
            trace_env_str = str(trace_env).strip()
            try:
                env_enum = EnvironmentType(trace_env_str.lower())
            except ValueError:
                try:
                    env_enum = EnvironmentType[trace_env_str.upper()]
                except KeyError as exc:
                    raise InputValidationError(
                        f"Invalid trace environment: {trace_env}", cause=exc
                    )

        trace_job_name = getattr(request.raw_args, "job_name", None)

        job_config_for_run = set_local_dataset(trace_config_template, tasks_path)
        job_config_for_run = set_job_metadata(
            job_config_for_run,
            job_name=trace_job_name or job_config_for_run.job_name,
            jobs_dir=trace_jobs_dir,
        )
        job_config_for_run = overwrite_agent_fields(
            job_config_for_run,
            name=str(
                agent_enum.value if isinstance(agent_enum, AgentName) else agent_enum
            ),
            model_name=trace_model_for_dispatch,
        )
        job_config_for_run = update_agent_kwargs(job_config_for_run, agent_kwargs)
        if derived_metrics_endpoint:
            ac = job_config_for_run.orchestrator.adaptive_concurrency
            if ac and ac.enabled and (
                not ac.metrics_endpoint
                or "replace-with" in str(ac.metrics_endpoint)
                or str(ac.metrics_endpoint).rstrip("/") != derived_metrics_endpoint.rstrip("/")
            ):
                ac.metrics_endpoint = derived_metrics_endpoint
        job_config_for_run.environment.type = env_enum
        job_config_for_run.orchestrator.n_concurrent_trials = int(trace_n_concurrent)
        agent_preview = job_config_for_run.agents[0] if job_config_for_run.agents else None
        if agent_preview is not None:
            self.logger.info(
                "[traces] prepared agent config name=%s model=%s kwargs=%s",
                getattr(agent_preview, "name", None),
                getattr(agent_preview, "model_name", None),
                getattr(agent_preview, "kwargs", None),
            )

        sandbox_cpu = request.metadata.get("sandbox_cpu")
        sandbox_mem = request.metadata.get("sandbox_memory_gb")
        sandbox_disk = request.metadata.get("sandbox_disk_gb")
        if sandbox_cpu is not None:
            job_config_for_run.environment.override_cpus = int(sandbox_cpu)
        if sandbox_mem is not None:
            job_config_for_run.environment.override_memory_mb = int(sandbox_mem) * 1024
        if sandbox_disk is not None:
            job_config_for_run.environment.override_storage_mb = int(sandbox_disk) * 1024

        job_dir = trace_jobs_dir / job_config_for_run.job_name
        if requires_endpoint:
            existing_config_path = job_dir / "config.json"
            existing_values: dict[str, str | None] = {}
            if existing_config_path.exists():
                try:
                    existing_payload = json.loads(existing_config_path.read_text())
                    existing_agent = ((existing_payload.get("agents") or [{}])[0]) or {}
                    existing_kwargs = existing_agent.get("kwargs") or {}
                    existing_values = {
                        "api_base": (existing_kwargs.get("api_base") or "").strip() or None,
                        "metrics_endpoint": (existing_kwargs.get("metrics_endpoint") or "").strip() or None,
                    }
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.warning(
                        "[traces] Failed to inspect existing Harbor config at %s (%s)",
                        existing_config_path,
                        exc,
                    )
            desired_api_base = (agent_kwargs.get("api_base") or "").strip() or None
            desired_metrics = (agent_kwargs.get("metrics_endpoint") or "").strip() or None
            if existing_values and desired_api_base:
                old_api = existing_values.get("api_base")
                old_metrics = existing_values.get("metrics_endpoint")
                mismatch_api = bool(
                    old_api
                    and old_api.rstrip("/") != desired_api_base.rstrip("/")
                )
                mismatch_metrics = bool(
                    old_metrics
                    and desired_metrics
                    and old_metrics.rstrip("/") != desired_metrics.rstrip("/")
                )
                if mismatch_api or mismatch_metrics:
                    message = (
                        "Existing trace job config at "
                        f"{existing_config_path} still references Pinggy URL "
                        f"{old_api or '<unset>'} (metrics {old_metrics or '<unset>'}) "
                        f"but the current vLLM endpoint is {desired_api_base} "
                        f"(metrics {desired_metrics or '<unset>'}). Delete the stale "
                        "trace_jobs directory or pick a fresh --experiments_dir/--job_name "
                        "to avoid reusing mismatched Harbor configs."
                    )
                    if trace_backend == "vllm":
                        raise InputValidationError(message)
                    self.logger.warning("[traces] %s", message)

        base_artifacts = {
            "trace_jobs_dir": str(trace_jobs_dir),
            "trace_job_name": job_config_for_run.job_name,
            "trace_result_json": str(job_dir / "result.json"),
        }
        if trace_output_dir:
            base_artifacts["trace_output_dir"] = str(trace_output_dir)
        if trace_dataset_type:
            base_artifacts["trace_dataset_type"] = trace_dataset_type

        trace_dataset = None
        try:
            trace_dataset = run_dataset_to_traces(
                job_config=job_config_for_run,
                dataset_path=tasks_path,
                episodes=trace_episodes,
                export_filter=trace_export_filter,
                agent_timeout_sec=getattr(args, "trace_agent_timeout_sec", None),
                verifier_timeout_sec=getattr(args, "trace_verifier_timeout_sec", None),
                disable_verification=disable_verification,
            )
        except NotImplementedError:
            raise

        if trace_dataset is None:
            raise GenerationError("Trace export returned no dataset")

        if trace_output_dir:
            trace_output_dir.mkdir(parents=True, exist_ok=True)
            trace_dataset.save_to_disk(str(trace_output_dir))

        upload_traces_to_hf(trace_dataset, trace_target_repo, trace_dataset_type)

        dataset_location = trace_output_dir or trace_jobs_dir

        request.metadata.setdefault("trace_model", trace_model_for_dispatch)
        if served_model:
            request.metadata.setdefault("trace_model_served", served_model)
        artifacts = {
            "trace_jobs_dir": str(trace_jobs_dir),
            "trace_output_dir": str(dataset_location),
            "trace_dataset_type": trace_dataset_type,
            "trace_job_name": job_config_for_run.job_name,
            "trace_result_json": str(job_dir / "result.json"),
        }
        return GenerationResult(
            dataset_path=str(dataset_location),
            status=GenerationStatus.SUCCESS,
            uploaded_repo=trace_target_repo,
            artifacts=artifacts,
        )


__all__ = ["BaseDataGenerator"]
