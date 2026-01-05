"""Inference engine abstractions and implementations."""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List

DEFAULT_GEMINI_MODELS = [
    "gemini-3-pro-preview",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro",
    "gemini-pro",
]

class InferenceEngine(ABC):
    """Base class for inference engines used during data generation."""

    #: Whether ``healthcheck`` should be invoked before the engine is used.
    requires_initial_healthcheck: bool = False

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Store non-secret kwargs for metadata/debugging.
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError

    def healthcheck(self) -> bool:
        """Return whether the engine is healthy."""
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata describing this engine without sensitive fields."""
        return {
            "engine_type": self.__class__.__name__,
            "kwargs": {k: v for k, v in self.kwargs.items() if k not in ["api_key", "token"]},
        }


class OpenAIEngine(InferenceEngine):
    """OpenAI API inference engine."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._default_max_tokens = kwargs.get("max_tokens")

        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter.")

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        client = OpenAI(api_key=self.api_key)

        if self._default_max_tokens is not None and "max_tokens" not in generation_kwargs:
            generation_kwargs["max_tokens"] = self._default_max_tokens

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **generation_kwargs
        )

        return response.choices[0].message.content

    def healthcheck(self) -> bool:
        """Check OpenAI API availability."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            client.models.list()
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("OpenAI healthcheck failed: %s", exc)
            return False


class AnthropicEngine(InferenceEngine):
    """Anthropic API inference engine."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs):
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._default_max_tokens = kwargs.get("max_tokens")

        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key parameter.")

    def generate(self, prompt: str, **generation_kwargs) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required. Install with: pip install anthropic") from exc

        client = Anthropic(api_key=self.api_key)

        max_tokens = generation_kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = self._default_max_tokens if self._default_max_tokens is not None else 4096

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **generation_kwargs
        )

        return response.content[0].text

    def healthcheck(self) -> bool:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            client.models.list()
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Anthropic healthcheck failed: %s", exc)
            return False


class GenericOpenAIEngine(InferenceEngine):
    """
    Generic OpenAI-compatible API engine (e.g., vLLM, local servers).
    Supports loading endpoint config from JSON file.
    """

    requires_initial_healthcheck = True

    def __init__(
        self,
        endpoint_json: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "fake_key",
        healthcheck_interval: int = 300,
        *,
        default_headers: Optional[Dict[str, str]] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        http_request_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            endpoint_json=endpoint_json,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            healthcheck_interval=healthcheck_interval,
            default_headers=default_headers,
            client_kwargs=client_kwargs,
            http_request_kwargs=http_request_kwargs,
            **kwargs,
        )

        self.healthcheck_interval = healthcheck_interval
        self.last_healthcheck = 0
        self.last_healthcheck_status = False

        if endpoint_json:
            endpoint_path = Path(endpoint_json)
            if not endpoint_path.exists():
                raise FileNotFoundError(f"Endpoint JSON not found: {endpoint_json}")

            with open(endpoint_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)

            raw_url = config["endpoint_url"].rstrip("/")
            api_suffix = (config.get("api_path", "") or "").strip()
            if api_suffix and not api_suffix.startswith("/"):
                api_suffix = f"/{api_suffix}"
            if api_suffix:
                candidate = f"{raw_url}{api_suffix}"
            elif raw_url.endswith("/v1"):
                candidate = raw_url
            else:
                candidate = f"{raw_url}/v1"

            self.base_url = candidate
            self.model_name = config["model_name"]
            if config.get("api_key"):
                api_key = config["api_key"]
            self.logger.info("Loaded endpoint from %s: %s", endpoint_json, self.base_url)
        elif base_url and model_name:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self.base_url = base_url
            self.model_name = model_name
        else:
            raise ValueError("Must provide either endpoint_json or both base_url and model_name")

        self.api_key = api_key
        self._default_max_tokens = kwargs.get("max_tokens")
        client_kwargs = dict(client_kwargs or {})
        if default_headers:
            merged_headers = dict(default_headers)
            existing = client_kwargs.get("default_headers")
            if isinstance(existing, dict):
                merged_headers = {**existing, **merged_headers}
            client_kwargs["default_headers"] = merged_headers

        self._client_kwargs: Dict[str, Any] = {"base_url": self.base_url, "api_key": self.api_key}
        self._client_kwargs.update(client_kwargs)
        self._requests_kwargs: Dict[str, Any] = dict(http_request_kwargs or {})

    def generate(self, prompt: str, **generation_kwargs) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required. Install with: pip install openai") from exc

        current_time = time.time()
        if current_time - self.last_healthcheck > self.healthcheck_interval:
            if not self.healthcheck():
                raise RuntimeError(f"Healthcheck failed for endpoint: {self.base_url}")

        client = OpenAI(**self._client_kwargs)
        if self._default_max_tokens is not None and "max_tokens" not in generation_kwargs:
            generation_kwargs["max_tokens"] = self._default_max_tokens
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **generation_kwargs,
        )
        return response.choices[0].message.content

    def healthcheck(self) -> bool:
        self.last_healthcheck = time.time()

        def _check_with_requests():
            import requests

            url = f"{self.base_url.rstrip('/')}/models"
            response = requests.get(url, timeout=10, **self._requests_kwargs)
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:
                self.logger.warning(
                    "Healthcheck received non-JSON response from %s; treating status 200 as healthy",
                    url,
                )
                return True, []

            data = payload.get("data", [])
            available = [entry.get("id") for entry in data if isinstance(entry, dict)]
            if self.model_name in available:
                return True, available

            if available:
                self.logger.warning(
                    "Healthcheck did not find model %s in advertised list %s; treating 200 response as healthy",
                    self.model_name,
                    available,
                )
            return True, available

        try:
            from openai import OpenAI

            client = OpenAI(**self._client_kwargs)
            models = client.models.list()
            available_models = [model.id for model in getattr(models, "data", [])]
            if self.model_name in available_models:
                self.logger.info(
                    "Healthcheck passed for %s (model: %s)",
                    self.base_url,
                    self.model_name,
                )
                self.last_healthcheck_status = True
                return True

            self.logger.warning(
                "Model %s not reported by OpenAI client list; falling back to raw HTTP check",
                self.model_name,
            )
            ok, available_models = _check_with_requests()
            if ok:
                self.logger.info(
                    "Healthcheck passed for %s (model: %s)",
                    self.base_url,
                    self.model_name,
                )
                self.last_healthcheck_status = True
                return True

            self.logger.error(
                "Model %s not found in available models: %s",
                self.model_name,
                available_models,
            )
            self.last_healthcheck_status = False
            return False
        except AttributeError as exc:
            self.logger.warning(
                "Healthcheck OpenAI client AttributeError (%s); retrying via raw HTTP fallback",
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "Healthcheck via OpenAI client failed for %s: %s; attempting raw HTTP fallback",
                self.base_url,
                exc,
            )

        try:
            ok, available_models = _check_with_requests()
            if ok:
                self.logger.info(
                    "Healthcheck passed for %s (model: %s)",
                    self.base_url,
                    self.model_name,
                )
                self.last_healthcheck_status = True
                return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                "Healthcheck FAILED for %s via raw HTTP: %s",
                self.base_url,
                exc,
            )

        self.last_healthcheck_status = False
        return False


class NoOpEngine(InferenceEngine):
    """Placeholder engine used when no inference backend is required."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class GeminiOpenAIEngine(InferenceEngine):
    """Gemini engine backed by the google-genai Vertex integration."""

    requires_initial_healthcheck = False

    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        *,
        location: str = "global",
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        prompt_template: str = "{prompt}",
        prompt_variable: str = "prompt",
        system_instruction: Optional[str] = None,
        model_candidates: Optional[List[str]] = None,
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        thinking_level: Optional[str] = "low",
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            location=location,
            project_id=project_id,
            prompt_template=prompt_template,
            system_instruction=system_instruction,
            credentials_path=credentials_path,
            thinking_level=thinking_level,
            **kwargs,
        )

        try:
            from google import genai
            from google.genai import types as genai_types
            from google.api_core import exceptions as api_exceptions
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The google-genai package is required for Gemini engines. "
                "Install it via `pip install google-genai`."
            ) from exc

        self._genai = genai
        self._genai_types = genai_types
        self._api_exceptions = api_exceptions

        self.model_name = model
        self.location = location or "global"
        self.system_instruction = system_instruction
        self.prompt_template = prompt_template
        self.prompt_variable = prompt_variable
        self.prompt_kwargs = dict(prompt_kwargs or {})
        self.thinking_level = thinking_level
        self.model_sequence = self._build_model_sequence(model, model_candidates)
        self.credentials_path = (
            credentials_path
            or os.environ.get("GCS_CREDENTIALS_PATH")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        self.project_id = self._resolve_project_id(project_id)
        if not self.project_id:
            raise ValueError(
                "Gemini engine requires a project_id. Provide via engine configuration, "
                "credentials JSON, or the GCP_PROJECT/GOOGLE_CLOUD_PROJECT environment variable."
            )

        self._configure_runtime_env()
        self._client = self._genai.Client()

    def _configure_runtime_env(self) -> None:
        cred_path = self._resolve_credentials_path(self.credentials_path)
        if cred_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
        if self.project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
        if self.location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        self._credential_path = cred_path

    @staticmethod
    def _resolve_credentials_path(path_value: Optional[str]) -> Optional[Path]:
        candidate = path_value or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not candidate:
            return None
        cred_path = Path(candidate).expanduser()
        if not cred_path.exists():
            raise FileNotFoundError(f"Gemini credentials file not found: {cred_path}")
        return cred_path

    def _resolve_project_id(self, explicit_project_id: Optional[str]) -> Optional[str]:
        if explicit_project_id:
            return explicit_project_id

        credential_source = (
            self.credentials_path
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        if credential_source:
            cred_path = Path(credential_source).expanduser()
            if cred_path.exists():
                try:
                    with cred_path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    candidate = data.get("project_id")
                    if candidate:
                        return candidate
                except Exception:  # pragma: no cover - defensive
                    self.logger.warning("Unable to read project_id from %s", cred_path)

        for env_key in ("GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "PROJECT_ID"):
            value = os.environ.get(env_key)
            if value:
                return value
        return None

    @staticmethod
    def _build_model_sequence(primary: str, candidates: Optional[List[str]]) -> List[str]:
        sequence = [primary]
        for candidate in (candidates or DEFAULT_GEMINI_MODELS):
            if candidate and candidate not in sequence:
                sequence.append(candidate)
        return sequence

    def _normalize_thinking_level(self, label: Optional[str]) -> Optional[Any]:
        if not label or str(label).lower() in {"", "off", "none"}:
            return None
        normalized = str(label).strip().upper()
        try:
            return getattr(self._genai_types.ThinkingLevel, normalized)
        except AttributeError as exc:
            valid = ", ".join(level.upper() for level in ("low", "medium", "high"))
            raise ValueError(f"Unsupported thinking level '{label}'. Choose from: {valid}") from exc

    def _build_generation_config(self, **config_overrides: Any):
        config_kwargs: Dict[str, Any] = {}
        config_kwargs.update(self.prompt_kwargs)

        prompt_override = config_overrides.pop("prompt_kwargs", {})
        if prompt_override:
            if not isinstance(prompt_override, dict):
                raise TypeError("prompt_kwargs overrides must be a mapping")
            config_kwargs.update(prompt_override)

        config_kwargs.update({k: v for k, v in config_overrides.items() if v is not None})

        if "max_tokens" in config_kwargs and "max_output_tokens" not in config_kwargs:
            config_kwargs["max_output_tokens"] = config_kwargs.pop("max_tokens")

        thinking_level = config_kwargs.pop("thinking_level", None) or self.thinking_level
        system_instruction = config_kwargs.pop("system_instruction", None) or self.system_instruction

        if thinking_level:
            tl_enum = self._normalize_thinking_level(thinking_level)
            if tl_enum:
                config_kwargs["thinking_config"] = self._genai_types.ThinkingConfig(
                    thinking_level=tl_enum
                )

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if not config_kwargs:
            return None

        return self._genai_types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                text = "".join(getattr(part, "text", "") or "" for part in parts)
                if text:
                    return text.strip()
        return ""

    def generate(self, prompt: str, **generation_kwargs) -> str:
        prompt_template = generation_kwargs.pop("prompt_template", self.prompt_template)
        prompt_variable = generation_kwargs.pop("prompt_variable", self.prompt_variable)
        prompt_kwargs = generation_kwargs.pop("prompt_kwargs", {})
        if prompt_kwargs and not isinstance(prompt_kwargs, dict):
            raise TypeError("prompt_kwargs must be a mapping when provided")
        assembled_prompt = prompt_template.format(**{prompt_variable: prompt})

        config = self._build_generation_config(
            prompt_kwargs=prompt_kwargs,
            **generation_kwargs,
        )

        last_exception: Optional[Exception] = None
        for model_name in self.model_sequence:
            try:
                request_kwargs = {
                    "model": model_name,
                    "contents": assembled_prompt,
                }
                if config is not None:
                    request_kwargs["config"] = config

                response = self._client.models.generate_content(**request_kwargs)
                text = self._extract_text(response)
                if text:
                    return text
            except self._api_exceptions.NotFound as exc:
                self.logger.warning("Gemini model %s unavailable: %s", model_name, exc.message)
                last_exception = exc
                continue
            except Exception as exc:  # pragma: no cover - network dependency
                self.logger.error("Gemini request failed for %s: %s", model_name, exc)
                last_exception = exc
                continue

        raise RuntimeError(
            "Unable to generate content with any configured Gemini models."
        ) from last_exception


def create_inference_engine(engine_type: str, **kwargs: Any) -> InferenceEngine:
    """Factory function to create inference engines."""

    engine_map = {
        "openai": OpenAIEngine,
        "anthropic": AnthropicEngine,
        "vllm_local": GenericOpenAIEngine,
        "gemini_openai": GeminiOpenAIEngine,
        "google_gemini": GeminiOpenAIEngine,
        "none": NoOpEngine,
    }

    engine_class = engine_map.get(engine_type.lower())
    if not engine_class:
        raise ValueError(
            f"Unknown engine type: {engine_type}. Valid options: {list(engine_map.keys())}"
        )

    return engine_class(**kwargs)


__all__ = [
    "InferenceEngine",
    "OpenAIEngine",
    "AnthropicEngine",
    "GenericOpenAIEngine",
    "GeminiOpenAIEngine",
    "NoOpEngine",
    "create_inference_engine",
]
