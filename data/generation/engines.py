"""Inference engine abstractions and implementations."""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

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


class GeminiOpenAIEngine(GenericOpenAIEngine):
    """Google Gemini endpoint exposed via the OpenAI-compatible API surface."""

    requires_initial_healthcheck = False

    def __init__(
        self,
        model: str = "gemini-1.5-flash-002",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini OpenAI-compatible engine requires GEMINI_API_KEY or GOOGLE_API_KEY to be set."
            )

        resolved_base = (
            base_url
            or os.environ.get("GEMINI_OPENAI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai"
        )

        default_headers = kwargs.pop("default_headers", None) or {}
        if "x-goog-api-key" not in default_headers:
            default_headers = {**default_headers, "x-goog-api-key": resolved_key}

        http_request_kwargs = kwargs.pop("http_request_kwargs", None) or {}
        headers = dict(http_request_kwargs.get("headers") or {})
        headers.setdefault("x-goog-api-key", resolved_key)
        http_request_kwargs["headers"] = headers

        super().__init__(
            base_url=resolved_base,
            model_name=model,
            api_key=resolved_key,
            default_headers=default_headers,
            http_request_kwargs=http_request_kwargs,
            **kwargs,
        )

    def generate(self, prompt: str, **generation_kwargs) -> str:
        raise RuntimeError(
            "NoOpEngine does not support text generation. "
            "Configure an inference backend or avoid calling generate()."
        )


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
