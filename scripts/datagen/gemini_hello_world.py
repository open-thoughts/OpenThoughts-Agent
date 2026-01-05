"""Simple Gemini prompt-template smoke test using the google-genai SDK."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Iterable, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "The gemini_hello_world script requires the 'google-genai' package. "
        "Install it via `pip install google-genai`."
    ) from exc


DEFAULT_CREDENTIALS = pathlib.Path("/Users/benjaminfeuer/Documents/.gcs_datacomp.json")
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-3-pro-preview"
DEFAULT_PUBLISHER_FILTER = "publishers/google/"
DEFAULT_MAX_MODELS = 25
DEFAULT_THINKING = "low"
DEFAULT_PROMPT_TEMPLATE = "Do {animal} {activity}?"
DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful zoologist."

THINKING_LEVEL_CHOICES = ("off", "low", "medium", "high")

SampleVariables = [
    {"animal": "Eagles", "activity": "eat berries"},
    {"animal": "Coyotes", "activity": "jump"},
    {"animal": "Squirrels", "activity": "fly"},
]


def _normalize_credentials_path(path: pathlib.Path) -> pathlib.Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Credential file not found: {resolved}")
    return resolved


def _determine_project_id(path: pathlib.Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    project_id = data.get("project_id")
    if not project_id:
        raise ValueError("The provided credentials JSON does not include a project_id")
    return project_id


def _configure_vertex_env(creds_path: pathlib.Path, project_id: str, location: str) -> None:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"


def _resolve_thinking_level(label: str | None) -> Optional[genai_types.ThinkingLevel]:
    if not label or label.lower() == "off":
        return None
    normalized = label.strip().upper()
    try:
        return getattr(genai_types.ThinkingLevel, normalized)
    except AttributeError as exc:
        valid = ", ".join(level.upper() for level in THINKING_LEVEL_CHOICES if level != "off")
        raise ValueError(f"Unsupported thinking level '{label}'. Choose from: {valid}") from exc


def _build_generation_config(
    thinking_level: Optional[genai_types.ThinkingLevel],
    system_instruction: Optional[str],
) -> None:
    cfg_kwargs: dict[str, object] = {}
    if thinking_level:
        cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=thinking_level
        )
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    if not cfg_kwargs:
        return None
    return genai_types.GenerateContentConfig(**cfg_kwargs)


def _list_available_models(
    client: genai.Client,
    publisher_filter: Optional[str],
    max_results: Optional[int],
) -> None:
    print(
        f"Listing up to {max_results or 'all'} models via google-genai "
        f"(filter: {publisher_filter or 'none'})..."
    )
    try:
        response = client.models.list()
    except Exception as exc:  # pragma: no cover - network dependency
        print(f"Unable to list models: {exc}")
        return

    models_iter: Iterable = getattr(response, "models", response)
    printed = 0
    for model in models_iter:
        name = getattr(model, "name", "")
        if publisher_filter and publisher_filter not in name:
            continue
        display = getattr(model, "display_name", None) or name
        print(f"  - {name} :: {display}")
        printed += 1
        if max_results and printed >= max_results:
            break

    if printed == 0:
        print("  (no models matched the current filter)")


def _run_prompt_sequences(
    client: genai.Client,
    model_name: str,
    prompt_template: str,
    variables: Iterable[dict[str, str]],
    *,
    generation_config: Optional[genai_types.GenerateContentConfig],
) -> None:
    print(f"Running Gemini hello world using model '{model_name}'...")
    for variable_set in variables:
        assembled = prompt_template.format(**variable_set)
        response = client.models.generate_content(
            model=model_name,
            contents=assembled,
            config=generation_config,
        )
        text = getattr(response, "text", "") or ""
        if not text and hasattr(response, "candidates"):
            for candidate in getattr(response, "candidates", []):
                parts = getattr(candidate, "content", None)
                if not parts or not getattr(parts, "parts", None):
                    continue
                text = "".join(getattr(part, "text", "") for part in parts.parts)
                if text:
                    break

        print(f"Prompt: {assembled}")
        print(f"Response: {text.strip()}\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini hello-world sanity check using google-genai"
    )
    parser.add_argument(
        "--credentials",
        type=pathlib.Path,
        default=DEFAULT_CREDENTIALS,
        help="Path to the service-account JSON with Gemini Vertex access.",
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help="Vertex AI region to use with google-genai (default: global)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--publisher-filter",
        default=DEFAULT_PUBLISHER_FILTER,
        help="Substring filter applied to the fully-qualified model name during listing.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=DEFAULT_MAX_MODELS,
        help="Maximum number of models to display before running the prompt (default: 25, set 0 for all)",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format string used to assemble each prompt.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=THINKING_LEVEL_CHOICES,
        default=DEFAULT_THINKING,
        help="Thinking level for Gemini 3 runs (default: low, use 'off' to disable).",
    )
    parser.add_argument(
        "--system-instruction",
        default=DEFAULT_SYSTEM_INSTRUCTION,
        help="Optional system instruction to include in the generation config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    credentials_path = _normalize_credentials_path(args.credentials)
    project_id = _determine_project_id(credentials_path)
    _configure_vertex_env(credentials_path, project_id, args.location)

    thinking_level = _resolve_thinking_level(args.thinking_level)
    generation_config = _build_generation_config(
        thinking_level,
        args.system_instruction.strip() if args.system_instruction else None,
    )

    client = genai.Client()
    max_models = args.max_models if args.max_models and args.max_models > 0 else None
    _list_available_models(client, args.publisher_filter, max_models)
    _run_prompt_sequences(
        client,
        args.model,
        args.prompt_template,
        SampleVariables,
        generation_config=generation_config,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
