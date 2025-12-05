"""Utilities for TA RL dataset generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Tuple

from data.generation.engines import InferenceEngine, OpenAIEngine
from data.ta_rl_tasks.prompt import PROMPT

LOGGER = logging.getLogger(__name__)

SECTION_TEST_OPEN = "<bash_script>"
SECTION_TEST_CLOSE = "</bash_script>"
SECTION_CONFIG_OPEN = "<config_json>"
SECTION_CONFIG_CLOSE = "</config_json>"


def create_engine(*, model_name: str, engine_type: Optional[str] = None) -> InferenceEngine:
    """
    Create an inference engine for test generation.
    
    Args:
        model_name: Model identifier (e.g., 'gpt-5-mini', 'claude-3-5-sonnet-20241022')
        engine_type: Engine type ('openai', 'anthropic', etc.). If None, auto-detects from model_name.
    
    Returns:
        InferenceEngine instance
    """
    from data.generation.engines import OpenAIEngine, AnthropicEngine, create_inference_engine
    
    # Auto-detect engine type from model name if not specified
    if engine_type is None:
        if model_name.startswith("claude") or "anthropic" in model_name.lower():
            engine_type = "anthropic"
        elif model_name.startswith("gpt") or "openai" in model_name.lower():
            engine_type = "openai"
        else:
            # Default to OpenAI for backward compatibility
            engine_type = "openai"
    
    # Use the standard engine creation utility if available
    try:
        # Try using the standard engine creation (supports more engine types)
        return create_inference_engine(engine_type, model=model_name)
    except (ImportError, ValueError, AttributeError):
        # Fallback to direct engine creation for backward compatibility
        try:
            if engine_type == "openai":
                return OpenAIEngine(model=model_name)
            elif engine_type == "anthropic":
                return AnthropicEngine(model=model_name)
            else:
                raise ValueError(f"Unsupported engine type: {engine_type}")
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                f"Failed to initialize {engine_type} engine with model {model_name}. "
                f"Ensure API keys are configured and the model is available."
            ) from exc


def build_prompt(
    *,
    task_dir: Path,
    instruction: str,
    solution_script: str,
    trace_metadata: Iterable[dict],
) -> str:
    metadata_json = json.dumps(list(trace_metadata), indent=2, sort_keys=True)
    blocks = [
        "================ TASK IDENTIFIERS ================",
        f"Task directory: {task_dir.name}",
        "",
        "================ TARGET TASK ================",
        instruction.strip(),
        "",
        "================ HYDRATED SOLUTION SCRIPT ================",
        solution_script.strip(),
        "",
        "================ SOLUTION METADATA ================",
        metadata_json,
    ]
    return PROMPT + "\n\n" + "\n".join(blocks)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            lines = stripped.split("\n")
            if len(lines) > 1:
                stripped = "\n".join(lines[:-1])
    return stripped.strip()


def _extract_block(text: str, open_tag: str, close_tag: str) -> str:
    start = text.find(open_tag)
    if start == -1:
        raise ValueError(f"Missing required tag {open_tag}")
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        raise ValueError(f"Missing required closing tag {close_tag}")
    return text[start:end].strip()


def _parse_sections(raw: str) -> Tuple[str, str]:
    text = raw.replace("\r\n", "\n").strip()

    test_body = _extract_block(text, SECTION_TEST_OPEN, SECTION_TEST_CLOSE)
    config_body = _extract_block(text, SECTION_CONFIG_OPEN, SECTION_CONFIG_CLOSE)

    test_content = _strip_code_fence(test_body)
    config_content = _strip_code_fence(config_body)

    if not test_content or not config_content:
        raise ValueError("Model output produced empty file content")

    if not test_content.endswith("\n"):
        test_content += "\n"
    if not config_content.endswith("\n"):
        config_content += "\n"

    json.loads(config_content)  # validate JSON structure
    return test_content, config_content


def generate_tests(
    *,
    dest_dir: Path,
    instruction: str,
    solution_script: str,
    trace_metadata: Iterable[dict],
    engine: InferenceEngine,
) -> None:
    prompt = build_prompt(
        task_dir=dest_dir,
        instruction=instruction,
        solution_script=solution_script,
        trace_metadata=trace_metadata,
    )

    LOGGER.info("Generating Harbor tests for %s", dest_dir.name)
    response = engine.generate(prompt)
    test_sh, config_json = _parse_sections(response)

    tests_dir = dest_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test.sh").write_text(test_sh, encoding="utf-8")
    (tests_dir / "config.json").write_text(config_json, encoding="utf-8")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def write_metadata(dest_dir: Path, trace_row: dict, command_lines: int) -> dict:
    trace_meta = {
        "trace_id": trace_row.get("id") or trace_row.get("uuid"),
        "trial_name": trace_row.get("trial_name"),
        "run_id": trace_row.get("run_id"),
        "success": trace_row.get("success"),
        "episode_length": len(trace_row.get("conversations", [])),
        "command_lines": command_lines,
        "script": "solve.sh",
        "is_primary": True,
    }

    metadata_path = dest_dir / "metadata.json"
    payload = {}
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Existing metadata.json in %s is invalid; overwriting", dest_dir)
            payload = {}

    payload.setdefault("ta_rl_tasks", {})["traces"] = [trace_meta]
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return trace_meta


def commands_from_trace(row: dict) -> List[str]:
    commands: List[str] = []
    for turn in row.get("conversations", []):
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        payload = turn.get("content")
        if not isinstance(payload, str):
            continue
        try:
            block = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for step in block.get("commands", []):
            if not isinstance(step, dict):
                continue
            keystrokes = step.get("keystrokes")
            duration = step.get("duration")
            if isinstance(keystrokes, str) and keystrokes.strip():
                normalized = keystrokes.replace("\r\n", "\n").replace("\r", "\n")
                for line in normalized.split("\n"):
                    line = line.rstrip()
                    if not line:
                        continue
                    if line.startswith("C-"):
                        commands.append(f"# Control keystrokes: {line}")
                    else:
                        commands.append(line)
            elif isinstance(duration, (int, float)) and duration > 0:
                commands.append(f"sleep {duration}")
    return commands
