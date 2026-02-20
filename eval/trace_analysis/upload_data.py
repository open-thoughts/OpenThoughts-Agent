"""
Upload Evaluation traces from HuggingFace datasets to Docent.

Usage:
    # Upload a single dataset from huggingface to Docent
    python upload_data.py --dataset DCAgent/DCAgent_dev_set_71_tasks_DCAgent_nl2bash-nl2bash-bugsseq_Qwen3-8B-maxEps32-acc

    # Upload multiple datasets
    python upload_data.py --dataset DCAgent/dataset1 DCAgent/dataset2

    # Dry run (list but don't upload)
    python upload_data.py --dataset DCAgent/dataset1 --dry-run

Data Flow:
1. Load dataset(s) from HuggingFace using the datasets library
2. Parse each row to extract trajectory steps or conversations
3. Convert to AgentRun objects with Transcript and ChatMessage structures
4. Upload to Docent in batches, creating a new collection or using an existing one
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, List, Dict

from datasets import load_dataset
from pydantic_core import to_jsonable_python

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import ChatMessage, ToolCall, parse_chat_message

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_docent_client() -> Optional[Docent]:
    """Initialize and return Docent client, or None if API key is not available."""
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        return None
    try:
        return Docent(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Docent client: {e}")
        raise


def convert_trajectory_to_agent_run(
    trajectory_data: Dict[str, Any],
    result_data: Optional[Dict[str, Any]] = None,
    source_info: Optional[str] = None,
    row_metadata: Optional[Dict[str, Any]] = None,
) -> AgentRun:
    """
    Convert a trajectory dictionary to AgentRun format for Docent.

    Args:
        trajectory_data: Dictionary with trajectory steps
        result_data: Optional dictionary with result/verification data
        source_info: Optional source identifier (dataset name, row index, etc.)
        row_metadata: Optional row-level metadata (agent, model, model_provider, date, task, etc.)

    Returns:
        AgentRun object ready for Docent upload
    """
    steps = trajectory_data.get("steps", [])
    messages: List[ChatMessage] = []

    # Process each step
    for step_idx, step in enumerate(steps):
        source = step.get("source")
        message_content = step.get("message", "")
        tool_calls = step.get("tool_calls", [])
        observation = step.get("observation", {})
        metrics = step.get("metrics", {})
        step_id = step.get("step_id", step_idx)

        step_metadata = {
            "metrics": metrics,
            "step_id": step_id,
        }

        # Map source to role
        if source == "system":
            role = "system"
        elif source == "user":
            role = "user"
        elif source in ["agent", "assistant"]:
            role = "assistant"
        else:
            logger.warning(f"Unknown source '{source}' at step {step_id}, skipping")
            continue

        message_data: Dict[str, Any] = {
            "role": role,
            "content": message_content,
            "metadata": step_metadata,
        }

        # Process tool calls for assistant messages
        if role == "assistant" and tool_calls:
            parsed_tool_calls: List[ToolCall] = []

            for tc in tool_calls:
                tool_call_id = tc.get("tool_call_id", "")
                function_name = tc.get("function_name", "")
                arguments = tc.get("arguments", {})

                # Handle arguments that may be a JSON string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse arguments JSON string: {arguments}")
                        arguments = {"raw_arguments": arguments}
                elif not isinstance(arguments, dict):
                    logger.warning(f"Arguments is not a dict or JSON string: {arguments}")
                    arguments = {}

                # Clean up problematic characters
                if isinstance(arguments, dict):
                    if "keystrokes" in arguments:
                        arguments["keystrokes"] = to_jsonable_python(arguments["keystrokes"])
                    if "command" in arguments:
                        arguments["command"] = to_jsonable_python(arguments["command"])

                tool_call = ToolCall(
                    id=tool_call_id,
                    type="function",
                    function=function_name,
                    arguments=arguments,
                    parse_error=None,
                    view=None,
                )
                parsed_tool_calls.append(tool_call)

            message_data["tool_calls"] = parsed_tool_calls

        message = parse_chat_message(message_data)
        messages.append(message)

        # Process observation results as tool messages
        if observation and "results" in observation:
            for result in observation["results"]:
                source_call_id = result.get("source_call_id", "")
                content = result.get("content", "")

                tool_message_data = {
                    "id": source_call_id,
                    "content": content,
                    "role": "tool",
                    "metadata": step_metadata,
                }
                tool_message = parse_chat_message(tool_message_data)
                messages.append(tool_message)

    # Build metadata
    row_metadata = row_metadata or {}
    agent_metadata_raw: Dict[str, Any] = {
        "source": source_info or "hf_dataset",
        "agent": row_metadata.get("agent") or trajectory_data.get("agent", {}).get("name", "unknown"),
        "model": row_metadata.get("model") or trajectory_data.get("agent", {}).get("model_name", "unknown"),
        "model_provider": row_metadata.get("model_provider"),
        "date": row_metadata.get("date"),
        "task": row_metadata.get("task"),
        "episode": row_metadata.get("episode"),
        "run_id": row_metadata.get("run_id"),
        "trial_name": row_metadata.get("trial_name"),
        "schema_version": trajectory_data.get("schema_version", "unknown"),
        "final_metrics": trajectory_data.get("final_metrics", {}),
    }
    # Remove None values
    agent_metadata_raw = {k: v for k, v in agent_metadata_raw.items() if v is not None}

    # Add result data if available
    if result_data:
        result_metadata_raw: Dict[str, Any] = {}

        result_id = result_data.get("id")
        task_name = result_data.get("task_name")
        trial_name = result_data.get("trial_name")

        if task_name:
            result_metadata_raw["task_name"] = task_name
        if result_id:
            result_metadata_raw["result_id"] = result_id
        if trial_name:
            result_metadata_raw["trial_name"] = trial_name

        agent_result = result_data.get("agent_result")
        if agent_result:
            if isinstance(agent_result, dict) and "metadata" in agent_result:
                agent_result = {k: v for k, v in agent_result.items() if k != "metadata"}
            result_metadata_raw["agent_result"] = agent_result

        verifier_result = result_data.get("verifier_result")
        if verifier_result:
            result_metadata_raw["verifier_result"] = verifier_result

        exception_info = result_data.get("exception_info")
        if exception_info:
            result_metadata_raw["exception_type"] = exception_info.get("exception_type")
        else:
            result_metadata_raw["exception_type"] = None

        if result_metadata_raw:
            agent_metadata_raw["result"] = result_metadata_raw

    # Ensure JSON-serializable
    try:
        json.dumps(agent_metadata_raw)
        agent_metadata = agent_metadata_raw
    except (TypeError, ValueError):
        agent_metadata = to_jsonable_python(agent_metadata_raw)

    # Create transcript and AgentRun
    transcript = Transcript(messages=messages)
    agent_run = AgentRun(transcripts=[transcript], metadata=agent_metadata)

    return agent_run


def extract_json_from_content(content: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from content that may contain thinking tags and other text."""
    # Handle </think> tag variations - find the last occurrence of any marker
    # and slice the content from there
    last_marker_end_pos = -1
    for marker in ["</think>", "</think", "</ think>"]:
        pos = content.rfind(marker)
        if pos != -1:
            last_marker_end_pos = max(last_marker_end_pos, pos + len(marker))

    if last_marker_end_pos != -1:
        content = content[last_marker_end_pos:].strip()

    # Try to find JSON object (use rfind to get the last JSON object in case
    # there are curly braces in preceding text)
    start = content.rfind("{")
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i, c in enumerate(content[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(content[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


# --- Helper functions for convert_conversations_to_agent_run ---

def _parse_system_prompt_with_terminal(content: str) -> tuple:
    """
    Parse a system prompt that may have terminal state appended.

    Returns:
        Tuple of (system_content, terminal_content) where terminal_content may be None
    """
    terminal_markers = ["Current terminal state:", "Current Terminal Screen:"]
    system_content = content
    terminal_content = None

    for marker in terminal_markers:
        if marker in content:
            marker_idx = content.find(marker)
            system_content = content[:marker_idx].strip()
            terminal_content = content[marker_idx:].strip()
            break

    return system_content, terminal_content


def _map_message_role(role: str) -> str:
    """Map various role names to standard roles. Logs warning for unknown roles."""
    if role == "system":
        return "system"
    elif role == "user":
        return "user"
    elif role in ["assistant", "agent"]:
        return "assistant"
    elif role == "tool":
        return "tool"
    else:
        logger.warning(f"Unknown role '{role}', defaulting to 'user'")
        return "user"


def _parse_explicit_tool_calls(tool_calls_raw: Any) -> List[ToolCall]:
    """Parse explicit tool_calls field from assistant message."""
    if isinstance(tool_calls_raw, str):
        tool_calls_raw = json.loads(tool_calls_raw)

    parsed_tool_calls: List[ToolCall] = []
    for tc in tool_calls_raw:
        # Extract arguments with fallback to nested function.arguments
        arguments = tc.get("arguments")
        if arguments is None:
            arguments = tc.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        # Extract tool call id with fallback
        tool_call_id = tc.get("id") or tc.get("tool_call_id", "")

        # Extract function name with fallbacks
        func_name = tc.get("function_name")
        if not func_name:
            func_name = tc.get("function", {}).get("name")
        if not func_name:
            func_name = tc.get("name", "")

        tool_call = ToolCall(
            id=tool_call_id,
            type="function",
            function=func_name,
            arguments=arguments,
            parse_error=None,
            view=None,
        )
        parsed_tool_calls.append(tool_call)

    return parsed_tool_calls


def _parse_tool_call_tags(content: str, msg_idx: int) -> tuple:
    """
    Parse <tool_call> tags from content.

    Returns:
        Tuple of (tool_calls, new_content_or_none) where new_content_or_none is None
        if the content should not be modified, or the new content string if it should.
    """
    parsed_tool_calls: List[ToolCall] = []

    # Extract all tool_call blocks
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_call_matches = re.findall(tool_call_pattern, content, re.DOTALL)

    # Also try without closing tag (sometimes truncated)
    if not tool_call_matches:
        tool_call_pattern_open = r'<tool_call>\s*(\{[^<]*)'
        tool_call_matches = re.findall(tool_call_pattern_open, content, re.DOTALL)

    for tc_idx, tc_json in enumerate(tool_call_matches):
        try:
            tc_data = json.loads(tc_json.strip())
            func_name = tc_data.get("name", "bash_command")
            arguments = tc_data.get("arguments", {})

            tool_call = ToolCall(
                id=f"call_{msg_idx}_{tc_idx}",
                type="function",
                function=func_name,
                arguments=arguments,
                parse_error=None,
                view=None,
            )
            parsed_tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse tool_call JSON: {tc_json}. Error: {e}")

    # Only compute cleaned content if we have tool calls
    new_content = None
    if parsed_tool_calls:
        clean_content_parts = []
        if "<think>" in content:
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_end > think_start:
                thinking = content[think_start:think_end + 8].strip()
                if thinking and thinking != "<think>\n\n</think>" and thinking != "<think></think>":
                    clean_content_parts.append(thinking)
        # Get text before first <tool_call>
        pre_tool_call = content.split("<tool_call>")[0].strip()
        # Remove <think>...</think> from pre_tool_call if already added
        if "<think>" in pre_tool_call and "</think>" in pre_tool_call:
            pre_tool_call = re.sub(r'<think>.*?</think>', '', pre_tool_call, flags=re.DOTALL).strip()
        if pre_tool_call:
            clean_content_parts.append(pre_tool_call)
        if clean_content_parts:
            new_content = "\n\n".join(clean_content_parts)

    return parsed_tool_calls, new_content


def _parse_json_commands(content: str, msg_idx: int) -> tuple:
    """
    Parse embedded JSON with commands from content.

    Returns:
        Tuple of (tool_calls, new_content_or_none) where new_content_or_none is None
        if the content should not be modified, or the new content string if it should.
    """
    parsed_json = extract_json_from_content(content)
    if not parsed_json or "commands" not in parsed_json:
        return [], None

    commands = parsed_json.get("commands", [])
    parsed_tool_calls: List[ToolCall] = []

    for cmd_idx, cmd in enumerate(commands):
        # Handle keystrokes-style commands
        if "keystrokes" in cmd:
            tool_call = ToolCall(
                id=f"call_{msg_idx}_{cmd_idx}",
                type="function",
                function="terminal",
                arguments={"keystrokes": cmd.get("keystrokes", ""), "duration": cmd.get("duration", 0.1)},
                parse_error=None,
                view=None,
            )
            parsed_tool_calls.append(tool_call)
        # Handle command-style commands
        elif "command" in cmd:
            tool_call = ToolCall(
                id=f"call_{msg_idx}_{cmd_idx}",
                type="function",
                function="bash",
                arguments={"command": cmd.get("command", "")},
                parse_error=None,
                view=None,
            )
            parsed_tool_calls.append(tool_call)

    # Only compute cleaned content if we have tool calls
    new_content = None
    if parsed_tool_calls:
        clean_content_parts = []
        # Extract thinking content
        if "<think>" in content:
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_end > think_start:
                thinking = content[think_start:think_end + 8].strip()
                if thinking and thinking != "<think>\n\n</think>" and thinking != "<think></think>":
                    clean_content_parts.append(thinking)
        # Add analysis and plan from parsed JSON if available
        if parsed_json:
            if parsed_json.get("analysis"):
                clean_content_parts.append(f"**Analysis:** {parsed_json['analysis']}")
            if parsed_json.get("plan"):
                clean_content_parts.append(f"**Plan:** {parsed_json['plan']}")
        # Only set new content if we have parts
        if clean_content_parts:
            new_content = "\n\n".join(clean_content_parts)

    return parsed_tool_calls, new_content


def _parse_tool_response_feedback(content: str) -> Dict[str, Any]:
    """
    Parse tool response content to extract error/warning feedback and terminal output.

    The content may have formats like:
    - "Previous response had warnings:\nWARNINGS: - ...\n\nNew Terminal Output:\n..."
    - "Previous response had parsing errors:\nERROR: ...\nWARNINGS: - ...\n\nPlease fix..."
    - Just terminal output without any feedback

    Returns:
        Dict with keys: 'error', 'warning', 'terminal_output', 'raw_content'
    """
    result = {
        'error': None,
        'warning': None,
        'terminal_output': None,
        'raw_content': content,
    }

    # Check for parsing error feedback
    if content.startswith("Previous response had parsing errors:"):
        # Extract error
        error_match = re.search(r'ERROR:\s*([^\n]+)', content)
        if error_match:
            result['error'] = error_match.group(1).strip()

        # Extract warnings
        warning_match = re.search(r'WARNINGS:\s*((?:- [^\n]+\n?)+)', content)
        if warning_match:
            result['warning'] = warning_match.group(1).strip()

        # The rest is the "Please fix" message - no terminal output
        return result

    # Check for warning feedback
    if content.startswith("Previous response had warnings:"):
        # Extract warnings
        warning_match = re.search(r'WARNINGS:\s*((?:- [^\n]+\n?)+)', content)
        if warning_match:
            result['warning'] = warning_match.group(1).strip()

        # Extract terminal output after "New Terminal Output:"
        terminal_match = re.search(r'New Terminal Output:\s*\n(.*)', content, re.DOTALL)
        if terminal_match:
            result['terminal_output'] = terminal_match.group(1).strip()

        return result

    # No feedback prefix - content is just terminal output
    result['terminal_output'] = content
    return result


def convert_conversations_to_agent_run(
    row: Dict[str, Any],
    source_info: Optional[str] = None,
) -> AgentRun:
    """
    Convert a conversations-format row to AgentRun format for Docent.

    The conversations format has:
    - 'conversations': list of {role, content} messages
    - Optional metadata fields: agent, model, model_provider, date, task, episode, run_id, trial_name

    Args:
        row: Dataset row with 'conversations' field
        source_info: Optional source identifier

    Returns:
        AgentRun object ready for Docent upload
    """
    conversations = row.get("conversations", [])
    messages: List[ChatMessage] = []

    # Get model name from row for AssistantMessages
    model_name = row.get("model", "unknown")

    # Track pending tool calls from previous assistant message
    pending_tool_call_ids: List[str] = []

    for msg_idx, msg in enumerate(conversations):
        # Check if first user message is actually a system prompt
        if msg_idx == 0 and msg.get("role") == "user":
            content = msg.get("content", "")
            # Detect system prompt patterns
            if content.startswith("You are ") or content.startswith("You're "):
                system_content, terminal_content = _parse_system_prompt_with_terminal(content)

                # Add system message (without terminal state)
                system_message_data = {
                    "role": "system",
                    "content": system_content,
                }
                system_message = parse_chat_message(system_message_data)
                messages.append(system_message)

                # If there was terminal content, add as initial user message showing environment state
                if terminal_content:
                    initial_state_data = {
                        "role": "user",
                        "content": terminal_content,
                    }
                    initial_state_message = parse_chat_message(initial_state_data)
                    messages.append(initial_state_message)

                continue
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # If there are pending tool calls and this is a user message,
        # it's the tool response (observation) - regardless of content markers
        if role == "user" and pending_tool_call_ids:
            # Parse feedback from the tool response content
            feedback = _parse_tool_response_feedback(content)

            # Create tool message(s) for each pending tool call
            for tool_call_id in pending_tool_call_ids:
                # Build tool message content - prefer terminal output if available
                tool_content = feedback.get('terminal_output') or feedback.get('raw_content', content)

                # Build error dict from parsed feedback - only if there's an actual error
                # Warnings alone don't constitute an error
                error_dict = None
                if feedback.get('error'):
                    error_dict = {'error': feedback['error']}
                    if feedback.get('warning'):
                        error_dict['warning'] = feedback['warning']

                tool_message_data = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                    "function": "terminal",
                }
                if error_dict:
                    tool_message_data["error"] = error_dict

                tool_message = parse_chat_message(tool_message_data)
                messages.append(tool_message)

            pending_tool_call_ids = []
            continue

        # Map roles (logs warning for unknown roles)
        role = _map_message_role(role)

        message_data: Dict[str, Any] = {
            "role": role,
            "content": content,
        }

        # Add model field to assistant messages for LLM output tracking
        if role == "assistant":
            message_data["model"] = model_name

        # Handle tool_call_id for tool messages
        if role == "tool" and "tool_call_id" in msg:
            message_data["id"] = msg["tool_call_id"]

        # Handle explicit tool_calls field for assistant messages
        if role == "assistant" and "tool_calls" in msg:
            message_data["tool_calls"] = _parse_explicit_tool_calls(msg["tool_calls"])

        # Handle <tool_call> tags in assistant content
        elif role == "assistant" and "<tool_call>" in content:
            parsed_tool_calls, new_content = _parse_tool_call_tags(content, msg_idx)

            if parsed_tool_calls:
                message_data["tool_calls"] = parsed_tool_calls
                pending_tool_call_ids = [tc.id for tc in parsed_tool_calls]
                message_data["metadata"] = {
                    "finish_reason": "tool_calls",
                    "num_tool_calls": len(parsed_tool_calls),
                }
                # Only update content if we have cleaned content
                if new_content is not None:
                    message_data["content"] = new_content

        # Handle embedded JSON with commands in assistant content
        elif role == "assistant" and "{" in content:
            parsed_tool_calls, new_content = _parse_json_commands(content, msg_idx)

            if parsed_tool_calls:
                message_data["tool_calls"] = parsed_tool_calls
                # Track tool call IDs for linking to subsequent tool responses
                pending_tool_call_ids = [tc.id for tc in parsed_tool_calls]
                # Add metadata for LLM output tracking
                message_data["metadata"] = {
                    "finish_reason": "tool_calls",
                    "num_tool_calls": len(parsed_tool_calls),
                }
                # Only update content if we have cleaned content
                if new_content is not None:
                    message_data["content"] = new_content

        message = parse_chat_message(message_data)
        messages.append(message)

    # Build metadata from row fields
    agent_metadata: Dict[str, Any] = {
        "source": source_info or "hf_dataset",
        "agent": row.get("agent", "unknown"),
        "model": row.get("model", "unknown"),
        "model_provider": row.get("model_provider"),
        "date": row.get("date"),
        "task": row.get("task"),
        "episode": row.get("episode"),
        "run_id": row.get("run_id"),
        "trial_name": row.get("trial_name"),
    }
    # Remove None values
    agent_metadata = {k: v for k, v in agent_metadata.items() if v is not None}

    # Create transcript and AgentRun
    transcript = Transcript(messages=messages)
    agent_run = AgentRun(transcripts=[transcript], metadata=agent_metadata)

    return agent_run


def load_dataset_from_hf(dataset_name: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load a dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "DCAgent/dataset_name")
        split: Dataset split to load (default: "train")

    Returns:
        List of rows from the dataset
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")

    try:
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(dataset)} rows from {dataset_name}")
        return list(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def process_dataset_row(row: Dict[str, Any], row_idx: int, dataset_name: str) -> Optional[AgentRun]:
    """
    Process a single row from a HuggingFace dataset.

    The row may contain:
    - 'trajectory': JSON string or dict with trajectory data
    - 'result': JSON string or dict with result data
    - Or direct trajectory fields

    Args:
        row: Dataset row dictionary
        row_idx: Index of the row in the dataset
        dataset_name: Name of the source dataset

    Returns:
        AgentRun object or None if processing fails
    """
    try:
        # Try to get trajectory data
        trajectory_data = None
        result_data = None

        # Check for 'trajectory' field (may be JSON string or dict)
        if "trajectory" in row:
            traj = row["trajectory"]
            if isinstance(traj, str):
                trajectory_data = json.loads(traj)
            elif isinstance(traj, dict):
                trajectory_data = traj

        # Check for 'result' field
        if "result" in row:
            res = row["result"]
            if isinstance(res, str):
                try:
                    result_data = json.loads(res)
                except json.JSONDecodeError:
                    # Result is a plain string (e.g., "AgentTimeoutError"), not JSON
                    result_data = {"status": res}
            elif isinstance(res, dict):
                result_data = res

        # If no trajectory field, check if the row itself is trajectory-like
        if trajectory_data is None and "steps" in row:
            trajectory_data = row
            # Look for result-like fields
            if any(k in row for k in ["verifier_result", "agent_result", "task_name"]):
                result_data = row

        # Check for 'conversations' format (list of role/content messages)
        if trajectory_data is None and "conversations" in row:
            source_info = f"{dataset_name}:row_{row_idx}"
            return convert_conversations_to_agent_run(row, source_info)

        if trajectory_data is None:
            logger.warning(f"Row {row_idx}: No trajectory data found")
            return None

        source_info = f"{dataset_name}:row_{row_idx}"
        # Extract row-level metadata fields
        row_metadata = {
            "agent": row.get("agent"),
            "model": row.get("model"),
            "model_provider": row.get("model_provider"),
            "date": row.get("date"),
            "task": row.get("task"),
            "episode": row.get("episode"),
            "run_id": row.get("run_id"),
            "trial_name": row.get("trial_name"),
        }
        return convert_trajectory_to_agent_run(trajectory_data, result_data, source_info, row_metadata)

    except Exception as e:
        logger.error(f"Row {row_idx}: Error processing - {e}")
        return None


def upload_to_docent(
    agent_runs: List[AgentRun],
    collection_name: str,
    collection_id: Optional[str] = None,
) -> str:
    """
    Upload agent runs to Docent.

    Args:
        agent_runs: List of AgentRun objects to upload
        collection_name: Name for the collection (used if creating new)
        collection_id: Optional existing collection ID

    Returns:
        Collection ID
    """
    client = get_docent_client()
    if not client:
        raise ValueError("DOCENT_API_KEY not set")

    # Get or create collection
    if not collection_id:
        try:
            collection_id = client.create_collection(
                name=collection_name,
                description=f"RL eval traces: {collection_name}",
            )
            logger.info(f"Created collection '{collection_name}' with ID: {collection_id}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    else:
        logger.info(f"Using existing collection ID: {collection_id}")

    # Upload agent runs
    logger.info(f"Uploading {len(agent_runs)} agent runs to collection {collection_id}")
    client.add_agent_runs(collection_id=collection_id, agent_runs=agent_runs)
    logger.info("Upload complete!")

    return collection_id


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload RL evaluation traces from HuggingFace to Docent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single dataset
  python upload_data.py --dataset DCAgent/my_eval_traces

  # Upload multiple datasets
  python upload_data.py --dataset DCAgent/dataset1 DCAgent/dataset2

  # Dry run mode
  python upload_data.py --dataset DCAgent/dataset1 --dry-run

  # Use existing collection
  python upload_data.py --dataset DCAgent/dataset1 --collection-id abc123
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        nargs="+",
        required=True,
        help="HuggingFace dataset name(s) to upload",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train)",
    )
    parser.add_argument(
        "--collection-name",
        "-n",
        default=None,
        help="Name for the Docent collection (default: dataset name)",
    )
    parser.add_argument(
        "--collection-id",
        "-c",
        default=None,
        help="Existing Docent collection ID to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and process data without uploading to Docent",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum rows to process per dataset (0 = no limit)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    all_agent_runs: List[AgentRun] = []
    errors: List[str] = []

    # Process each dataset
    for dataset_name in args.dataset:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info("=" * 60)

        try:
            rows = load_dataset_from_hf(dataset_name, args.split)
        except Exception as e:
            errors.append(f"{dataset_name}: {e}")
            continue

        # Apply row limit
        if args.max_rows > 0 and len(rows) > args.max_rows:
            logger.info(f"Limiting to {args.max_rows} rows (total available: {len(rows)})")
            rows = rows[: args.max_rows]

        # Process rows
        dataset_runs: List[AgentRun] = []
        for idx, row in enumerate(rows):
            if idx % 50 == 0:
                logger.info(f"Processing row {idx + 1}/{len(rows)}")

            agent_run = process_dataset_row(row, idx, dataset_name)
            if agent_run:
                dataset_runs.append(agent_run)

        logger.info(f"Successfully processed {len(dataset_runs)} runs from {dataset_name}")
        all_agent_runs.extend(dataset_runs)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total agent runs: {len(all_agent_runs)}")

    if errors:
        logger.warning(f"Errors encountered: {len(errors)}")
        for err in errors:
            logger.warning(f"  - {err}")

    if not all_agent_runs:
        logger.warning("No agent runs to upload")
        return 1

    # Dry run mode
    if args.dry_run:
        logger.info(f"\nDRY RUN: Would upload {len(all_agent_runs)} agent runs to Docent")
        if all_agent_runs:
            sample = all_agent_runs[0]
            logger.info(f"Sample run has {len(sample.transcripts[0].messages)} messages")
            logger.info(f"Sample metadata: {json.dumps(sample.metadata, indent=2, default=str)[:500]}")
        return 0

    # Upload to Docent
    collection_name = args.collection_name or args.dataset[0].replace("/", "_")
    try:
        collection_id = upload_to_docent(
            agent_runs=all_agent_runs,
            collection_name=collection_name,
            collection_id=args.collection_id,
        )
        logger.info(f"\nSuccess! Collection ID: {collection_id}")
        logger.info(f"Uploaded {len(all_agent_runs)} agent runs")
        return 0
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
