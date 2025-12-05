#!/usr/bin/env python3
"""
Quick harness to sanity‑check whether OpenAI (or any LiteLLM-supported) chat model
returns ``reasoning_content`` when invoked the same way Harbor's Terminus-2 agent does.

Usage example:
    python scripts/datagen/test_reasoning_content.py \
        --model openai/gpt-5-mini-2025-08-07 \
        --prompt "Summarize the benefits of zero-knowledge proofs."

Set OPENAI_API_KEY (or the relevant provider key) in your environment first. Pass
--api-base to test against non-default endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from textwrap import dedent

import litellm

try:
    from harbor.utils.traces_utils import (  # type: ignore
        _extract_reasoning_content as harbor_extract_reasoning_content,
        _format_reasoning_block as harbor_format_reasoning_block,
    )
    HARBOR_UTILS_AVAILABLE = True
except Exception:  # pragma: no cover - diagnostic script
    harbor_extract_reasoning_content = None  # type: ignore
    harbor_format_reasoning_block = None  # type: ignore
    HARBOR_UTILS_AVAILABLE = False


def _fallback_extract_reasoning_content(response: dict) -> str | None:
    """Fallback extraction when Harbor utilities aren't available.

    Mimics what Harbor's _extract_reasoning_content should do.
    """
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})

    # Try reasoning_content first (OpenAI style)
    if reasoning := message.get("reasoning_content"):
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

    # Try thinking (Anthropic style)
    if thinking := message.get("thinking"):
        if isinstance(thinking, str) and thinking.strip():
            return thinking.strip()

    # Try thinking_blocks (Anthropic extended thinking)
    if thinking_blocks := message.get("thinking_blocks"):
        if isinstance(thinking_blocks, list):
            collected = []
            for block in thinking_blocks:
                if isinstance(block, dict) and (text := block.get("thinking")):
                    if isinstance(text, str) and text.strip():
                        collected.append(text.strip())
            if collected:
                return "\n\n".join(collected)

    return None


def _fallback_format_reasoning_block(reasoning: str | None) -> str:
    """Fallback formatting when Harbor utilities aren't available."""
    if not reasoning:
        return ""
    return f"<think>{reasoning}</think>"


def _build_messages(user_prompt: str, parser: str) -> list[dict]:
    """Reproduce the Terminus-2 style system/user message layout."""
    system_template = dedent(
        """\
        You are an AI assistant executing terminal-style tasks. Respond using the {parser}
        format unless instructed otherwise. Provide concise, actionable plans.
        """
    ).strip()
    return [
        {"role": "system", "content": system_template.format(parser=parser)},
        {"role": "user", "content": user_prompt},
    ]


async def _call_litellm_direct(args: argparse.Namespace) -> dict:
    """Issue a single chat completion mirroring Harbor's LiteLLM invocation."""
    messages = _build_messages(args.prompt, args.parser)

    completion_kwargs: dict = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "drop_params": True,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.api_base:
        completion_kwargs["api_base"] = args.api_base
    if args.session_id:
        completion_kwargs.setdefault("extra_body", {})["session_id"] = args.session_id

    response = await litellm.acompletion(**completion_kwargs)
    return response


async def _call_via_harbor_litellm(args: argparse.Namespace) -> dict:
    """Route the request through Harbor's LiteLLM wrapper for apples-to-apples debugging."""
    try:
        from harbor.llms.lite_llm import LiteLLM  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Unable to import harbor.llms.lite_llm; ensure Harbor is on PYTHONPATH."
        ) from exc

    messages = _build_messages(args.prompt, args.parser)
    prompt = messages[-1]["content"]
    history = messages[:-1]

    client = LiteLLM(
        model_name=args.model,
        temperature=args.temperature,
        api_base=args.api_base,
        session_id=args.session_id,
        max_thinking_tokens=args.max_thinking_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    response = await client.call(prompt=prompt, message_history=history)

    usage = response.usage
    usage_block = (
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_cost_usd": usage.cost_usd,
        }
        if usage
        else None
    )

    return {
        "model": args.model,
        "choices": [
            {
                "message": {
                    "content": response.content,
                    "reasoning_content": response.reasoning_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage_block,
    }


def _coerce_response_to_dict(response: object) -> dict:
    """Normalize litellm responses (ModelResponse) to a plain dict for logging."""
    if isinstance(response, dict):
        return response
    for attr in ("model_dump", "dict"):
        if hasattr(response, attr):
            return getattr(response, attr)()
    for attr in ("model_dump_json", "json"):
        if hasattr(response, attr):
            return json.loads(getattr(response, attr)())
    # Fall back to best-effort string conversion
    return json.loads(json.dumps(response, default=str))


def _print_summary(raw_response: object) -> dict:
    response = _coerce_response_to_dict(raw_response)
    choice = response["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content")

    print("\n=== Raw response ===")
    print(json.dumps(response, indent=2))

    print("\n=== Summary ===")
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(f"contains reasoning_content: {reasoning is not None}")
    if reasoning:
        print("\n--- reasoning_content ---")
        print(reasoning)
    print("\n--- content ---")
    print(message.get("content", ""))
    return response


def _print_harbor_projection(response: dict) -> None:
    # Use Harbor utilities if available, otherwise use fallback
    if HARBOR_UTILS_AVAILABLE:
        extract_fn = harbor_extract_reasoning_content
        format_fn = harbor_format_reasoning_block
        source_label = "Harbor utilities"
    else:
        extract_fn = lambda _agent, resp: _fallback_extract_reasoning_content(resp)
        format_fn = _fallback_format_reasoning_block
        source_label = "fallback (Harbor utilities unavailable)"

    reasoning = extract_fn(None, response)
    block = format_fn(reasoning)

    choice = response["choices"][0]
    message = choice["message"]

    projected_step = {
        "step_id": 1,  # illustrative
        "source": "agent",
        "model_name": response.get("model"),
        "message": message.get("content", ""),
        "reasoning_content": reasoning,
        "reasoning_embedded": block,
    }

    print(f"\n=== Harbor trajectory projection ({source_label}) ===")
    if reasoning:
        print("Harbor would capture reasoning_content (and embed it inside <think> tags):")
        print(block)
    else:
        print("Harbor reasoning_content would be empty for this response.")
    print("\nStep payload preview:")
    print(json.dumps(projected_step, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Fully-qualified model name, e.g. openai/gpt-5-mini-2025-08-07",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "harbor"],
        nargs="+",
        default=["direct"],
        help="Which client path(s) to exercise (default: direct litellm call).",
    )
    default_prompt = (
        "You are an AI assistant tasked with solving command-line tasks in a Linux environment. "
        "You will be given a task description and the output from previously executed commands. "
        "Your goal is to solve the task by providing batches of shell commands.\n\n"
        "Format your response as JSON with the following structure:\n\n"
        "{\n"
        "  \"analysis\": \"Analyze the current state based on the terminal output provided. "
        "What do you see? What has been accomplished? What still needs to be done?\",\n"
        "  \"plan\": \"Describe your plan for the next steps. What commands will you run and why? "
        "Be specific about what you expect each command to accomplish.\",\n"
        "  \"commands\": [\n"
        "    {\n"
        "      \"keystrokes\": \"ls -la\\n\",\n"
        "      \"duration\": 0.1\n"
        "    },\n"
        "    {\n"
        "      \"keystrokes\": \"cd project\\n\",\n"
        "      \"duration\": 0.1\n"
        "    }\n"
        "  ],\n"
        "  \"task_complete\": true\n"
        "}\n\n"
        "Required fields:\n"
        "- \"analysis\": Your analysis of the current situation\n"
        "- \"plan\": Your plan for the next steps\n"
        "- \"commands\": Array of command objects to execute\n\n"
        "Optional fields:\n"
        "- \"task_complete\": Boolean indicating if the task is complete (defaults to false if not present)\n\n"
        "Command object structure:\n"
        "- \"keystrokes\": String containing the exact keystrokes to send to the terminal (required)\n"
        "- \"duration\": Number of seconds to wait for the command to complete before the next command "
        "will be executed (defaults to 1.0 if not present)\n\n"
        "IMPORTANT: The text inside \"keystrokes\" will be used completely verbatim as keystrokes. "
        "Write commands exactly as you want them sent to the terminal:\n"
        "- Most bash commands should end with a newline (\\n) to cause them to execute\n"
        "- For special key sequences, use tmux-style escape sequences:\n"
        "  - C-c for Ctrl+C\n"
        "  - C-d for Ctrl+D\n\n"
        "The \"duration\" attribute specifies the number of seconds to wait for the command to complete "
        "(default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) "
        "set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On "
        "slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as "
        "you determine necessary.\n\n"
        "It is better to set a smaller duration than a longer duration. It is always possible to wait again if "
        "the prior output has not finished, by running {\"keystrokes\": \"\", \"duration\": 10.0} on "
        "subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see "
        "intermediate result status.\n\n"
        "Important notes:\n"
        "- Each command's keystrokes are sent exactly as written to the terminal\n"
        "- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command\n"
        "- Extra text before or after the JSON will generate warnings but be tolerated\n"
        "- The JSON must be valid - use proper escaping for quotes and special characters within strings\n"
        "- Commands array can be empty if you want to wait without taking action\n\n"
        "Task Description:\n"
        "There's a /app/filter.py script that attempts to remove JavaScript from HTML files to prevent XSS attacks. "
        "The /app/filter.py script takes an HTML file as a command-line argument and modifies it in-place.\n\n"
        "Usage: python /app/filter.py <html_file>\n\n"
        "Your task is to create a file called /app/out.html that, even after being processed by /app/filter.py, "
        "will still trigger a JavaScript alert() when opened in a browser. Your file should not require any "
        "interaction with the webpage and should trigger automatically.\n\n"
        "You can run /app/test_outputs.py to verify.\n\n"
        "Current terminal state:\n"
        "root@container:/app#"
    )

    parser.add_argument(
        "--prompt",
        default=default_prompt,
        help="User prompt to send to the model (defaults to a Terminus-2 style system prompt + task).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature to mirror Harbor defaults.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Max completion tokens for the request.",
    )
    parser.add_argument(
        "--parser",
        default="json",
        help="Parser name to mention in the system prompt (default: json).",
    )
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=8192,
        help="Thinking budget when requesting Anthropic extended reasoning.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Reasoning effort hint passed through LiteLLM (default: medium).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (useful for gateways or Azure-style deployments).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session identifier to include in the request body.",
    )
    return parser.parse_args()


def _print_comparison(results: dict[str, dict]) -> None:
    """Print a comparison of reasoning extraction across modes."""
    if len(results) < 2:
        return

    print(f"\n{'=' * 80}")
    print("COMPARISON: reasoning_content extraction")
    print("=" * 80)

    for mode, response in results.items():
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        reasoning = message.get("reasoning_content")
        has_reasoning = reasoning is not None and (
            isinstance(reasoning, str) and reasoning.strip()
        )
        status = "✅ PRESENT" if has_reasoning else "❌ MISSING"
        print(f"  {mode.upper():10s}: {status}")
        if has_reasoning:
            preview = reasoning[:80].replace("\n", " ") + "..." if len(reasoning) > 80 else reasoning.replace("\n", " ")
            print(f"             Preview: {preview}")

    # Check for the specific bug: direct has it, harbor doesn't
    direct_resp = results.get("direct", {})
    harbor_resp = results.get("harbor", {})

    direct_reasoning = direct_resp.get("choices", [{}])[0].get("message", {}).get("reasoning_content")
    harbor_reasoning = harbor_resp.get("choices", [{}])[0].get("message", {}).get("reasoning_content")

    direct_has = direct_reasoning is not None and bool(str(direct_reasoning).strip())
    harbor_has = harbor_reasoning is not None and bool(str(harbor_reasoning).strip())

    print()
    if direct_has and not harbor_has:
        print("⚠️  DIAGNOSIS: Harbor is LOSING thinking tokens that LiteLLM provides!")
        print("   This is a bug in Harbor's reasoning extraction logic.")
        print("   The raw LiteLLM response contains reasoning_content, but Harbor's")
        print("   normalize_reasoning_from_message() fails to extract it.")
    elif direct_has and harbor_has:
        print("✅ DIAGNOSIS: Both modes correctly extract reasoning_content.")
    elif not direct_has and not harbor_has:
        print("ℹ️  DIAGNOSIS: Model did not return reasoning_content in either mode.")
        print("   This may be expected if the model doesn't support extended thinking.")
    else:
        print("🤔 DIAGNOSIS: Unexpected state - Harbor has reasoning but direct doesn't?")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set – request may fail unless using another provider.")

    args = _parse_args()
    results: dict[str, dict] = {}

    for mode in args.mode:
        print(f"\n{'=' * 80}\nMODE: {mode.upper()}\n{'=' * 80}")
        if mode == "direct":
            response = asyncio.run(_call_litellm_direct(args))
        elif mode == "harbor":
            response = asyncio.run(_call_via_harbor_litellm(args))
        else:  # pragma: no cover
            raise ValueError(f"Unknown mode: {mode}")

        response_dict = _print_summary(response)
        results[mode] = response_dict
        _print_harbor_projection(response_dict)

    # Print comparison if multiple modes were run
    if len(results) > 1:
        _print_comparison(results)


if __name__ == "__main__":
    main()
